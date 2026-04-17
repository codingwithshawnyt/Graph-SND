"""Batched per-agent Gaussian MLP policies for scalable training.

At ``n = 100`` the original ``training/train_navigation.py`` runs one
separate ``nn.Linear`` per agent per layer, which means the Python-side
loop dominates wall-clock time even though the GPU is nearly idle. This
module stacks ``n_agents`` independent per-agent MLPs into a single
module whose forward pass is one ``torch.bmm`` per layer, so the whole
team is evaluated in one CUDA call.

Key invariants this file preserves
----------------------------------
1. **No parameter sharing.** Each agent still owns a distinct slice of
   every weight tensor, so gradient for agent ``i`` depends only on
   agent ``i``'s loss -- this is the same heterogeneous / IPPO setup
   the original paper and ``graphsnd.policies.GaussianMLPPolicy``
   implement.
2. **Numerical equivalence to the per-agent module.** Given two
   modules initialised so that agent ``i``'s weight slice equals the
   per-agent module's weights, the batched forward returns exactly the
   same outputs as stacking the per-agent modules.
3. **Same ``PolicyConfig``.** The same dataclass configures both, so
   higher-level code can accept either implementation.

Shapes
------
All batched forwards expect ``obs`` of shape ``(n_agents, B, obs_dim)``
where ``B`` is any batch size (number of vectorised envs, flattened
rollout steps, or a PPO minibatch). They return tensors whose leading
axis is ``n_agents`` so per-agent quantities are trivial to pluck out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from graphsnd.policies import (
    GaussianMLPPolicy,
    PolicyConfig,
    ValueMLP,
)


class BatchedLinear(nn.Module):
    """Linear layer with independent parameters for each of ``n_agents``.

    Computes ``y[i] = x[i] @ W[i] + b[i]`` using a single ``torch.bmm``
    call. ``W`` has shape ``(n_agents, in_features, out_features)`` and
    ``b`` has shape ``(n_agents, out_features)``, so different agents do
    not share parameters and the gradient w.r.t. ``W[i]`` depends only
    on the loss at agent ``i``.
    """

    def __init__(self, n_agents: int, in_features: int, out_features: int):
        super().__init__()
        self.n_agents = n_agents
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(n_agents, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.zeros(n_agents, out_features))

    def orthogonal_init(self, gain: float = 1.0) -> None:
        for i in range(self.n_agents):
            nn.init.orthogonal_(self.weight[i], gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[0] != self.n_agents or x.shape[-1] != self.in_features:
            raise ValueError(
                f"BatchedLinear expects input of shape (n_agents={self.n_agents}, "
                f"B, in_features={self.in_features}); got {tuple(x.shape)}"
            )
        return torch.bmm(x, self.weight) + self.bias.unsqueeze(1)


class BatchedGaussianMLPPolicy(nn.Module):
    """``n_agents`` independent Gaussian MLP policies stacked via ``bmm``.

    Parameterisation mirrors :class:`graphsnd.policies.GaussianMLPPolicy`:

    - state-dependent mean via two ``tanh`` hidden layers then ``tanh``
      on the output scaled by ``u_range``;
    - state-independent ``log_std`` per agent per action dim.
    """

    def __init__(
        self,
        n_agents: int,
        config: PolicyConfig,
        seed_base: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.config = config
        h1, h2 = config.hidden_sizes
        self.l1 = BatchedLinear(n_agents, config.obs_dim, h1)
        self.l2 = BatchedLinear(n_agents, h1, h2)
        self.mean_head = BatchedLinear(n_agents, h2, config.act_dim)
        self.log_std = nn.Parameter(
            torch.full((n_agents, config.act_dim), float(config.init_log_std))
        )
        self._init_weights(seed_base=seed_base)

    def _init_weights(self, seed_base: Optional[int] = None) -> None:
        """Initialise weights.

        If ``seed_base`` is provided, use ``torch.manual_seed(seed_base + i)``
        before initialising each agent's slice. This matches the
        per-agent seeding pattern used by the original training script
        (``torch.manual_seed(seed * 1000 + i)``) so the first-iter
        checkpoint of a batched run is comparable to a per-agent run at
        the same seed.
        """
        if seed_base is None:
            self.l1.orthogonal_init(gain=1.0)
            self.l2.orthogonal_init(gain=1.0)
            self.mean_head.orthogonal_init(gain=0.01)
            return
        for i in range(self.n_agents):
            torch.manual_seed(seed_base + i)
            nn.init.orthogonal_(self.l1.weight[i], gain=1.0)
            nn.init.orthogonal_(self.l2.weight[i], gain=1.0)
            nn.init.orthogonal_(self.mean_head.weight[i], gain=0.01)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """``obs: (n_agents, B, obs_dim)`` -> ``(mean, std)`` each ``(n_agents, B, act_dim)``."""
        if obs.ndim != 3 or obs.shape[0] != self.n_agents:
            raise ValueError(
                f"BatchedGaussianMLPPolicy expects obs shape (n_agents={self.n_agents}, "
                f"B, obs_dim={self.config.obs_dim}); got {tuple(obs.shape)}"
            )
        hidden = torch.tanh(self.l1(obs))
        hidden = torch.tanh(self.l2(hidden))
        raw_mean = self.mean_head(hidden)
        mean = torch.tanh(raw_mean) * self.config.u_range
        log_std = self.log_std.clamp(
            self.config.log_std_min, self.config.log_std_max
        )
        std = log_std.exp().unsqueeze(1).expand_as(mean)
        return mean, std

    def distribution(self, obs: Tensor) -> torch.distributions.Independent:
        mean, std = self(obs)
        base = torch.distributions.Normal(mean, std)
        return torch.distributions.Independent(base, 1)

    def sample(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns ``(action, log_prob, mean, std)``.

        ``action`` is clipped to ``[-u_range, u_range]`` so VMAS accepts
        it; ``log_prob`` is computed against the unclipped sample which
        is standard PPO practice.
        """
        mean, std = self(obs)
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[:-1], device=mean.device)
            return action, log_prob, mean, std
        base = torch.distributions.Normal(mean, std)
        dist = torch.distributions.Independent(base, 1)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw)
        action = raw.clamp(-self.config.u_range, self.config.u_range)
        return action, log_prob, mean, std

    def log_prob(self, obs: Tensor, action: Tensor) -> Tensor:
        mean, std = self(obs)
        base = torch.distributions.Normal(mean, std)
        dist = torch.distributions.Independent(base, 1)
        return dist.log_prob(action)

    def entropy(self, obs: Tensor) -> Tensor:
        return self.distribution(obs).entropy()

    @classmethod
    def from_per_agent_policies(
        cls, policies: Sequence[GaussianMLPPolicy]
    ) -> "BatchedGaussianMLPPolicy":
        """Stack a list of per-agent policies into a single batched module.

        Useful for evaluating Experiment-1 checkpoints (which are saved
        as per-agent states) with the fast batched forward, without
        retraining.
        """
        if len(policies) == 0:
            raise ValueError("empty policy list")
        cfg = policies[0].config
        for i, p in enumerate(policies):
            if p.config.obs_dim != cfg.obs_dim or p.config.act_dim != cfg.act_dim:
                raise ValueError(
                    f"policy {i} has mismatched (obs_dim, act_dim); "
                    f"expected ({cfg.obs_dim},{cfg.act_dim}), got "
                    f"({p.config.obs_dim},{p.config.act_dim})"
                )
        merged = cls(len(policies), cfg)
        with torch.no_grad():
            for i, p in enumerate(policies):
                merged.l1.weight[i].copy_(p.trunk[0].weight.t())
                merged.l1.bias[i].copy_(p.trunk[0].bias)
                merged.l2.weight[i].copy_(p.trunk[2].weight.t())
                merged.l2.bias[i].copy_(p.trunk[2].bias)
                merged.mean_head.weight[i].copy_(p.mean_head.weight.t())
                merged.mean_head.bias[i].copy_(p.mean_head.bias)
                merged.log_std[i].copy_(p.log_std)
        return merged

    def to_per_agent_policies(self) -> List[GaussianMLPPolicy]:
        """Split a batched module into ``n_agents`` independent
        :class:`GaussianMLPPolicy` instances.

        This lets the exp1 pipeline (``pairwise_behavioral_distance``,
        ``evaluate_policies_on_observations``) consume the overnight
        n=100 checkpoint without any new loader code.
        """
        out: List[GaussianMLPPolicy] = []
        for i in range(self.n_agents):
            p = GaussianMLPPolicy(self.config)
            with torch.no_grad():
                p.trunk[0].weight.copy_(self.l1.weight[i].t())
                p.trunk[0].bias.copy_(self.l1.bias[i])
                p.trunk[2].weight.copy_(self.l2.weight[i].t())
                p.trunk[2].bias.copy_(self.l2.bias[i])
                p.mean_head.weight.copy_(self.mean_head.weight[i].t())
                p.mean_head.bias.copy_(self.mean_head.bias[i])
                p.log_std.copy_(self.log_std[i])
            out.append(p)
        return out


class BatchedValueMLP(nn.Module):
    """Scalar state-value net, ``n_agents`` independent parameter sets."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hidden_sizes: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        h1, h2 = hidden_sizes
        self.l1 = BatchedLinear(n_agents, obs_dim, h1)
        self.l2 = BatchedLinear(n_agents, h1, h2)
        self.head = BatchedLinear(n_agents, h2, 1)
        self.l1.orthogonal_init(gain=1.0)
        self.l2.orthogonal_init(gain=1.0)
        self.head.orthogonal_init(gain=1.0)

    def forward(self, obs: Tensor) -> Tensor:
        """``obs: (n_agents, B, obs_dim)`` -> ``(n_agents, B)`` value estimates."""
        hidden = torch.tanh(self.l1(obs))
        hidden = torch.tanh(self.l2(hidden))
        return self.head(hidden).squeeze(-1)

    @classmethod
    def from_per_agent_values(
        cls, values: Sequence[ValueMLP], obs_dim: int
    ) -> "BatchedValueMLP":
        if len(values) == 0:
            raise ValueError("empty values list")
        h1 = int(values[0].net[0].out_features)
        h2 = int(values[0].net[2].out_features)
        merged = cls(len(values), obs_dim, hidden_sizes=(h1, h2))
        with torch.no_grad():
            for i, v in enumerate(values):
                merged.l1.weight[i].copy_(v.net[0].weight.t())
                merged.l1.bias[i].copy_(v.net[0].bias)
                merged.l2.weight[i].copy_(v.net[2].weight.t())
                merged.l2.bias[i].copy_(v.net[2].bias)
                merged.head.weight[i].copy_(v.net[4].weight.t())
                merged.head.bias[i].copy_(v.net[4].bias)
        return merged

    def to_per_agent_values(self) -> List[ValueMLP]:
        out: List[ValueMLP] = []
        for i in range(self.n_agents):
            v = ValueMLP(self.obs_dim, hidden_sizes=self.hidden_sizes)
            with torch.no_grad():
                v.net[0].weight.copy_(self.l1.weight[i].t())
                v.net[0].bias.copy_(self.l1.bias[i])
                v.net[2].weight.copy_(self.l2.weight[i].t())
                v.net[2].bias.copy_(self.l2.bias[i])
                v.net[4].weight.copy_(self.head.weight[i].t())
                v.net[4].bias.copy_(self.head.bias[i])
            out.append(v)
        return out


def save_batched_checkpoint(
    path: Union[str, Path],
    policy: BatchedGaussianMLPPolicy,
    value: Optional[BatchedValueMLP] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Save a batched module in a format loadable both as batched and as
    a list of per-agent policies.

    The saved file duplicates the state in two layouts:

    - ``batched_policy_state``: the direct ``state_dict`` of the
      batched module, loadable via :func:`load_batched_checkpoint`.
    - ``agents``: per-agent entries in the same format as
      ``graphsnd.policies.save_checkpoint``, so all existing code
      (Experiment 1, the metric validators) can ingest the same file
      without modification.

    The double-write is tiny (each agent MLP is ~4.5k floats); keeping
    both makes the overnight checkpoints plug-and-play with the rest of
    the repo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_agent_policies = policy.to_per_agent_policies()
    per_agent_values = value.to_per_agent_values() if value is not None else None
    agent_entries: List[Dict] = []
    for i in range(policy.n_agents):
        entry: Dict = {
            "config": policy.config.__dict__.copy(),
            "policy_state": per_agent_policies[i].state_dict(),
        }
        if per_agent_values is not None:
            entry["value_state"] = per_agent_values[i].state_dict()
        agent_entries.append(entry)
    payload = {
        "agents": agent_entries,
        "batched_policy_state": policy.state_dict(),
        "batched_value_state": value.state_dict() if value is not None else None,
        "n_agents": policy.n_agents,
        "config": policy.config.__dict__.copy(),
        "value_hidden_sizes": (
            list(value.hidden_sizes) if value is not None else None
        ),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_batched_checkpoint(
    path: Union[str, Path],
    map_location: Optional[str] = None,
) -> Tuple[BatchedGaussianMLPPolicy, Optional[BatchedValueMLP], Dict]:
    """Inverse of :func:`save_batched_checkpoint`.

    If the file was saved only as per-agent (i.e. by the non-batched
    ``save_checkpoint``), we fall back to stacking the per-agent
    policies into a new batched module -- so *any* existing Exp-1
    checkpoint can be loaded as batched on the fly.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    extra = payload.get("extra", {})
    if "batched_policy_state" in payload and payload["batched_policy_state"] is not None:
        cfg = PolicyConfig(**payload["config"])
        n = int(payload["n_agents"])
        policy = BatchedGaussianMLPPolicy(n, cfg)
        policy.load_state_dict(payload["batched_policy_state"])
        value: Optional[BatchedValueMLP] = None
        if payload.get("batched_value_state") is not None:
            v_hidden = payload.get("value_hidden_sizes") or (64, 64)
            value = BatchedValueMLP(n, cfg.obs_dim, hidden_sizes=tuple(v_hidden))
            value.load_state_dict(payload["batched_value_state"])
        return policy, value, extra
    from graphsnd.policies import load_checkpoint as _load_per_agent

    policies, values, extra2 = _load_per_agent(path, map_location=map_location)
    batched_policy = BatchedGaussianMLPPolicy.from_per_agent_policies(policies)
    batched_value: Optional[BatchedValueMLP] = None
    non_null_values = [v for v in values if v is not None]
    if len(non_null_values) == len(values) and len(values) > 0:
        batched_value = BatchedValueMLP.from_per_agent_values(
            values, policies[0].config.obs_dim
        )
    return batched_policy, batched_value, extra2
