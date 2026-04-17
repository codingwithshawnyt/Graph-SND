"""Per-agent Gaussian MLP policies and matching value network.

No parameter sharing across agents: each ``GaussianMLPPolicy`` is
instantiated separately and trained independently, which is the
heterogeneous-policy setup the paper's metrics are designed for.
A policy outputs:

- ``mean(o)``: state-dependent via a two-layer MLP with ``tanh`` hidden
  activations, followed by ``tanh`` on the output scaled by ``u_range``
  so actions live in ``[-u_range, u_range]^{d_act}``.
- ``log_std``: a single per-dimension state-independent learnable
  parameter vector, matching the original SND paper's parameterisation.

The value network has the same hidden architecture but is scalar-valued
and unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PolicyConfig:
    obs_dim: int
    act_dim: int
    hidden_sizes: Tuple[int, int] = (64, 64)
    u_range: float = 1.0
    init_log_std: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class GaussianMLPPolicy(nn.Module):
    """Stochastic diagonal-Gaussian policy with per-agent parameters."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        h1, h2 = config.hidden_sizes
        self.trunk = nn.Sequential(
            nn.Linear(config.obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(h2, config.act_dim)
        self.log_std = nn.Parameter(
            torch.full((config.act_dim,), float(config.init_log_std))
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(mean, std)``. ``std`` is a per-dim tensor of std devs.

        Shapes
        ------
        - obs: ``(..., obs_dim)``
        - mean: ``(..., act_dim)`` bounded in ``[-u_range, u_range]``
        - std:  broadcast of ``(act_dim,)`` to match ``mean``
        """
        hidden = self.trunk(obs)
        raw_mean = self.mean_head(hidden)
        mean = torch.tanh(raw_mean) * self.config.u_range
        log_std = self.log_std.clamp(
            self.config.log_std_min, self.config.log_std_max
        )
        std = log_std.exp().expand_as(mean)
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
        """Sample an action. Returns ``(action, log_prob, mean, std)``.

        ``action`` is clipped to ``[-u_range, u_range]`` so VMAS does not
        reject it; ``log_prob`` is computed against the unclipped sample
        which is standard practice for PPO.
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


class ValueMLP(nn.Module):
    """Scalar state-value network used for independent PPO baselines."""

    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, int] = (64, 64)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class AgentCheckpoint:
    """Serializable snapshot of one agent's policy + (optional) value net."""

    config: PolicyConfig
    policy_state: Dict[str, Tensor] = field(default_factory=dict)
    value_state: Optional[Dict[str, Tensor]] = None


def save_checkpoint(
    path: Path,
    policies: List[GaussianMLPPolicy],
    values: Optional[List[ValueMLP]] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Save a heterogeneous set of per-agent policies to one ``.pt`` file.

    We store config alongside each state dict so loaders can instantiate
    the matching modules without needing to rediscover shapes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints: List[Dict] = []
    for i, policy in enumerate(policies):
        entry = {
            "config": policy.config.__dict__.copy(),
            "policy_state": policy.state_dict(),
        }
        if values is not None:
            entry["value_state"] = values[i].state_dict()
        checkpoints.append(entry)
    payload = {"agents": checkpoints, "extra": extra or {}}
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> Tuple[List[GaussianMLPPolicy], List[Optional[ValueMLP]], Dict]:
    """Inverse of :func:`save_checkpoint`.

    Returns ``(policies, values, extra)`` where ``values[i]`` may be
    ``None`` for agents stored without a baseline network.

    Value-network hidden sizes are inferred from the saved state-dict
    shapes, so this loader works for any ``ValueMLP`` configuration
    including the non-default sizes used by the scaling experiment.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    policies: List[GaussianMLPPolicy] = []
    values: List[Optional[ValueMLP]] = []
    for entry in payload["agents"]:
        cfg = PolicyConfig(**entry["config"])
        policy = GaussianMLPPolicy(cfg)
        policy.load_state_dict(entry["policy_state"])
        policies.append(policy)
        value_state = entry.get("value_state")
        if value_state is not None:
            h1 = int(value_state["net.0.weight"].shape[0])
            h2 = int(value_state["net.2.weight"].shape[0])
            vnet = ValueMLP(cfg.obs_dim, hidden_sizes=(h1, h2))
            vnet.load_state_dict(value_state)
            values.append(vnet)
        else:
            values.append(None)
    return policies, values, payload.get("extra", {})
