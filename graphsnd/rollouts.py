"""Rollout collection for evaluating the behavioral-distance matrix.

The paper's eq. (1) averages ``W_2`` between two policies' action
distributions over a set ``B`` of observations drawn from the rollouts
of the multi-agent system. In practice ``B`` is the union over all
rollout observations and all agent positions. Concretely, we

1. roll the policies forward in a vectorised VMAS environment for
   ``n_steps`` steps,
2. concatenate each agent's observations across steps (and across the
   vectorised batch),
3. stack them across agents to build one ``(n, T_total, obs_dim)``
   tensor, and then
4. re-evaluate every policy on every agent's observations to get the
   action-distribution parameters used by
   :func:`graphsnd.metrics.pairwise_behavioral_distance`.

Step 4 is what makes the matrix meaningful: each policy is evaluated on
the same observation set, so any difference in the resulting action
distributions is attributable to policy heterogeneity rather than to
input drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from graphsnd.policies import GaussianMLPPolicy


@dataclass
class RolloutBatch:
    """Container for a rollout over a vectorised env.

    Attributes
    ----------
    observations: Tensor of shape ``(n_agents, T_total, obs_dim)``. Here
        ``T_total = n_steps * num_envs``, i.e. the vectorised-env batch
        has already been flattened into the time axis.
    actions: Tensor of shape ``(n_agents, T_total, act_dim)`` of the
        actions that were sampled during the rollout.
    log_probs: same shape minus the action dim. Sampled-policy
        log-probabilities; populated so the IPPO training loop can reuse
        this structure without a second forward pass.
    rewards: Tensor ``(n_agents, T_total)``.
    dones: Bool/float Tensor ``(T_total,)``; shared across agents
        because VMAS emits a single per-env done signal.
    values: optional Tensor ``(n_agents, T_total)``. Populated by the
        training loop when a value network is available.
    """

    observations: Tensor
    actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    dones: Tensor
    values: Optional[Tensor] = None


@torch.no_grad()
def collect_rollouts(
    env,
    policies: Sequence[GaussianMLPPolicy],
    n_steps: int,
    deterministic: bool = False,
    device: Optional[torch.device] = None,
) -> RolloutBatch:
    """Run ``n_steps`` environment steps with per-agent policies.

    Parameters
    ----------
    env: a VMAS environment. ``env.n_agents`` must match ``len(policies)``,
        and ``env.batch_dim`` controls the vectorisation.
    policies: one policy per agent; the ``i``-th policy acts for agent
        ``i``.
    n_steps: how many steps of the environment to roll out. The env is
        NOT reset between rollouts; callers manage that.
    deterministic: if ``True``, take the mean action each step. Useful
        when you want a noise-free observation set.
    device: where to place the returned tensors. Defaults to
        ``policies[0]``'s device.

    Returns
    -------
    RolloutBatch with the shapes described on the dataclass.
    """
    n_agents = len(policies)
    if env.n_agents != n_agents:
        raise ValueError(
            f"env has {env.n_agents} agents but {n_agents} policies were provided"
        )

    tgt_device = device if device is not None else next(policies[0].parameters()).device
    batch = int(env.batch_dim)

    obs_list = env.reset()
    current_obs = [o.to(tgt_device) for o in obs_list]

    obs_buf: List[List[Tensor]] = [[] for _ in range(n_agents)]
    act_buf: List[List[Tensor]] = [[] for _ in range(n_agents)]
    logp_buf: List[List[Tensor]] = [[] for _ in range(n_agents)]
    rew_buf: List[List[Tensor]] = [[] for _ in range(n_agents)]
    done_buf: List[Tensor] = []

    for _ in range(n_steps):
        step_actions: List[Tensor] = []
        step_logps: List[Tensor] = []
        for i, policy in enumerate(policies):
            obs_i = current_obs[i]
            action, log_prob, _, _ = policy.sample(obs_i, deterministic=deterministic)
            step_actions.append(action)
            step_logps.append(log_prob)
            obs_buf[i].append(obs_i.detach().cpu())
            act_buf[i].append(action.detach().cpu())
            logp_buf[i].append(log_prob.detach().cpu())

        next_obs, rewards, dones, _ = env.step(step_actions)
        for i, r in enumerate(rewards):
            rew_buf[i].append(r.detach().cpu())
        done_buf.append(
            (dones.detach().cpu() if isinstance(dones, Tensor) else torch.as_tensor(dones))
        )

        current_obs = [o.to(tgt_device) for o in next_obs]

    observations = torch.stack(
        [torch.cat(agent_obs, dim=0) for agent_obs in obs_buf], dim=0
    )
    actions = torch.stack(
        [torch.cat(agent_acts, dim=0) for agent_acts in act_buf], dim=0
    )
    log_probs = torch.stack(
        [torch.cat(agent_logps, dim=0) for agent_logps in logp_buf], dim=0
    )
    rewards = torch.stack(
        [torch.cat(agent_rews, dim=0) for agent_rews in rew_buf], dim=0
    )
    dones = torch.cat(done_buf, dim=0).float()

    return RolloutBatch(
        observations=observations.to(tgt_device),
        actions=actions.to(tgt_device),
        log_probs=log_probs.to(tgt_device),
        rewards=rewards.to(tgt_device),
        dones=dones.to(tgt_device),
        values=None,
    )


@torch.no_grad()
def evaluate_policies_on_observations(
    policies: Sequence[GaussianMLPPolicy],
    observations: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Evaluate every policy on the concatenated per-agent observations.

    Input ``observations`` has shape ``(n_agents, T_total, obs_dim)``.
    Every policy is evaluated on the flattened ``(n_agents * T_total,
    obs_dim)`` tensor so the matrix position ``D[i, j]`` is computed on
    exactly the same observation set for every pair.

    Returns
    -------
    means, stds: Tensors of shape ``(n_policies, n_agents * T_total,
    act_dim)`` suitable to pass to
    :func:`graphsnd.metrics.pairwise_behavioral_distance`.
    """
    if observations.ndim != 3:
        raise ValueError(
            f"observations must have shape (n_agents, T_total, obs_dim); "
            f"got {tuple(observations.shape)}"
        )
    n_agents, t_total, _ = observations.shape
    flat = observations.reshape(n_agents * t_total, -1)

    means_list: List[Tensor] = []
    stds_list: List[Tensor] = []
    for policy in policies:
        policy_device = next(policy.parameters()).device
        mean, std = policy(flat.to(policy_device))
        means_list.append(mean.detach().cpu())
        stds_list.append(std.detach().cpu())
    means = torch.stack(means_list, dim=0)
    stds = torch.stack(stds_list, dim=0)
    return means, stds
