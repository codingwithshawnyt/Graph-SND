"""Minimal Independent PPO training on VMAS Multi-Agent Goal Navigation.

Each agent has its own ``GaussianMLPPolicy`` and ``ValueMLP`` (no
parameter sharing) and is updated with a standard clipped-PPO
objective against its own rewards. This is the same "heterogeneous,
independently parameterised" setup the SND paper measures behavioural
diversity over.

The goal of this script is not to produce great policies; it is to
produce **frozen heterogeneous checkpoints** that the Experiment 1
scripts can load to build a non-trivial behavioural-distance matrix.
We therefore keep the training loop intentionally small:

- one rollout of ``rollout_steps`` environment steps per iteration,
- PPO mini-batch updates for a fixed number of epochs,
- checkpoint at iter 0 (random init) and iter ``iters`` (partially
  trained), both of which are saved with every agent's policy/value
  state and enough config to reload.

CLI::

    python training/train_navigation.py --n-agents 4  --iters 100
    python training/train_navigation.py --n-agents 8  --iters 100
    python training/train_navigation.py --n-agents 16 --iters 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from graphsnd.policies import (
    GaussianMLPPolicy,
    PolicyConfig,
    ValueMLP,
    save_checkpoint,
)


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    num_envs: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 512
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_steps: int = 200


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float,
) -> Tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation for a single agent.

    Parameters
    ----------
    rewards: Tensor[T, num_envs]
    values: Tensor[T, num_envs] (value estimate at step t)
    bootstrap: Tensor[num_envs] (value estimate at step T)
    dones: Tensor[T, num_envs] (1 if env resets after step t)
    """
    T, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)
    next_value = bootstrap
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: GaussianMLPPolicy,
    value_net: ValueMLP,
    optimizer: torch.optim.Optimizer,
    obs: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    cfg: PPOConfig,
) -> dict:
    """Run one PPO update (several epochs of minibatch SGD) for one agent."""
    advantages = (advantages - advantages.mean()) / (advantages.std().clamp(min=1e-8))

    batch_size = obs.shape[0]
    mb_size = min(cfg.minibatch_size, batch_size)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    n_updates = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, mb_size):
            idx = perm[start : start + mb_size]
            obs_mb = obs[idx]
            act_mb = actions[idx]
            old_lp_mb = old_log_probs[idx]
            adv_mb = advantages[idx]
            ret_mb = returns[idx]

            new_lp = policy.log_prob(obs_mb, act_mb)
            entropy = policy.entropy(obs_mb).mean()
            new_value = value_net(obs_mb)

            ratio = (new_lp - old_lp_mb).exp()
            surr1 = ratio * adv_mb
            surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (new_value - ret_mb).pow(2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()),
                cfg.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                kl = (old_lp_mb - new_lp).mean().item()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["kl"] += kl
            n_updates += 1
    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats


def build_env(scenario: str, num_envs: int, n_agents: int, max_steps: int, device: str, seed: int):
    import vmas

    env = vmas.make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        max_steps=max_steps,
        seed=seed,
        n_agents=n_agents,
    )
    if env.n_agents != n_agents:
        raise RuntimeError(
            f"Requested n_agents={n_agents}, but the scenario built n={env.n_agents}."
        )
    return env


def collect_and_update(
    env,
    policies: List[GaussianMLPPolicy],
    values: List[ValueMLP],
    optimizers: List[torch.optim.Optimizer],
    cfg: PPOConfig,
    device: torch.device,
) -> dict:
    """Run one full PPO iteration: rollout + per-agent update.

    Returns a dict of diagnostic stats, averaged over agents.
    """
    n_agents = len(policies)
    num_envs = int(env.batch_dim)
    T = cfg.rollout_steps

    obs_buf = [torch.zeros(T, num_envs, policies[i].config.obs_dim, device=device) for i in range(n_agents)]
    act_buf = [torch.zeros(T, num_envs, policies[i].config.act_dim, device=device) for i in range(n_agents)]
    logp_buf = [torch.zeros(T, num_envs, device=device) for _ in range(n_agents)]
    val_buf = [torch.zeros(T, num_envs, device=device) for _ in range(n_agents)]
    rew_buf = [torch.zeros(T, num_envs, device=device) for _ in range(n_agents)]
    done_buf = torch.zeros(T, num_envs, device=device)

    current_obs = [o.to(device) for o in env.reset()]
    episode_reward = torch.zeros(n_agents, device=device)
    n_episodes = 0

    for t in range(T):
        step_actions: List[Tensor] = []
        for i in range(n_agents):
            obs_i = current_obs[i]
            with torch.no_grad():
                action, log_prob, _, _ = policies[i].sample(obs_i)
                value = values[i](obs_i)
            obs_buf[i][t] = obs_i
            act_buf[i][t] = action
            logp_buf[i][t] = log_prob
            val_buf[i][t] = value
            step_actions.append(action)

        next_obs, rewards, dones, _ = env.step(step_actions)
        dones_t = dones.to(device).float()
        done_buf[t] = dones_t
        for i, r in enumerate(rewards):
            r_t = r.to(device)
            rew_buf[i][t] = r_t
            episode_reward[i] += r_t.mean()

        if dones_t.any():
            n_episodes += int(dones_t.sum().item())
            current_obs = [o.to(device) for o in env.reset()]
        else:
            current_obs = [o.to(device) for o in next_obs]

    bootstraps = []
    for i in range(n_agents):
        with torch.no_grad():
            bootstraps.append(values[i](current_obs[i]))

    agent_stats = []
    for i in range(n_agents):
        adv, ret = compute_gae(
            rew_buf[i], val_buf[i], bootstraps[i], done_buf,
            cfg.gamma, cfg.gae_lambda,
        )
        obs_flat = obs_buf[i].reshape(-1, policies[i].config.obs_dim)
        act_flat = act_buf[i].reshape(-1, policies[i].config.act_dim)
        lp_flat = logp_buf[i].reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        stats = ppo_update(
            policies[i], values[i], optimizers[i],
            obs_flat, act_flat, lp_flat, adv_flat, ret_flat, cfg,
        )
        agent_stats.append(stats)

    avg = {k: float(np.mean([s[k] for s in agent_stats])) for k in agent_stats[0]}
    avg["mean_episode_reward_per_step"] = float(episode_reward.mean().item() / T)
    avg["episodes_finished"] = n_episodes
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal IPPO training on VMAS navigation for Graph-SND."
    )
    parser.add_argument("--n-agents", type=int, required=True, choices=[2, 4, 6, 8, 12, 16])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--scenario", type=str, default="navigation")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory in which to save iter_0 and iter_{iters} checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device to train on. CPU is fast enough for these sizes.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
    )
    set_seeds(args.seed)
    device = torch.device(args.device)

    env = build_env(
        args.scenario, cfg.num_envs, args.n_agents, cfg.max_steps, args.device, args.seed,
    )
    example_obs = env.reset()
    obs_dim = int(example_obs[0].shape[-1])
    act_dim = int(env.get_agent_action_size(env.agents[0]))
    u_range = float(env.agents[0].u_range)
    print(
        f"[n={args.n_agents}] scenario={args.scenario} obs_dim={obs_dim} "
        f"act_dim={act_dim} u_range={u_range} num_envs={cfg.num_envs}"
    )

    policies: List[GaussianMLPPolicy] = []
    values: List[ValueMLP] = []
    optimizers: List[torch.optim.Optimizer] = []
    for i in range(args.n_agents):
        torch.manual_seed(args.seed * 1000 + i)
        pc = PolicyConfig(obs_dim=obs_dim, act_dim=act_dim, u_range=u_range)
        policy = GaussianMLPPolicy(pc).to(device)
        value_net = ValueMLP(obs_dim).to(device)
        optim = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=cfg.lr,
        )
        policies.append(policy)
        values.append(value_net)
        optimizers.append(optim)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    iter0_path = ckpt_dir / f"n{args.n_agents}_iter0.pt"
    iterN_path = ckpt_dir / f"n{args.n_agents}_iter{args.iters}.pt"

    save_checkpoint(
        iter0_path, policies, values,
        extra={
            "n_agents": args.n_agents,
            "iter": 0,
            "seed": args.seed,
            "scenario": args.scenario,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "u_range": u_range,
        },
    )
    print(f"  saved random-init checkpoint -> {iter0_path}")

    t_start = time.time()
    for it in range(1, args.iters + 1):
        stats = collect_and_update(env, policies, values, optimizers, cfg, device)
        if it == 1 or it % args.log_every == 0 or it == args.iters:
            elapsed = time.time() - t_start
            print(
                f"  iter {it:4d}/{args.iters}  "
                f"reward/step={stats['mean_episode_reward_per_step']: .4f}  "
                f"eps_done={stats['episodes_finished']:4d}  "
                f"pi_loss={stats['policy_loss']: .4f}  "
                f"v_loss={stats['value_loss']: .4f}  "
                f"ent={stats['entropy']: .3f}  "
                f"kl={stats['kl']: .4f}  "
                f"({elapsed:5.1f}s)"
            )

    save_checkpoint(
        iterN_path, policies, values,
        extra={
            "n_agents": args.n_agents,
            "iter": args.iters,
            "seed": args.seed,
            "scenario": args.scenario,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "u_range": u_range,
        },
    )
    print(f"  saved iter-{args.iters} checkpoint -> {iterN_path}")

    meta_path = ckpt_dir / f"n{args.n_agents}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "n_agents": args.n_agents,
                "iters": args.iters,
                "seed": args.seed,
                "scenario": args.scenario,
                "rollout_steps": cfg.rollout_steps,
                "num_envs": cfg.num_envs,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "u_range": u_range,
                "checkpoints": {
                    "iter_0": str(iter0_path),
                    f"iter_{args.iters}": str(iterN_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
