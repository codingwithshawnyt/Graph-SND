#!/usr/bin/env python3
"""Minimal categorical-action PPO on MPE simple-spread (no BenchMARL, no DiCo).

Trains independent categorical policies via PPO with GAE on PettingZoo's
simple_spread_v3. Logs Graph-SND (TVD) passively each iteration as a
measurement — NOT as a control signal. Saves checkpoints loadable by
``mpe_measurement_panel.py``.

This is a secondary, time-gated script. If the expander experiments
finish and time remains, run this to produce trained checkpoints for the
measurement panel. Otherwise, the panel falls back to frozen random-init
policies.

Usage:
  python experiments/mpe_ippo_training.py
  python experiments/mpe_ippo_training.py --n-agents 10 --n-iters 200 --seed 0
  python experiments/mpe_ippo_training.py --timeout-hours 24

Requirements:
  pip install pettingzoo[mpe]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphsnd.graphs import bernoulli_edges, complete_edges, random_regular_edges
from graphsnd.tvd import tvd, tvd_pairwise


# ---------------------------------------------------------------------------
# Policy and Value networks
# ---------------------------------------------------------------------------

class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def probs(self, obs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(obs), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Graph-SND measurement (passive, TVD-based)
# ---------------------------------------------------------------------------

def measure_graph_snd_tvd(probs_list: List[torch.Tensor]) -> float:
    """Full SND using TVD over all agent pairs."""
    probs = torch.stack(probs_list)  # (n_agents, n_actions)
    mat = tvd_pairwise(probs)
    n = probs.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    return float(mat[idx[0], idx[1]].mean().item())


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    policy: CategoricalPolicy,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    action_batch: torch.Tensor,
    return_batch: torch.Tensor,
    advantage_batch: torch.Tensor,
    old_logprob_batch: torch.Tensor,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    minibatch_size: int = 256,
):
    """Standard clipped PPO update."""
    n = obs_batch.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            obs = obs_batch[idx]
            act = action_batch[idx]
            ret = return_batch[idx]
            adv = advantage_batch[idx]
            old_lp = old_logprob_batch[idx]

            dist = policy(obs)
            new_lp = dist.log_prob(act)
            ratio = (new_lp - old_lp).exp()

            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = value_net(obs)
            value_loss = F.mse_loss(value_pred, ret)

            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()), 0.5
            )
            optimizer.step()


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout collection using PettingZoo
# ---------------------------------------------------------------------------

def collect_rollout(env, policies, value_nets, n_steps: int = 100):
    """Collect one rollout from the PettingZoo AEC environment."""
    agent_data = {agent: {"obs": [], "actions": [], "rewards": [], "logprobs": [], "values": [], "dones": []}
                  for agent in env.possible_agents}

    env.reset()
    step = 0
    while step < n_steps:
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if agent in agent_data:
                if len(agent_data[agent]["obs"]) > 0:
                    agent_data[agent]["rewards"].append(reward)
                    agent_data[agent]["dones"].append(done)

            if done:
                env.step(None)
                continue

            obs_t = torch.tensor(obs, dtype=torch.float32)
            idx = env.possible_agents.index(agent)
            with torch.no_grad():
                dist = policies[idx](obs_t)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = value_nets[idx](obs_t)

            agent_data[agent]["obs"].append(obs_t)
            agent_data[agent]["actions"].append(action)
            agent_data[agent]["logprobs"].append(logprob)
            agent_data[agent]["values"].append(value.item())

            env.step(action.item())
            step += 1
            if step >= n_steps:
                break

    # Pad final rewards/dones
    for agent in env.possible_agents:
        d = agent_data[agent]
        while len(d["rewards"]) < len(d["obs"]):
            d["rewards"].append(0.0)
            d["dones"].append(True)

    return agent_data


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    n_agents: int = 3,
    n_iters: int = 200,
    n_rollouts_per_iter: int = 10,
    n_steps_per_rollout: int = 25,
    seed: int = 0,
    lr: float = 3e-4,
    output_dir: str = "results/mpe_ippo",
    timeout_hours: float = 24.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Import PettingZoo MPE
    try:
        from pettingzoo.mpe import simple_spread_v3
    except ImportError:
        print("ERROR: pettingzoo[mpe] not installed. Run: pip install pettingzoo[mpe]")
        sys.exit(1)

    env = simple_spread_v3.env(N=n_agents, max_cycles=n_steps_per_rollout)
    env.reset(seed=seed)

    # Infer dimensions from first agent
    sample_obs = env.observe(env.possible_agents[0])
    obs_dim = sample_obs.shape[0]
    n_actions = env.action_space(env.possible_agents[0]).n

    print(f"MPE simple_spread: n_agents={n_agents}, obs_dim={obs_dim}, n_actions={n_actions}")

    # Create per-agent policies and value nets
    policies = [CategoricalPolicy(obs_dim, n_actions) for _ in range(n_agents)]
    value_nets = [ValueNet(obs_dim) for _ in range(n_agents)]

    all_params = []
    for p, v in zip(policies, value_nets):
        all_params.extend(p.parameters())
        all_params.extend(v.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # CSV logger
    csv_path = out_path / "mpe_training_log.csv"
    csv_fields = ["iter", "seed", "n_agents", "reward_mean", "graph_snd_tvd", "wallclock_s"]
    csv_fh = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    csv_writer.writeheader()

    start_time = time.time()
    timeout_sec = timeout_hours * 3600

    for it in range(n_iters):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print(f"[iter {it}] TIMEOUT after {elapsed/3600:.1f}h. Saving checkpoint and exiting.")
            break

        iter_rewards = []

        for agent_idx in range(n_agents):
            all_obs, all_actions, all_returns, all_advantages, all_logprobs = [], [], [], [], []

            for _ in range(n_rollouts_per_iter):
                env.reset(seed=seed * 10000 + it * 100 + agent_idx)
                data = collect_rollout(env, policies, value_nets, n_steps_per_rollout * n_agents)

                agent_name = env.possible_agents[agent_idx]
                d = data[agent_name]
                if len(d["obs"]) == 0:
                    continue

                advantages, returns = compute_gae(d["rewards"], d["values"], d["dones"])

                all_obs.extend(d["obs"])
                all_actions.extend(d["actions"])
                all_returns.extend(returns)
                all_advantages.extend(advantages)
                all_logprobs.extend(d["logprobs"])
                iter_rewards.extend(d["rewards"])

            if len(all_obs) == 0:
                continue

            obs_batch = torch.stack(all_obs)
            action_batch = torch.stack(all_actions)
            return_batch = torch.tensor(all_returns, dtype=torch.float32)
            advantage_batch = torch.tensor(all_advantages, dtype=torch.float32)
            if advantage_batch.std() > 1e-8:
                advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
            logprob_batch = torch.stack(all_logprobs)

            ppo_update(policies[agent_idx], value_nets[agent_idx], optimizer,
                       obs_batch, action_batch, return_batch, advantage_batch, logprob_batch)

        # Passive Graph-SND measurement
        env.reset(seed=seed)
        sample_obs_t = torch.tensor(env.observe(env.possible_agents[0]), dtype=torch.float32)
        with torch.no_grad():
            probs_list = [p.probs(sample_obs_t) for p in policies]
        snd_tvd = measure_graph_snd_tvd(probs_list)

        reward_mean = float(np.mean(iter_rewards)) if iter_rewards else float("nan")
        elapsed_s = time.time() - start_time

        csv_writer.writerow({
            "iter": it,
            "seed": seed,
            "n_agents": n_agents,
            "reward_mean": f"{reward_mean:.6f}",
            "graph_snd_tvd": f"{snd_tvd:.6f}",
            "wallclock_s": f"{elapsed_s:.1f}",
        })
        csv_fh.flush()

        if it % 10 == 0:
            print(f"[iter {it:4d}] reward={reward_mean:.4f}  snd_tvd={snd_tvd:.4f}  elapsed={elapsed_s:.0f}s")

    csv_fh.close()

    # Save checkpoint
    ckpt = {}
    for i, p in enumerate(policies):
        ckpt[f"agent_{i}"] = p.state_dict()
    ckpt_path = out_path / f"checkpoint_seed{seed}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    print(f"Training log: {csv_path}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="MPE IPPO Training (no DiCo)")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="results/mpe_ippo")
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()

    train(
        n_agents=args.n_agents,
        n_iters=args.n_iters,
        n_rollouts_per_iter=args.n_rollouts,
        seed=args.seed,
        lr=args.lr,
        output_dir=args.output_dir,
        timeout_hours=args.timeout_hours,
    )


if __name__ == "__main__":
    main()
