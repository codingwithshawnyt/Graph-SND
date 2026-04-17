"""Scalable batched IPPO training on VMAS Multi-Agent Goal Navigation.

This is the trainer to use for the **Graph-SND scaling experiment**: at
team sizes ``n = 50`` or ``n = 100`` the per-agent training script
(``training/train_navigation.py``) becomes Python-loop-bound because it
instantiates ``n`` independent ``nn.Linear`` modules and invokes each of
them serially inside the rollout. This script stacks the per-agent
policies into a single :class:`BatchedGaussianMLPPolicy` so the entire
team's forward pass is one ``torch.bmm`` call per layer, and the entire
loop is GPU-resident.

Design invariants
-----------------
* **No parameter sharing.** Each agent owns a distinct slice of every
  weight tensor; gradient for agent ``i`` depends only on agent ``i``'s
  loss, matching the original SND paper's heterogeneous IPPO setup.
* **GPU-first.** Default device is ``cuda`` if available, falling back
  to CPU. The VMAS environment, all buffers, the policy, the value net,
  and the optimizer state all live on the same device; no host round
  trips during the rollout.
* **Scenario scales with n.** VMAS navigation uses rejection sampling
  for spawn positions; with the default 1x1 world and 0.1 agent radius
  the env cannot pack ``n >= 50`` agents and ``reset()`` stalls
  indefinitely. This script auto-enlarges ``world_spawning_x/y`` and
  auto-shrinks ``agent_radius`` when ``n > 16``, or lets the caller
  override via CLI.
* **Checkpoint cadence.** Checkpoints are saved at iter 0, every
  ``--ckpt-every`` iters, and at the final iter; the latest is always
  duplicated to ``checkpoints/n{n}_latest.pt`` so a scaling evaluator
  can always grab the freshest snapshot.
* **Resume-from-checkpoint.** ``--resume PATH`` restores the policy,
  value net, optimizer state, starting iter, and torch/numpy RNGs.

Operational-SND log
-------------------
At every ``--snd-every`` iters the loop measures both full SND and
Graph-SND at ``p = 0.1`` directly from the current rollout's
``(means, stds)`` tensors and prints them alongside wall-clock cost.
At ``n = 100`` this is the very thing the scaling experiment is about:
full SND is ``(100 choose 2) = 4,950`` Wasserstein evaluations per
measurement; Graph-SND at ``p = 0.1`` is roughly ``495``. The log line
shows both the metric value and the ratio of wall-clock times so the
operational speedup is visible while training -- you can tail ``-f``
the log overnight and see that the speedup survives real training
dynamics.

CLI::

    python training/train_navigation_batched.py \\
        --n-agents 100 --iters 5000 \\
        --device cuda:0 \\
        --num-envs 64 --rollout-steps 128 \\
        --ckpt-every 250 --snd-every 50

For the 2-GPU overnight plan, run two instances, one per card::

    CUDA_VISIBLE_DEVICES=0 python training/train_navigation_batched.py \\
        --n-agents 100 --iters 5000 --device cuda:0 ...
    CUDA_VISIBLE_DEVICES=1 python training/train_navigation_batched.py \\
        --n-agents 50  --iters 5000 --device cuda:0 ...

They are independent processes and independent checkpoint namespaces
(``n100_*.pt`` vs ``n50_*.pt``), so they compose trivially.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from graphsnd.batched_policies import (
    BatchedGaussianMLPPolicy,
    BatchedValueMLP,
    save_batched_checkpoint,
)
from graphsnd.graphs import bernoulli_edges, complete_edges
from graphsnd.metrics import (
    graph_snd_from_rollouts,
    pairwise_distances_on_edges,
)
from graphsnd.policies import PolicyConfig


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    num_envs: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 2048
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_steps: int = 200


@dataclass
class ScenarioKwargs:
    """Kwargs forwarded to VMAS navigation.

    The defaults below are overridden automatically when ``n_agents`` is
    large, because navigation's default 1x1 world cannot pack 50+
    agents with a 0.1 radius and rejection-sampling spawn will hang.
    """

    world_spawning_x: float = 1.0
    world_spawning_y: float = 1.0
    agent_radius: float = 0.1


def scenario_kwargs_for(n_agents: int) -> ScenarioKwargs:
    """Heuristic scenario sizing that keeps navigation packable at any n.

    Square packing density: each agent occupies a disk of radius
    ``min_dist / 2 = agent_radius + 0.025``. Ensuring the area
    ``(2 * world_spawning)^2`` fits 2 * n_agents entities (agents and
    their goals) with comfortable headroom produces the breakpoints
    below. These were confirmed to run in well under a second at
    ``n in {50, 100, 128}`` on CPU, and are quadratic in ``sqrt(n)``
    so they stay usable into the low thousands.
    """
    if n_agents <= 16:
        return ScenarioKwargs()
    if n_agents <= 32:
        return ScenarioKwargs(
            world_spawning_x=1.5, world_spawning_y=1.5, agent_radius=0.06
        )
    if n_agents <= 64:
        return ScenarioKwargs(
            world_spawning_x=2.0, world_spawning_y=2.0, agent_radius=0.05
        )
    if n_agents <= 128:
        return ScenarioKwargs(
            world_spawning_x=3.0, world_spawning_y=3.0, agent_radius=0.04
        )
    size = 0.3 * math.sqrt(n_agents)
    return ScenarioKwargs(
        world_spawning_x=size, world_spawning_y=size, agent_radius=0.035
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae_batched(
    rewards: Tensor,
    values: Tensor,
    bootstrap: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float,
) -> Tuple[Tensor, Tensor]:
    """Vectorised GAE over all agents simultaneously.

    Shapes
    ------
    rewards, values: (T, n_agents, num_envs)
    bootstrap: (n_agents, num_envs) value estimate at step T
    dones: (T, num_envs) shared across agents (VMAS emits one done per env)
    """
    T, n_agents, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(n_agents, num_envs, dtype=rewards.dtype, device=rewards.device)
    next_value = bootstrap
    for t in reversed(range(T)):
        not_done = (1.0 - dones[t]).unsqueeze(0)
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def build_env(
    scenario: str,
    num_envs: int,
    n_agents: int,
    max_steps: int,
    device: str,
    seed: int,
    scen_kwargs: ScenarioKwargs,
):
    import vmas

    env = vmas.make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        max_steps=max_steps,
        seed=seed,
        n_agents=n_agents,
        world_spawning_x=scen_kwargs.world_spawning_x,
        world_spawning_y=scen_kwargs.world_spawning_y,
        agent_radius=scen_kwargs.agent_radius,
    )
    if env.n_agents != n_agents:
        raise RuntimeError(
            f"Requested n_agents={n_agents}, scenario built n={env.n_agents}"
        )
    return env


@dataclass
class RolloutBuffers:
    obs: Tensor
    acts: Tensor
    logps: Tensor
    vals: Tensor
    rews: Tensor
    dones: Tensor


def allocate_buffers(
    T: int, n_agents: int, num_envs: int, obs_dim: int, act_dim: int, device: torch.device
) -> RolloutBuffers:
    return RolloutBuffers(
        obs=torch.zeros(T, n_agents, num_envs, obs_dim, device=device),
        acts=torch.zeros(T, n_agents, num_envs, act_dim, device=device),
        logps=torch.zeros(T, n_agents, num_envs, device=device),
        vals=torch.zeros(T, n_agents, num_envs, device=device),
        rews=torch.zeros(T, n_agents, num_envs, device=device),
        dones=torch.zeros(T, num_envs, device=device),
    )


def collect_rollout_batched(
    env,
    policy: BatchedGaussianMLPPolicy,
    value: BatchedValueMLP,
    buffers: RolloutBuffers,
    T: int,
    n_agents: int,
    device: torch.device,
) -> Tuple[Tensor, float, int]:
    """Fill rollout buffers; returns (bootstrap, mean reward/step, episodes)."""
    current_obs = torch.stack(
        [o.to(device) for o in env.reset()], dim=0
    )
    episode_reward_sum = torch.zeros(n_agents, device=device)
    n_episodes = 0
    for t in range(T):
        with torch.no_grad():
            action, log_prob, _, _ = policy.sample(current_obs)
            v = value(current_obs)
        buffers.obs[t] = current_obs
        buffers.acts[t] = action
        buffers.logps[t] = log_prob
        buffers.vals[t] = v
        actions_list = [action[i] for i in range(n_agents)]
        next_obs, rewards, dones, _ = env.step(actions_list)
        dones_t = dones.to(device).float()
        buffers.dones[t] = dones_t
        for i, r in enumerate(rewards):
            r_t = r.to(device)
            buffers.rews[t, i] = r_t
            episode_reward_sum[i] += r_t.mean()
        if dones_t.any():
            n_episodes += int(dones_t.sum().item())
            current_obs = torch.stack([o.to(device) for o in env.reset()], dim=0)
        else:
            current_obs = torch.stack([o.to(device) for o in next_obs], dim=0)
    with torch.no_grad():
        bootstrap = value(current_obs)
    mean_reward_per_step = float(episode_reward_sum.mean().item() / T)
    return bootstrap, mean_reward_per_step, n_episodes


def ppo_update_batched(
    policy: BatchedGaussianMLPPolicy,
    value: BatchedValueMLP,
    optimizer: torch.optim.Optimizer,
    buffers: RolloutBuffers,
    bootstrap: Tensor,
    cfg: PPOConfig,
) -> Dict[str, float]:
    """One PPO update with per-agent-isolated gradients via bmm."""
    adv, ret = compute_gae_batched(
        buffers.rews, buffers.vals, bootstrap, buffers.dones,
        cfg.gamma, cfg.gae_lambda,
    )
    T, n_agents, num_envs = buffers.rews.shape
    obs_dim = buffers.obs.shape[-1]
    act_dim = buffers.acts.shape[-1]
    obs_flat = buffers.obs.permute(1, 0, 2, 3).reshape(n_agents, T * num_envs, obs_dim)
    act_flat = buffers.acts.permute(1, 0, 2, 3).reshape(n_agents, T * num_envs, act_dim)
    logp_old = buffers.logps.permute(1, 0, 2).reshape(n_agents, T * num_envs)
    adv_flat = adv.permute(1, 0, 2).reshape(n_agents, T * num_envs)
    ret_flat = ret.permute(1, 0, 2).reshape(n_agents, T * num_envs)

    mean_adv = adv_flat.mean(dim=1, keepdim=True)
    std_adv = adv_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    adv_flat = (adv_flat - mean_adv) / std_adv

    batch_size = T * num_envs
    mb_size = min(cfg.minibatch_size, batch_size)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    n_updates = 0
    params = list(policy.parameters()) + list(value.parameters())
    for _ in range(cfg.epochs):
        perm = torch.randperm(batch_size, device=adv_flat.device)
        for start in range(0, batch_size, mb_size):
            idx = perm[start : start + mb_size]
            obs_mb = obs_flat[:, idx]
            act_mb = act_flat[:, idx]
            logp_old_mb = logp_old[:, idx]
            adv_mb = adv_flat[:, idx]
            ret_mb = ret_flat[:, idx]

            new_lp = policy.log_prob(obs_mb, act_mb)
            entropy = policy.entropy(obs_mb).mean()
            new_value = value(obs_mb)

            ratio = (new_lp - logp_old_mb).exp()
            surr1 = ratio * adv_mb
            surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (new_value - ret_mb).pow(2).mean()
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = (logp_old_mb - new_lp).mean().item()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["kl"] += kl
            n_updates += 1
    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats


def measure_snd_during_training(
    buffers: RolloutBuffers,
    policy: BatchedGaussianMLPPolicy,
    p_sample: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute full SND and Graph-SND (Bernoulli-p) directly from the
    current rollout. Returns values and wall-clock costs of each.
    """
    n_agents = buffers.obs.shape[1]
    flat_obs = buffers.obs.permute(1, 0, 2, 3).reshape(n_agents, -1, buffers.obs.shape[-1])
    with torch.no_grad():
        means, stds = policy(flat_obs)
    means = means.detach()
    stds = stds.detach()
    edges_full = complete_edges(n_agents).to(means.device)
    t0 = time.perf_counter()
    d_full = pairwise_distances_on_edges(means, stds, edges_full)
    full_snd = float(d_full.mean().item())
    if means.is_cuda:
        torch.cuda.synchronize(means.device)
    dt_full = time.perf_counter() - t0

    edges_sample = bernoulli_edges(n_agents, float(p_sample), rng).to(means.device)
    t0 = time.perf_counter()
    if edges_sample.numel() == 0:
        graph_snd = 0.0
    else:
        graph_snd = float(
            graph_snd_from_rollouts(means, stds, edges_sample).item()
        )
    if means.is_cuda:
        torch.cuda.synchronize(means.device)
    dt_sample = time.perf_counter() - t0
    return {
        "SND_full": full_snd,
        "GraphSND_p": graph_snd,
        "p_sample": float(p_sample),
        "n_pairs_full": int(edges_full.shape[0]),
        "n_pairs_sample": int(edges_sample.shape[0]),
        "t_full_ms": dt_full * 1e3,
        "t_sample_ms": dt_sample * 1e3,
        "speedup": (dt_full / dt_sample) if dt_sample > 0 else float("inf"),
    }


def build_checkpoint_payload(
    iter_num: int,
    policy: BatchedGaussianMLPPolicy,
    value: BatchedValueMLP,
    optimizer: torch.optim.Optimizer,
    cfg: PPOConfig,
    scen_kwargs: ScenarioKwargs,
    meta: Dict,
) -> Dict:
    return {
        "iter": iter_num,
        "policy_state": policy.state_dict(),
        "value_state": value.state_dict(),
        "optim_state": optimizer.state_dict(),
        "ppo_cfg": asdict(cfg),
        "scenario_kwargs": asdict(scen_kwargs),
        "policy_config": policy.config.__dict__.copy(),
        "value_hidden_sizes": list(value.hidden_sizes),
        "n_agents": policy.n_agents,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "meta": meta,
    }


def save_training_checkpoint(
    path: Path,
    payload: Dict,
    policy: BatchedGaussianMLPPolicy,
    value: BatchedValueMLP,
) -> None:
    """Write both the training-state checkpoint (for resume) and the
    metric-friendly batched checkpoint (loadable by Experiment 1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    metric_path = path.with_suffix(".metric.pt")
    save_batched_checkpoint(
        metric_path,
        policy,
        value,
        extra={
            "iter": payload["iter"],
            "n_agents": payload["n_agents"],
            "ppo_cfg": payload["ppo_cfg"],
            "scenario_kwargs": payload["scenario_kwargs"],
            "meta": payload["meta"],
        },
    )


def load_training_checkpoint(
    path: Path,
    map_location: str,
) -> Dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rngs(payload: Dict) -> None:
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        for i, state in enumerate(payload["cuda_rng"]):
            torch.cuda.set_rng_state(state.cpu(), device=i)
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalable batched IPPO trainer for Graph-SND.",
    )
    parser.add_argument("--n-agents", type=int, required=True,
                        help="Any positive integer; scenario sizing auto-scales for n > 16.")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scenario", type=str, default="navigation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="torch device string. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=250)
    parser.add_argument("--snd-every", type=int, default=50,
                        help="Iterations between online SND / Graph-SND measurements. 0 disables.")
    parser.add_argument("--snd-p", type=float, default=0.1,
                        help="Bernoulli edge-inclusion probability for the online Graph-SND log.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a *.pt training checkpoint produced by this script to resume from.")
    parser.add_argument("--world-x", type=float, default=None,
                        help="Override world_spawning_x. If unset, a heuristic scales with n.")
    parser.add_argument("--world-y", type=float, default=None)
    parser.add_argument("--agent-radius", type=float, default=None)
    parser.add_argument("--tag", type=str, default="overnight",
                        help="Short tag included in filenames, logs, and checkpoints.")
    args = parser.parse_args()

    device = torch.device(args.device)
    scen_kwargs = scenario_kwargs_for(args.n_agents)
    if args.world_x is not None:
        scen_kwargs.world_spawning_x = float(args.world_x)
    if args.world_y is not None:
        scen_kwargs.world_spawning_y = float(args.world_y)
    if args.agent_radius is not None:
        scen_kwargs.agent_radius = float(args.agent_radius)

    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        minibatch_size=args.minibatch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    set_seeds(args.seed)

    env = build_env(
        args.scenario, cfg.num_envs, args.n_agents, cfg.max_steps,
        args.device, args.seed, scen_kwargs,
    )
    example_obs = env.reset()
    obs_dim = int(example_obs[0].shape[-1])
    act_dim = int(env.get_agent_action_size(env.agents[0]))
    u_range = float(env.agents[0].u_range)
    policy_cfg = PolicyConfig(obs_dim=obs_dim, act_dim=act_dim, u_range=u_range)

    policy = BatchedGaussianMLPPolicy(
        args.n_agents, policy_cfg, seed_base=args.seed * 1000
    ).to(device)
    value = BatchedValueMLP(args.n_agents, obs_dim).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=cfg.lr,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base = f"n{args.n_agents}_{args.tag}"
    iter0_path = ckpt_dir / f"{base}_iter0.pt"
    latest_path = ckpt_dir / f"{base}_latest.pt"
    final_path = ckpt_dir / f"{base}_iter{args.iters}.pt"

    log_dir = Path("results/scaling")
    log_dir.mkdir(parents=True, exist_ok=True)
    snd_log_path = log_dir / f"{base}_snd_log.csv"
    train_log_path = log_dir / f"{base}_train_log.csv"

    start_iter = 1
    if args.resume is not None:
        resume_payload = load_training_checkpoint(args.resume, map_location=args.device)
        policy.load_state_dict(resume_payload["policy_state"])
        value.load_state_dict(resume_payload["value_state"])
        optimizer.load_state_dict(resume_payload["optim_state"])
        restore_rngs(resume_payload)
        start_iter = int(resume_payload["iter"]) + 1
        print(f"  resumed from {args.resume} at iter {start_iter - 1}")
    else:
        payload0 = build_checkpoint_payload(
            0, policy, value, optimizer, cfg, scen_kwargs,
            meta={"tag": args.tag, "seed": args.seed, "scenario": args.scenario},
        )
        save_training_checkpoint(iter0_path, payload0, policy, value)
        print(f"  saved random-init checkpoint -> {iter0_path}")

    scen_summary = (
        f"world={scen_kwargs.world_spawning_x}x{scen_kwargs.world_spawning_y}  "
        f"radius={scen_kwargs.agent_radius}"
    )
    print(
        f"[n={args.n_agents} tag={args.tag}] scenario={args.scenario}  "
        f"obs_dim={obs_dim} act_dim={act_dim} u_range={u_range}  "
        f"{scen_summary}  device={args.device}  num_envs={cfg.num_envs}"
    )
    if not train_log_path.exists():
        with train_log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "iter", "elapsed_s", "reward_per_step", "episodes",
                "policy_loss", "value_loss", "entropy", "kl",
            ])
    if not snd_log_path.exists():
        with snd_log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "iter", "n_agents", "p", "SND_full", "GraphSND_p",
                "t_full_ms", "t_sample_ms", "speedup",
                "n_pairs_full", "n_pairs_sample",
            ])

    buffers = allocate_buffers(
        cfg.rollout_steps, args.n_agents, cfg.num_envs, obs_dim, act_dim, device
    )
    snd_rng = np.random.default_rng(args.seed + 777)

    t_start = time.time()
    for it in range(start_iter, args.iters + 1):
        bootstrap, reward_per_step, n_eps = collect_rollout_batched(
            env, policy, value, buffers, cfg.rollout_steps, args.n_agents, device,
        )
        stats = ppo_update_batched(policy, value, optimizer, buffers, bootstrap, cfg)
        elapsed = time.time() - t_start
        with train_log_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                it, f"{elapsed:.2f}", f"{reward_per_step:.6f}", n_eps,
                f"{stats['policy_loss']:.6f}", f"{stats['value_loss']:.6f}",
                f"{stats['entropy']:.6f}", f"{stats['kl']:.6f}",
            ])
        if it == start_iter or it % args.log_every == 0 or it == args.iters:
            print(
                f"  iter {it:5d}/{args.iters}  "
                f"r/step={reward_per_step: .4f}  eps={n_eps:4d}  "
                f"pi={stats['policy_loss']: .4f}  "
                f"v={stats['value_loss']: .4f}  "
                f"H={stats['entropy']: .3f}  "
                f"kl={stats['kl']: .4f}  "
                f"({elapsed:6.1f}s)"
            )
        if args.snd_every > 0 and (it == start_iter or it % args.snd_every == 0 or it == args.iters):
            snd_info = measure_snd_during_training(buffers, policy, args.snd_p, snd_rng)
            with snd_log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    it, args.n_agents, snd_info["p_sample"],
                    f"{snd_info['SND_full']:.6f}", f"{snd_info['GraphSND_p']:.6f}",
                    f"{snd_info['t_full_ms']:.3f}", f"{snd_info['t_sample_ms']:.3f}",
                    f"{snd_info['speedup']:.2f}",
                    snd_info["n_pairs_full"], snd_info["n_pairs_sample"],
                ])
            print(
                f"    [SND] full={snd_info['SND_full']:.4f} "
                f"({snd_info['n_pairs_full']} pairs, {snd_info['t_full_ms']:.1f}ms)  "
                f"Graph-SND(p={snd_info['p_sample']:.2f})="
                f"{snd_info['GraphSND_p']:.4f} "
                f"({snd_info['n_pairs_sample']} pairs, {snd_info['t_sample_ms']:.1f}ms)  "
                f"speedup={snd_info['speedup']:.2f}x"
            )
        if it % args.ckpt_every == 0 or it == args.iters:
            payload = build_checkpoint_payload(
                it, policy, value, optimizer, cfg, scen_kwargs,
                meta={"tag": args.tag, "seed": args.seed, "scenario": args.scenario},
            )
            step_path = ckpt_dir / f"{base}_iter{it}.pt"
            save_training_checkpoint(step_path, payload, policy, value)
            save_training_checkpoint(latest_path, payload, policy, value)
            if it == args.iters:
                save_training_checkpoint(final_path, payload, policy, value)
            print(f"    checkpoint -> {step_path}  (latest -> {latest_path})")

    meta_path = ckpt_dir / f"{base}_meta.json"
    meta_path.write_text(json.dumps({
        "n_agents": args.n_agents,
        "iters": args.iters,
        "seed": args.seed,
        "tag": args.tag,
        "scenario": args.scenario,
        "scenario_kwargs": asdict(scen_kwargs),
        "ppo_cfg": asdict(cfg),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "u_range": u_range,
        "device": args.device,
        "checkpoints": {
            "iter_0": str(iter0_path),
            "latest": str(latest_path),
            "final": str(final_path),
        },
        "logs": {
            "train": str(train_log_path),
            "snd": str(snd_log_path),
        },
    }, indent=2))
    print(f"  meta -> {meta_path}")


if __name__ == "__main__":
    main()
