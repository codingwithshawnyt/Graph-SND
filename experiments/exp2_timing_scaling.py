"""Experiment 2: frozen-init wall-clock scaling at large n.

Isolates the O(n^2) -> O(|E|) prediction at team sizes
where the n=100 scaling run (:mod:`training.train_navigation_batched`)
and the Experiment 1 CPU timing sweep (:mod:`experiments.exp1_metric_comparison`)
start to be dominated by memory and Python-loop overhead.

The experiment is deliberately minimal: we instantiate ``n_agents``
independently-initialised Gaussian MLP policies (identical architecture
to the n=100 training run), evaluate them on a synthetic
rollout batch, and time both full-SND and Graph-SND on the resulting
``(means, stds)`` tensors. No environment is stepped and no checkpoints
are required, so the experiment can be launched for any ``n`` up to
the point where the ``(n, T_total, d_act)`` tensor exceeds GPU memory.

All timings call ``torch.cuda.synchronize`` on both sides of each
``time.perf_counter`` measurement, matching the stricter
timing regime.

CLI::

    python experiments/exp2_timing_scaling.py \
        --n-agents 500 --device cuda:0 \
        --out results/exp2/timing_n500.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphsnd.batched_policies import BatchedGaussianMLPPolicy
from graphsnd.graphs import bernoulli_edges, complete_edges
from graphsnd.metrics import (
    graph_snd_from_rollouts,
    pairwise_distances_on_edges,
)
from graphsnd.policies import PolicyConfig


@dataclass
class TimingConfig:
    n_agents_list: Tuple[int, ...] = (500,)
    num_envs: int = 32
    rollout_steps: int = 128
    obs_dim: int = 18
    act_dim: int = 2
    u_range: float = 1.0
    hidden_sizes: Tuple[int, int] = (64, 64)
    p_values: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    timing_trials: int = 20
    warmup_trials: int = 3
    seed: int = 42
    device: str = "cuda:0"
    dtype: str = "float32"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def build_frozen_rollouts(
    n_agents: int, cfg: TimingConfig, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce ``(means, stds)`` of shape ``(n_agents, T_total, act_dim)``.

    Uses ``BatchedGaussianMLPPolicy`` so all ``n_agents`` policies share
    one forward kernel. Observations are uniform on ``[-1, 1]`` per the
    convention used elsewhere in this repo for synthetic smoke tests; the
    policies are initialised with orthogonal weights and
    ``log_std = -0.5``.
    """
    T_total = cfg.num_envs * cfg.rollout_steps
    policy_cfg = PolicyConfig(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        hidden_sizes=cfg.hidden_sizes,
        u_range=cfg.u_range,
    )
    policy = BatchedGaussianMLPPolicy(
        n_agents=n_agents, config=policy_cfg, seed_base=cfg.seed
    ).to(device=device, dtype=dtype)
    policy.eval()

    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    obs = (
        torch.rand(
            (n_agents, T_total, cfg.obs_dim),
            generator=gen, device=device, dtype=dtype,
        )
        * 2.0
        - 1.0
    )

    with torch.no_grad():
        means, stds = policy(obs)
    means = means.detach()
    stds = stds.detach()
    return means, stds


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_full_snd(
    means: torch.Tensor,
    stds: torch.Tensor,
    edges_full: torch.Tensor,
    trials: int,
    device: torch.device,
) -> Tuple[List[float], float]:
    """Time the full-SND aggregation ``trials`` times; return times and value."""
    times = []
    last_value = float("nan")
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        d_full = pairwise_distances_on_edges(means, stds, edges_full)
        value = d_full.mean()
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_value = float(value.item())
    return times, last_value


def time_graph_snd(
    means: torch.Tensor,
    stds: torch.Tensor,
    n_agents: int,
    p: float,
    trials: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[List[float], List[int]]:
    """Time Graph-SND with a fresh Bernoulli-``p`` graph each trial.

    The per-trial cost includes both graph construction (CPU-side
    sampling + host->device copy) and the sampled-edge aggregation, which
    matches ``measure_snd_during_training`` and so produces
    directly comparable absolute times.
    """
    times = []
    sizes = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        edges = bernoulli_edges(n_agents, p, rng).to(device)
        if edges.numel() == 0:
            _sync(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            sizes.append(0)
            continue
        _ = graph_snd_from_rollouts(means, stds, edges).item()
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        sizes.append(int(edges.shape[0]))
    return times, sizes


def run_for_n(
    n_agents: int,
    cfg: TimingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> List[dict]:
    print(f"[n={n_agents}] building (means, stds) on {device} ({dtype})...")
    means, stds = build_frozen_rollouts(n_agents, cfg, device, dtype)
    R = int(means.shape[0] * means.shape[1])
    n_pairs = n_agents * (n_agents - 1) // 2
    edges_full = complete_edges(n_agents).to(device)
    print(
        f"[n={n_agents}] means.shape={tuple(means.shape)} n_pairs={n_pairs} "
        f"R={R} warmup={cfg.warmup_trials} trials={cfg.timing_trials}"
    )

    if cfg.warmup_trials > 0:
        _ = time_full_snd(means, stds, edges_full, cfg.warmup_trials, device)
        _ = time_graph_snd(
            means, stds, n_agents, 0.1, cfg.warmup_trials,
            np.random.default_rng(cfg.seed + 1000), device,
        )

    full_times, full_value = time_full_snd(
        means, stds, edges_full, cfg.timing_trials, device
    )
    full_mean = float(np.mean(full_times))
    full_std = float(np.std(full_times, ddof=1)) if len(full_times) > 1 else 0.0
    full_med = float(np.median(full_times))
    print(
        f"[n={n_agents}] full SND: mean={full_mean*1e3:.1f} ms  "
        f"median={full_med*1e3:.1f} ms  SND={full_value:.6f}"
    )

    rows: List[dict] = []
    for p in cfg.p_values:
        rng = np.random.default_rng(cfg.seed + int(1000 * p) + n_agents)
        times, sizes = time_graph_snd(
            means, stds, n_agents, float(p), cfg.timing_trials, rng, device
        )
        sm = float(np.mean(times))
        ss = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        smed = float(np.median(times))
        speedup = full_mean / sm if sm > 0 else float("inf")
        speedup_med = full_med / smed if smed > 0 else float("inf")
        print(
            f"[n={n_agents}] p={p:.2f}: graph mean={sm*1e3:.1f} ms  "
            f"med={smed*1e3:.1f} ms  speedup(mean)={speedup:.2f}x  "
            f"speedup(med)={speedup_med:.2f}x  |E|={np.mean(sizes):.0f}"
        )
        rows.append(
            {
                "n": n_agents,
                "R": R,
                "n_pairs": n_pairs,
                "p": float(p),
                "mean_edges": float(np.mean(sizes)),
                "full_time_mean": full_mean,
                "full_time_std": full_std,
                "full_time_median": full_med,
                "sample_time_mean": sm,
                "sample_time_std": ss,
                "sample_time_median": smed,
                "speedup": speedup,
                "speedup_median": speedup_med,
                "SND_full": full_value,
                "device": str(device),
                "dtype": cfg.dtype,
                "timing_trials": cfg.timing_trials,
                "num_envs": cfg.num_envs,
                "rollout_steps": cfg.rollout_steps,
                "obs_dim": cfg.obs_dim,
                "act_dim": cfg.act_dim,
            }
        )

    del means, stds, edges_full
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _parse_n_list(s: str) -> Tuple[int, ...]:
    parts = [p for p in s.replace(",", " ").split() if p]
    return tuple(int(x) for x in parts)


def _parse_float_list(s: str) -> Tuple[float, ...]:
    parts = [p for p in s.replace(",", " ").split() if p]
    return tuple(float(x) for x in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-init wall-clock scaling sweep for full SND vs "
            "Graph-SND. Designed for n up to 500+ on a single GPU."
        )
    )
    parser.add_argument(
        "--n-agents", type=str, default="500",
        help="Comma/space-separated list of team sizes to sweep."
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--obs-dim", type=int, default=18)
    parser.add_argument("--act-dim", type=int, default=2)
    parser.add_argument(
        "--p-values", type=str, default="0.1,0.25,0.5,0.75,1.0",
    )
    parser.add_argument("--timing-trials", type=int, default=20)
    parser.add_argument("--warmup-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--out", type=str, default="results/exp2/timing_scaling.csv"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device requested cuda but torch.cuda.is_available() is False"
        )
    dtype = _resolve_dtype(args.dtype)

    cfg = TimingConfig(
        n_agents_list=_parse_n_list(args.n_agents),
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        p_values=_parse_float_list(args.p_values),
        timing_trials=args.timing_trials,
        warmup_trials=args.warmup_trials,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    set_seeds(cfg.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    all_rows: List[dict] = []
    for n in cfg.n_agents_list:
        try:
            rows = run_for_n(n, cfg, device, dtype)
            all_rows.extend(rows)
        except torch.cuda.OutOfMemoryError as exc:
            print(
                f"[n={n}] OOM: {exc}. Recording OOM row and continuing."
            )
            all_rows.append(
                {
                    "n": n,
                    "R": 0,
                    "n_pairs": n * (n - 1) // 2,
                    "p": float("nan"),
                    "mean_edges": float("nan"),
                    "full_time_mean": float("nan"),
                    "full_time_std": float("nan"),
                    "full_time_median": float("nan"),
                    "sample_time_mean": float("nan"),
                    "sample_time_std": float("nan"),
                    "sample_time_median": float("nan"),
                    "speedup": float("nan"),
                    "speedup_median": float("nan"),
                    "SND_full": float("nan"),
                    "device": str(device),
                    "dtype": cfg.dtype,
                    "timing_trials": cfg.timing_trials,
                    "num_envs": cfg.num_envs,
                    "rollout_steps": cfg.rollout_steps,
                    "obs_dim": cfg.obs_dim,
                    "act_dim": cfg.act_dim,
                    "oom": True,
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    elapsed = time.time() - t_start

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path} (elapsed {elapsed/60:.1f} min)")

    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in cfg.__dict__.items()
        },
        "elapsed_sec": elapsed,
        "rows_written": int(len(df)),
        "per_n_summary": [],
    }
    for n, grp in df.groupby("n"):
        if grp["full_time_mean"].isna().all():
            summary["per_n_summary"].append({"n": int(n), "oom": True})
            continue
        best = grp.loc[grp["speedup"].idxmax()]
        summary["per_n_summary"].append(
            {
                "n": int(n),
                "full_time_mean_ms": float(best["full_time_mean"]) * 1e3,
                "full_time_median_ms": float(best["full_time_median"]) * 1e3,
                "best_speedup_p": float(best["p"]),
                "best_speedup_mean": float(best["speedup"]),
                "best_speedup_median": float(best["speedup_median"]),
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
