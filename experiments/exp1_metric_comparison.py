"""Experiment 1: metric validation on VMAS Multi-Agent Goal Navigation.

Given frozen heterogeneous Gaussian policies (produced by
``training/train_navigation.py``), this script runs the three
validations of the paper's core Graph-SND results plus a runtime
comparison:

1. **Proposition 1 (recovery on K_n)** at n=4. Computes full SND and
   complete-graph Graph-SND from the same rollout-derived distance
   matrix ``D``; they should match to floating-point precision.
2. **Proposition 5 (HT unbiasedness)** at n=8. For each ``p`` in a
   small grid, draws many Bernoulli graphs, evaluates the
   Horvitz-Thompson estimator on each, and reports the sample mean
   versus the true SND with a 95 percent CI on the mean. The CI should
   bracket zero bias.
3. **Theorem 6 / Remark 3 (concentration)** at n=16. For each ``m`` in
   a grid, draws many uniform-size edge samples, evaluates the
   uniform-weight estimator on each, and plots the empirical
   ``P(|estimator - SND| >= t)`` against the Hoeffding and Serfling
   bounds. The empirical tail should sit under both.
4. **Timing**: wall-clock cost of full SND (all pairs) versus
   Graph-SND evaluated on a Bernoulli-sampled subgraph, both computed
   directly from the rollout ``(means, stds)`` so only the pairs in
   ``E`` are materialised.

All numeric outputs are written as tidy CSVs under ``results/exp1/``.

CLI::

    python experiments/exp1_metric_comparison.py \
        --checkpoint-dir checkpoints \
        --out results/exp1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from graphsnd.graphs import (
    bernoulli_edges,
    complete_edges,
    uniform_size_edges,
)
from graphsnd.metrics import (
    graph_snd,
    graph_snd_from_rollouts,
    hoeffding_bound,
    ht_estimator,
    pairwise_behavioral_distance,
    pairwise_distances_on_edges,
    serfling_bound,
    snd,
    uniform_sample_estimator,
)
from graphsnd.policies import load_checkpoint
from graphsnd.rollouts import collect_rollouts, evaluate_policies_on_observations


@dataclass
class ExperimentConfig:
    scenario: str = "navigation"
    num_envs: int = 32
    rollout_steps: int = 128
    max_steps: int = 200
    device: str = "cpu"
    seed: int = 42
    n_agents_list: tuple = (4, 8, 16)
    iter_tags: tuple = ("iter0", "iter100")
    ht_p_values: tuple = (0.1, 0.25, 0.5, 0.75)
    ht_trials: int = 2000
    conc_m_fracs: tuple = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
    conc_trials: int = 2000
    conc_delta: float = 0.1
    timing_trials: int = 20


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_policies_for_n(
    ckpt_dir: Path, n: int, tag: str, device: str
):
    """Load ``(policies, meta)`` for a given ``(n, iter-tag)``.

    ``tag`` is e.g. ``"iter0"`` or ``"iter100"``.
    """
    path = ckpt_dir / f"n{n}_{tag}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Run training/train_navigation.py "
            f"with --n-agents {n} first."
        )
    policies, _, extra = load_checkpoint(path, map_location=device)
    for p in policies:
        p.eval().to(device)
    return policies, extra


def build_rollouts_and_distances(
    n: int, policies, cfg: ExperimentConfig, device: str
):
    """Roll the env out once and return ``D``, ``(means, stds)``, and ``R``.

    ``R`` is the rollout observation count: ``n * num_envs * rollout_steps``.
    """
    import vmas

    env = vmas.make_env(
        scenario=cfg.scenario,
        num_envs=cfg.num_envs,
        device=device,
        continuous_actions=True,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
        n_agents=n,
    )
    if env.n_agents != n:
        raise RuntimeError(
            f"Requested n={n}, env built with n={env.n_agents}"
        )
    batch = collect_rollouts(
        env, policies, n_steps=cfg.rollout_steps, deterministic=False, device=torch.device(device)
    )
    means, stds = evaluate_policies_on_observations(policies, batch.observations)
    D = pairwise_behavioral_distance(means, stds)
    return D, means, stds, int(batch.observations.shape[0] * batch.observations.shape[1])


def run_prop1(
    cfg: ExperimentConfig,
    out_dir: Path,
    ckpt_dir: Path,
) -> pd.DataFrame:
    """Proposition 1: exact recovery of SND on K_n for n=4."""
    print("[Prop 1] recovery on K_n at n=4...")
    rows = []
    for tag in cfg.iter_tags:
        policies, extra = load_policies_for_n(ckpt_dir, 4, tag, cfg.device)
        D, means, stds, R = build_rollouts_and_distances(4, policies, cfg, cfg.device)

        full_snd = float(snd(D).item())
        edges = complete_edges(4)
        graph_snd_matrix = float(graph_snd(D, edges).item())
        graph_snd_direct = float(graph_snd_from_rollouts(means, stds, edges).item())
        abs_err_matrix = abs(full_snd - graph_snd_matrix)
        abs_err_direct = abs(full_snd - graph_snd_direct)

        rows.append(
            {
                "iter_tag": tag,
                "n": 4,
                "R": R,
                "SND_full": full_snd,
                "GraphSND_K4_via_D": graph_snd_matrix,
                "GraphSND_K4_direct": graph_snd_direct,
                "abs_err_D_path": abs_err_matrix,
                "abs_err_direct_path": abs_err_direct,
            }
        )
        print(
            f"  {tag}: SND={full_snd:.8f}  "
            f"GraphSND(K_4)={graph_snd_matrix:.8f}  "
            f"|err|={abs_err_matrix:.2e}"
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "recovery.csv", index=False)
    return df


def run_prop5(
    cfg: ExperimentConfig,
    out_dir: Path,
    ckpt_dir: Path,
) -> pd.DataFrame:
    """Proposition 5: HT estimator unbiasedness at n=8."""
    print("[Prop 5] HT unbiasedness at n=8...")
    rows = []
    for tag in cfg.iter_tags:
        policies, extra = load_policies_for_n(ckpt_dir, 8, tag, cfg.device)
        D, _, _, R = build_rollouts_and_distances(8, policies, cfg, cfg.device)
        full_snd = float(snd(D).item())

        rng = np.random.default_rng(cfg.seed + {"iter0": 11, "iter100": 22}[tag])
        for p in cfg.ht_p_values:
            ests = np.empty(cfg.ht_trials, dtype=np.float64)
            for k in range(cfg.ht_trials):
                ests[k] = float(ht_estimator(D, p=float(p), rng=rng).item())
            mean = float(ests.mean())
            std = float(ests.std(ddof=1))
            se = std / math.sqrt(cfg.ht_trials)
            ci95 = 1.96 * se
            rows.append(
                {
                    "iter_tag": tag,
                    "n": 8,
                    "R": R,
                    "p": float(p),
                    "trials": cfg.ht_trials,
                    "SND_full": full_snd,
                    "HT_mean": mean,
                    "HT_std": std,
                    "HT_se": se,
                    "HT_CI95_halfwidth": ci95,
                    "bias": mean - full_snd,
                    "bias_over_se": (mean - full_snd) / se if se > 0 else 0.0,
                }
            )
            print(
                f"  {tag} p={p:.2f}: mean={mean:.6f}  SND={full_snd:.6f}  "
                f"bias={mean - full_snd:+.2e}  se={se:.2e}  "
                f"bias/se={(mean - full_snd) / se if se > 0 else 0:+.2f}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unbiasedness.csv", index=False)
    return df


def run_thm6(
    cfg: ExperimentConfig,
    out_dir: Path,
    ckpt_dir: Path,
) -> pd.DataFrame:
    """Theorem 6 / Remark 3: concentration at n=16."""
    print("[Thm 6] concentration at n=16...")
    rows = []
    for tag in cfg.iter_tags:
        policies, extra = load_policies_for_n(ckpt_dir, 16, tag, cfg.device)
        D, _, _, R = build_rollouts_and_distances(16, policies, cfg, cfg.device)
        full_snd = float(snd(D).item())
        d_max = float(D.max().item())
        n = 16
        n_pairs = n * (n - 1) // 2

        rng = np.random.default_rng(cfg.seed + {"iter0": 33, "iter100": 44}[tag])
        for frac in cfg.conc_m_fracs:
            m = max(1, int(round(frac * n_pairs)))
            ests = np.empty(cfg.conc_trials, dtype=np.float64)
            for k in range(cfg.conc_trials):
                ests[k] = float(uniform_sample_estimator(D, m=m, rng=rng).item())
            abs_err = np.abs(ests - full_snd)
            t_hoeff = hoeffding_bound(d_max, m, cfg.conc_delta)
            t_serfl = serfling_bound(d_max, m, n_pairs, cfg.conc_delta)
            tail_emp = float((abs_err >= t_hoeff).mean())
            tail_emp_serfl = float((abs_err >= t_serfl).mean())
            rows.append(
                {
                    "iter_tag": tag,
                    "n": n,
                    "R": R,
                    "m": m,
                    "n_pairs": n_pairs,
                    "m_frac": float(frac),
                    "trials": cfg.conc_trials,
                    "SND_full": full_snd,
                    "D_max": d_max,
                    "mean_abs_err": float(abs_err.mean()),
                    "p95_abs_err": float(np.percentile(abs_err, 95)),
                    "p99_abs_err": float(np.percentile(abs_err, 99)),
                    "delta": cfg.conc_delta,
                    "hoeffding_t": t_hoeff,
                    "serfling_t": t_serfl,
                    "empirical_tail_hoeff": tail_emp,
                    "empirical_tail_serfl": tail_emp_serfl,
                }
            )
            print(
                f"  {tag} m={m:3d} ({frac*100:4.1f}%): "
                f"p95|err|={np.percentile(abs_err, 95):.4f}  "
                f"t_H={t_hoeff:.4f}  t_S={t_serfl:.4f}  "
                f"emp_tail_H={tail_emp:.3f}  emp_tail_S={tail_emp_serfl:.3f}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "concentration.csv", index=False)
    return df


def run_timing(
    cfg: ExperimentConfig,
    out_dir: Path,
    ckpt_dir: Path,
) -> pd.DataFrame:
    """Timing sweep: full SND vs Graph-SND on a Bernoulli-sampled subgraph."""
    print("[Timing] full SND vs Graph-SND (sampled) at n=4,8,16...")
    rows = []
    tag = "iter100"
    p_values = (0.1, 0.25, 0.5, 0.75, 1.0)
    for n in cfg.n_agents_list:
        policies, extra = load_policies_for_n(ckpt_dir, n, tag, cfg.device)
        D, means, stds, R = build_rollouts_and_distances(n, policies, cfg, cfg.device)
        full_snd = float(snd(D).item())
        n_pairs = n * (n - 1) // 2

        full_times = []
        for _ in range(cfg.timing_trials):
            t0 = time.perf_counter()
            D_loop = pairwise_behavioral_distance(means, stds)
            s = snd(D_loop)
            _ = s.item()
            full_times.append(time.perf_counter() - t0)
        full_mean = float(np.mean(full_times))
        full_std = float(np.std(full_times, ddof=1)) if len(full_times) > 1 else 0.0

        for p in p_values:
            rng = np.random.default_rng(cfg.seed + int(100 * p) + n)
            sample_times = []
            sample_sizes = []
            for _ in range(cfg.timing_trials):
                t0 = time.perf_counter()
                edges = bernoulli_edges(n, float(p), rng)
                if edges.shape[0] == 0:
                    _ = 0.0
                else:
                    w = torch.full((edges.shape[0],), 1.0 / float(p))
                    d_vals = pairwise_distances_on_edges(means, stds, edges)
                    est = (w * d_vals).sum() / n_pairs
                    _ = float(est.item())
                sample_times.append(time.perf_counter() - t0)
                sample_sizes.append(int(edges.shape[0]))
            smean = float(np.mean(sample_times))
            sstd = float(np.std(sample_times, ddof=1)) if len(sample_times) > 1 else 0.0
            rows.append(
                {
                    "n": n,
                    "R": R,
                    "n_pairs": n_pairs,
                    "p": float(p),
                    "mean_edges": float(np.mean(sample_sizes)),
                    "full_time_mean": full_mean,
                    "full_time_std": full_std,
                    "sample_time_mean": smean,
                    "sample_time_std": sstd,
                    "speedup": full_mean / smean if smean > 0 else float("inf"),
                    "SND_full": full_snd,
                }
            )
            print(
                f"  n={n:2d} p={p:.2f}: full={full_mean*1e3:.1f}ms  "
                f"sample={smean*1e3:.1f}ms  speedup={full_mean / smean:5.2f}x  "
                f"|E|={np.mean(sample_sizes):.1f}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "timing.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph-SND Experiment 1: metric-comparison on VMAS navigation."
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--out", type=str, default="results/exp1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ht-trials", type=int, default=2000)
    parser.add_argument("--conc-trials", type=int, default=2000)
    parser.add_argument("--timing-trials", type=int, default=20)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        device=args.device,
        seed=args.seed,
        ht_trials=args.ht_trials,
        conc_trials=args.conc_trials,
        timing_trials=args.timing_trials,
    )
    set_seeds(cfg.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_recov = run_prop1(cfg, out_dir, ckpt_dir)
    df_unbias = run_prop5(cfg, out_dir, ckpt_dir)
    df_conc = run_thm6(cfg, out_dir, ckpt_dir)
    df_timing = run_timing(cfg, out_dir, ckpt_dir)

    summary_path = out_dir / "summary.json"
    summary = {
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in cfg.__dict__.items()
        },
        "outputs": {
            "recovery_csv": str(out_dir / "recovery.csv"),
            "unbiasedness_csv": str(out_dir / "unbiasedness.csv"),
            "concentration_csv": str(out_dir / "concentration.csv"),
            "timing_csv": str(out_dir / "timing.csv"),
        },
        "summary_numbers": {
            "prop1_max_abs_err": float(df_recov["abs_err_D_path"].abs().max()),
            "prop1_max_abs_err_direct": float(df_recov["abs_err_direct_path"].abs().max()),
            "prop5_max_bias_over_se": float(df_unbias["bias_over_se"].abs().max()),
            "thm6_max_empirical_tail_hoeff": float(df_conc["empirical_tail_hoeff"].max()),
            "thm6_max_empirical_tail_serfl": float(df_conc["empirical_tail_serfl"].max()),
            "thm6_delta": cfg.conc_delta,
            "timing_best_speedup": float(df_timing["speedup"].max()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {summary_path}")
    for k, v in summary["summary_numbers"].items():
        print(f"  {k:40s}: {v}")


if __name__ == "__main__":
    main()
