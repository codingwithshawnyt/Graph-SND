"""Experiment 3: expander sparsification ablation (Theorem 10, Corollary 11).

Validates the deterministic fixed-G distortion bound and its expander
sparsification corollary.  For a d-regular spectral expander with
d = Theta(log n), Theorem 10 promises SND / SND_G^u in
[|E|/|P|, |E|*pi(G)/|P|], and the corollary gives O(log n) worst-case
relative distortion at |E| = Theta(n log n) edges.

The experiment measures:

1. Empirical distortion ratio SND_G^u / SND across four graph families
   (random d-regular, Bernoulli-p, k-NN, uniform-sample) at matched
   edge counts.
2. Spectral properties (lambda_2, spectral gap, Ramanujan bound check).
3. Edge forwarding index pi(G) to validate the theorem's mechanism.
4. Wall-clock timing for full SND vs graph SND.

Near-zero SND guard: configurations with SND < SND_FLOOR are skipped
and both the ratio and absolute distortion are reported.

CLI (CPU smoke test)::

    python experiments/exp3_expander_distortion.py \
        --n-agents 50 --device cpu --out results/exp3/smoke.csv

CLI (full GPU sweep)::

    python experiments/exp3_expander_distortion.py \
        --n-agents 50,100,200,500 --device cuda:0 \
        --out results/exp3/expander_distortion.csv

CLI (with n=100 trained checkpoint)::

    python experiments/exp3_expander_distortion.py \
        --n-agents 50,100,200,500 --device cuda:0 \
        --checkpoint results/graphsnd_n100_results/n100_overnight_iter500.metric.pt \
        --out results/exp3/expander_distortion.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from graphsnd.batched_policies import (
    BatchedGaussianMLPPolicy,
    load_batched_checkpoint,
)
from graphsnd.graphs import (
    bernoulli_edges,
    complete_edges,
    knn_edges,
    random_regular_edges,
    spectral_gap,
    uniform_size_edges,
)
from graphsnd.metrics import (
    graph_snd,
    graph_snd_from_rollouts,
    pairwise_behavioral_distance,
    pairwise_distances_on_edges,
    snd,
)
from graphsnd.policies import PolicyConfig

SND_FLOOR = 1e-8


def _ceil_log2(n: int) -> int:
    return max(1, math.ceil(math.log2(n)))


def _d_list_for_n(n: int) -> List[int]:
    logn = _ceil_log2(n)
    return [logn, 2 * logn, 4 * logn]


def _valid_d(n: int, d: int) -> bool:
    return d >= 1 and d < n and (n * d) % 2 == 0


def _clamp_d(n: int, d: int) -> int:
    if d >= n:
        d = (n - 1) if (n % 2 == 1) else (n - 2)
    if (n * d) % 2 != 0:
        d -= 1
    return max(1, d)


def forwarding_index(n: int, edges: Tensor) -> float:
    if edges.numel() == 0:
        return 0.0
    adj: Dict[int, List[int]] = defaultdict(list)
    for k in range(edges.shape[0]):
        u = int(edges[k, 0])
        v = int(edges[k, 1])
        adj[u].append(v)
        adj[v].append(u)

    edge_congestion: Dict[Tuple[int, int], float] = defaultdict(float)

    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        n_shortest = [0] * n
        n_shortest[s] = 1
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    n_shortest[v] = n_shortest[u]
                elif dist[v] == dist[u] + 1:
                    n_shortest[v] += n_shortest[u]

        pred_count: Dict[int, float] = defaultdict(float)
        for u in reversed(queue):
            if u == s:
                continue
            flow = 1.0 + pred_count[u]
            total_preds = 0
            preds = []
            for v in adj[u]:
                if dist[v] == dist[u] - 1:
                    total_preds += n_shortest[v]
                    preds.append(v)
            if total_preds == 0:
                continue
            for v in preds:
                frac = n_shortest[v] / total_preds
                load = flow * frac
                pred_count[v] += load
                e_key = (min(u, v), max(u, v))
                edge_congestion[e_key] += load

    if not edge_congestion:
        return 0.0
    return max(edge_congestion.values())


@dataclass
class ExpConfig:
    n_agents_list: Tuple[int, ...] = (50, 100, 200, 500)
    num_envs: int = 32
    rollout_steps: int = 128
    obs_dim: int = 18
    act_dim: int = 2
    u_range: float = 1.0
    hidden_sizes: Tuple[int, ...] = (64, 64)
    n_graph_seeds: int = 5
    timing_trials: int = 10
    warmup_trials: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    checkpoint: Optional[str] = None


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_frozen_rollouts(
    n_agents: int,
    cfg: ExpConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
            generator=gen,
            device=device,
            dtype=dtype,
        )
        * 2.0
        - 1.0
    )

    with torch.no_grad():
        means, stds = policy(obs)
        means = means.detach()
        stds = stds.detach()
    return means, stds


def build_checkpoint_rollouts(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
    cfg: ExpConfig,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_agents = ckpt["n_agents"]
    ckpt_config = ckpt["config"]

    policy_cfg = PolicyConfig(
        obs_dim=ckpt_config["obs_dim"],
        act_dim=ckpt_config["act_dim"],
        hidden_sizes=tuple(ckpt_config["hidden_sizes"]),
        u_range=ckpt_config["u_range"],
    )
    policy = BatchedGaussianMLPPolicy(
        n_agents=n_agents, config=policy_cfg, seed_base=cfg.seed
    ).to(device=device, dtype=dtype)
    load_batched_checkpoint(policy, ckpt)
    policy.eval()

    T_total = cfg.num_envs * cfg.rollout_steps
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 2)
    obs = (
        torch.rand(
            (n_agents, T_total, policy_cfg.obs_dim),
            generator=gen,
            device=device,
            dtype=dtype,
        )
        * 2.0
        - 1.0
    )

    with torch.no_grad():
        means, stds = policy(obs)
        means = means.detach()
        stds = stds.detach()
    return means, stds, n_agents


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


def time_graph_snd_single(
    means: torch.Tensor,
    stds: torch.Tensor,
    edges: torch.Tensor,
    trials: int,
    device: torch.device,
) -> Tuple[List[float], float]:
    times = []
    last_value = float("nan")
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        value = graph_snd_from_rollouts(means, stds, edges)
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_value = float(value.item())
    return times, last_value


GRAPH_FAMILIES = ("random_regular", "bernoulli", "uniform", "knn")


def run_single_config(
    n: int,
    d: int,
    graph_family: str,
    graph_seed: int,
    D: Tensor,
    snd_val: Tensor,
    means: Tensor,
    stds: Tensor,
    device: torch.device,
    cfg: ExpConfig,
    timing_trials: int,
    data_source: str,
) -> Optional[dict]:
    n_pairs = n * (n - 1) // 2
    target_edges = n * d // 2
    rng = np.random.default_rng(graph_seed)

    if graph_family == "random_regular":
        if not _valid_d(n, d):
            d = _clamp_d(n, d)
            if not _valid_d(n, d):
                return None
            target_edges = n * d // 2
        edges = random_regular_edges(n, d, rng)
    elif graph_family == "bernoulli":
        p = target_edges / n_pairs if n_pairs > 0 else 1.0
        p = min(p, 1.0)
        edges = bernoulli_edges(n, p, rng)
    elif graph_family == "uniform":
        m = min(target_edges, n_pairs)
        edges = uniform_size_edges(n, m, rng)
    elif graph_family == "knn":
        k = max(1, min(d, n - 1))
        feature_means = means.mean(dim=1).detach().cpu()
        edges = knn_edges(feature_means, k=k)
    else:
        raise ValueError(f"unknown graph_family: {graph_family}")

    num_edges = int(edges.shape[0])
    if num_edges == 0:
        return None

    lam2, gap, d_max, is_ram = spectral_gap(n, edges)

    snd_g_u = graph_snd(D, edges)
    snd_val_f = float(snd_val.item())
    snd_g_u_f = float(snd_g_u.item())
    abs_distortion = abs(snd_val_f - snd_g_u_f)

    if snd_val_f < SND_FLOOR:
        ratio = float("nan")
        ratio_reciprocal = float("nan")
    elif snd_g_u_f < SND_FLOOR:
        ratio = float("nan")
        ratio_reciprocal = float("nan")
    else:
        ratio = snd_val_f / snd_g_u_f
        ratio_reciprocal = snd_g_u_f / snd_val_f

    lower_bound = num_edges / n_pairs
    pi_G = forwarding_index(n, edges)
    upper_bound = num_edges * pi_G / n_pairs if n_pairs > 0 else float("nan")

    edges_full = complete_edges(n).to(device)
    ftimes, _ = time_full_snd(means, stds, edges_full, max(1, timing_trials), device)
    gtimes, _ = time_graph_snd_single(means, stds, edges.to(device), max(1, timing_trials), device)

    full_mean = float(np.mean(ftimes)) * 1e3
    graph_mean = float(np.mean(gtimes)) * 1e3

    ramanujan_bound = 2.0 * math.sqrt(max(d_max - 1, 0))

    return {
        "n": n,
        "d": d,
        "graph_family": graph_family,
        "graph_seed": graph_seed,
        "num_edges": num_edges,
        "n_pairs": n_pairs,
        "edge_fraction": lower_bound,
        "SND": snd_val_f,
        "SND_G_u": snd_g_u_f,
        "ratio": ratio,
        "ratio_reciprocal": ratio_reciprocal,
        "abs_distortion": abs_distortion,
        "lower_bound": lower_bound,
        "pi_G": pi_G,
        "upper_bound": upper_bound,
        "lambda_2": lam2,
        "spectral_gap": gap,
        "d_max": d_max,
        "is_ramanujan": bool(is_ram),
        "ramanujan_bound": ramanujan_bound,
        "time_full_ms": full_mean,
        "time_graph_ms": graph_mean,
        "data_source": data_source,
    }


def run_for_n(
    n_agents: int,
    cfg: ExpConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> List[dict]:
    print(f"[n={n_agents}] building frozen-init rollouts on {device}...")
    means, stds = build_frozen_rollouts(n_agents, cfg, device, dtype)
    rows = _run_on_rollouts(n_agents, means, stds, device, cfg, "synthetic")

    if cfg.checkpoint is not None and n_agents == 100:
        print(f"[n={n_agents}] loading checkpoint rollouts from {cfg.checkpoint}...")
        try:
            ckpt_means, ckpt_stds, ckpt_n = build_checkpoint_rollouts(
                cfg.checkpoint, device, dtype, cfg
            )
            assert ckpt_n == n_agents, f"checkpoint n={ckpt_n} != {n_agents}"
            ckpt_rows = _run_on_rollouts(
                n_agents, ckpt_means, ckpt_stds, device, cfg, "checkpoint"
            )
            rows.extend(ckpt_rows)
            del ckpt_means, ckpt_stds
        except Exception as exc:
            print(f"[n={n_agents}] checkpoint failed: {exc}")

    del means, stds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _run_on_rollouts(
    n_agents: int,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: torch.device,
    cfg: ExpConfig,
    data_source: str,
) -> List[dict]:
    print(f"[n={n_agents}::{data_source}] computing full D and SND...")
    D = pairwise_behavioral_distance(means, stds)
    snd_val = snd(D)
    print(
        f"[n={n_agents}::{data_source}] SND={snd_val.item():.6f} "
        f"n_pairs={n_agents*(n_agents-1)//2}"
    )

    if cfg.warmup_trials > 0:
        edges_full = complete_edges(n_agents).to(device)
        _ = time_full_snd(means, stds, edges_full, cfg.warmup_trials, device)

    d_values = _d_list_for_n(n_agents)
    rows: List[dict] = []

    for d in d_values:
        cd = d
        if not _valid_d(n_agents, d):
            cd = _clamp_d(n_agents, d)
        if not _valid_d(n_agents, cd):
            print(f"[n={n_agents}] d={d} (clamped={cd}) invalid, skipping")
            continue

        for gf in GRAPH_FAMILIES:
            for si in range(cfg.n_graph_seeds):
                graph_seed = cfg.seed + si * 10000 + cd * 100 + hash(gf) % 1000
                result = run_single_config(
                    n=n_agents,
                    d=cd,
                    graph_family=gf,
                    graph_seed=graph_seed,
                    D=D,
                    snd_val=snd_val,
                    means=means,
                    stds=stds,
                    device=device,
                    cfg=cfg,
                    timing_trials=cfg.timing_trials,
                    data_source=data_source,
                )
                if result is not None:
                    rows.append(result)
                    if si == 0:
                        print(
                            f"  [{gf} d={cd}] ratio={result['ratio']:.4f} "
                            f"|E|={result['num_edges']} "
                            f"lambda_2={result['lambda_2']:.4f} "
                            f"pi_G={result['pi_G']:.1f}"
                        )

    del D
    return rows


def _parse_n_list(s: str) -> Tuple[int, ...]:
    parts = [p for p in s.replace(",", " ").split() if p]
    return tuple(int(x) for x in parts)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expander sparsification ablation for Theorem 10 / Corollary 11."
    )
    parser.add_argument(
        "--n-agents", type=str, default="50",
        help="Comma/space-separated list of team sizes.",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--obs-dim", type=int, default=18)
    parser.add_argument("--act-dim", type=int, default=2)
    parser.add_argument("--n-graph-seeds", type=int, default=5)
    parser.add_argument("--timing-trials", type=int, default=10)
    parser.add_argument("--warmup-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to n=100 trained checkpoint for real-policy validation.",
    )
    parser.add_argument(
        "--out", type=str, default="results/exp3/expander_distortion.csv"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    dtype = _resolve_dtype(args.dtype)

    cfg = ExpConfig(
        n_agents_list=_parse_n_list(args.n_agents),
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        n_graph_seeds=args.n_graph_seeds,
        timing_trials=args.timing_trials,
        warmup_trials=args.warmup_trials,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        checkpoint=args.checkpoint,
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
            print(f"[n={n}] OOM: {exc}")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[n={n}] OOM (RuntimeError): {exc}")
            else:
                raise

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t_start
    print(f"\nWrote {len(df)} rows to {out_path} (elapsed {elapsed/60:.1f} min)")

    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "config": {
            "n_agents_list": list(cfg.n_agents_list),
            "num_envs": cfg.num_envs,
            "rollout_steps": cfg.rollout_steps,
            "obs_dim": cfg.obs_dim,
            "act_dim": cfg.act_dim,
            "n_graph_seeds": cfg.n_graph_seeds,
            "timing_trials": cfg.timing_trials,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "checkpoint": cfg.checkpoint,
            "seed": cfg.seed,
        },
        "elapsed_sec": elapsed,
        "rows_written": int(len(df)),
        "per_n_summary": [],
    }
    for n, grp in df.groupby("n"):
        n_info = {"n": int(n)}
        for src, src_grp in grp.groupby("data_source"):
            rr = src_grp[src_grp["graph_family"] == "random_regular"]
            if len(rr) > 0:
                n_info[f"{src}_expander_ratio_mean"] = float(rr["ratio"].mean())
                n_info[f"{src}_expander_ratio_recip_mean"] = float(
                    rr["ratio_reciprocal"].mean()
                )
        summary["per_n_summary"].append(n_info)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
