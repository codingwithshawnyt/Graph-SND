"""OOM Barrier Profiler for Graph-SND vs Full SND.

Systematically probes team sizes n in [50, 100, 250, 500, 1000] and
measures whether Full SND OOMs on a 24 GB RTX 4090 while Graph-SND
survives. Outputs ``oom_profiling_results.csv`` with columns
[n_agents, estimator, p, vram_used_mb, time_ms, OOM_crashed].

Three estimators per n:
- ``full``: the C(n, 2) pairwise Wasserstein (the thing that OOMs)
- ``graph_p01`` with p=0.1 (practical Graph-SND default)
- ``graph_p01`` with p=0.01 (extreme sparsity, scales to n=1000)

Design: standalone (no BenchMARL / Hydra). Uses the exact
``het_control.snd.compute_behavioral_distance`` /
``het_control.graph_snd.compute_diversity`` functions that run inside
the DiCo training loop.

**Paper-language note (for Figure 5):** This profiler stress-tests the
*metric computation itself*, not the full PPO training loop (actor,
critic, optimiser states, replay buffer).  The claim it supports is:

    "Graph-SND removes the O(n^2) memory bottleneck of the diversity
    metric computation, which is the dominant cost for large n."

It does **not** claim that end-to-end PPO training at n=1000 fits on a
24 GB card — policy/optimiser memory grows linearly with n and may
still OOM.  Reviewers should be pointed to the metric-level claim.

Usage:
python scripts/profile_oom_barrier.py [--device cuda:0] [--output oom_profiling_results.csv]
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ControllingBehavioralDiversity-fork"))

from het_control.snd import compute_behavioral_distance
from het_control.graph_snd import compute_diversity


TEAM_SIZES = [50, 100, 250, 500, 1000]

BATCH_SIZE = 64

ACTION_DIM = 2

P_VALUES = [0.01, 0.1]

DUMMY_OBS_DIM = 18


def _is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def _build_dummy_actions(
    n_agents: int,
    batch_size: int,
    action_dim: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Create per-agent action tensors that mimic DiCo's estimate_snd output.

    Each element has shape (batch_size, action_dim) as if produced by
    the per-agent MLP heads inside HetControlMlpEmpirical.
    """
    return [
        torch.randn(batch_size, action_dim, device=device, dtype=torch.float32)
        for _ in range(n_agents)
    ]


def _profile_full_snd(
    agent_actions: List[torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Try full SND; catch OOM. Returns result dict."""
    is_cuda = _is_cuda(device)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    oom = False
    vram_mb = float("nan")
    time_ms = float("nan")
    snd_val = float("nan")

    try:
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = compute_behavioral_distance(agent_actions, just_mean=True).mean()
        if is_cuda:
            torch.cuda.synchronize(device)
        time_ms = (time.perf_counter() - t0) * 1000.0
        snd_val = float(result.item())
        if is_cuda:
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except torch.cuda.OutOfMemoryError:
        oom = True
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    return {
        "estimator": "full",
        "p": 1.0,
        "vram_used_mb": vram_mb,
        "time_ms": time_ms,
        "OOM_crashed": oom,
    }


def _profile_graph_snd(
    agent_actions: List[torch.Tensor],
    p: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Try Graph-SND with Bernoulli(p); catch OOM."""
    is_cuda = _is_cuda(device)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    oom = False
    vram_mb = float("nan")
    time_ms = float("nan")
    snd_val = float("nan")

    rng = torch.Generator()
    rng.manual_seed(42)

    try:
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = compute_diversity(
            agent_actions,
            estimator="graph_p01",
            p=p,
            rng=rng,
            just_mean=True,
        )
        if is_cuda:
            torch.cuda.synchronize(device)
        time_ms = (time.perf_counter() - t0) * 1000.0
        snd_val = float(result.item())
        if is_cuda:
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except torch.cuda.OutOfMemoryError:
        oom = True
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    label = f"graph_p{p}".replace(".", "")
    return {
        "estimator": label,
        "p": p,
        "vram_used_mb": vram_mb,
        "time_ms": time_ms,
        "OOM_crashed": oom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOM Barrier Profiler: Full SND vs Graph-SND."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="oom_profiling_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Simulated batch size per agent for dummy actions.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"OOM Barrier Profiler")
    print(f"  device:  {device}")
    print(f"  output:  {output_path}")
    print(f"  team sizes: {TEAM_SIZES}")
    print(f"  batch size: {args.batch_size}")
    print(f"  Graph-SND p values: {P_VALUES}")
    print()

    csv_fields = ["n_agents", "estimator", "p", "vram_used_mb", "time_ms", "OOM_crashed"]
    rows: List[Dict[str, Any]] = []

    for n in TEAM_SIZES:
        print(f"--- n_agents = {n} ---")
        agent_actions = _build_dummy_actions(n, args.batch_size, ACTION_DIM, device)

        # Full SND
        row_full = {"n_agents": n}
        row_full.update(_profile_full_snd(agent_actions, device))
        rows.append(row_full)
        status = "OOM" if row_full["OOM_crashed"] else f"OK ({row_full['vram_used_mb']:.0f} MB, {row_full['time_ms']:.1f} ms)"
        print(f"  full SND:      {status}")

        # Graph-SND at each p
        for p in P_VALUES:
            row_g = {"n_agents": n}
            row_g.update(_profile_graph_snd(agent_actions, p, device))
            rows.append(row_g)
            status = "OOM" if row_g["OOM_crashed"] else f"OK ({row_g['vram_used_mb']:.0f} MB, {row_g['time_ms']:.1f} ms)"
            print(f"  graph p={p:<4}: {status}")

        # Free memory before next team size
        del agent_actions
        gc.collect()
        if _is_cuda(device):
            torch.cuda.empty_cache()

    # Write CSV
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nResults written to {output_path}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
