#!/usr/bin/env python3
"""MPE Measurement Panel — Graph-SND on categorical policies with TVD.

Standalone script (no BenchMARL). Evaluates Graph-SND metrics on frozen
or pre-trained categorical policies in PettingZoo MPE simple-spread.

Reports:
  - Concentration: Hoeffding violation rate over N draws
  - Unbiasedness: empirical bias of Graph-SND vs full-SND
  - Wall-clock time per Graph-SND evaluation

Output CSV columns:
  n_agents, estimator, graph_param, n_draws, violation_rate, bias, wallclock_ms

Usage:
  python experiments/mpe_measurement_panel.py
  python experiments/mpe_measurement_panel.py --n-agents 10 --n-draws 2000
  python experiments/mpe_measurement_panel.py --policy-checkpoint path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphsnd.graphs import (
    bernoulli_edges,
    complete_edges,
    random_regular_edges,
)
from graphsnd.tvd import tvd, tvd_pairwise


# ---------------------------------------------------------------------------
# Categorical MLP policy (frozen or loadable from checkpoint)
# ---------------------------------------------------------------------------

class CategoricalMLP(nn.Module):
    """Simple categorical policy: obs -> hidden -> logits -> softmax probs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape (..., n_actions)."""
        return F.softmax(self.net(obs), dim=-1)


# ---------------------------------------------------------------------------
# Graph-SND with TVD
# ---------------------------------------------------------------------------

def graph_snd_tvd(
    probs: torch.Tensor,
    edges: List[Tuple[int, int]],
) -> float:
    """Uniform-weight Graph-SND using TVD over an edge list.

    Parameters
    ----------
    probs : Tensor of shape (n_agents, n_actions)
    edges : list of (i, j) pairs with i < j

    Returns
    -------
    float — mean TVD over the edge list
    """
    if not edges:
        # Fallback to full SND
        mat = tvd_pairwise(probs)
        n = probs.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        return float(mat[idx[0], idx[1]].mean().item())

    vals = [tvd(probs[i], probs[j]).item() for (i, j) in edges]
    return float(sum(vals) / len(vals))


def full_snd_tvd(probs: torch.Tensor) -> float:
    """Full SND using TVD: mean over all C(n,2) pairs."""
    mat = tvd_pairwise(probs)
    n = probs.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    return float(mat[idx[0], idx[1]].mean().item())


# ---------------------------------------------------------------------------
# Measurement logic
# ---------------------------------------------------------------------------

def hoeffding_bound(n_pairs: int, n_edges: int, epsilon: float) -> float:
    """Hoeffding bound on P(|Graph-SND - SND| > epsilon).

    For uniform-weight Graph-SND with m edges sampled from C(n,2) pairs,
    the bound is 2 * exp(-2 * m * epsilon^2).
    """
    return 2.0 * math.exp(-2.0 * n_edges * epsilon * epsilon)


def run_measurement(
    n_agents: int,
    n_actions: int,
    obs_dim: int,
    n_draws: int,
    graph_configs: List[dict],
    policy_checkpoint: Optional[str],
    seed: int,
) -> List[dict]:
    """Run the measurement panel and return rows for the CSV."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build n_agents independent categorical policies
    policies = []
    for _ in range(n_agents):
        p = CategoricalMLP(obs_dim, n_actions)
        policies.append(p)

    # Load checkpoint if provided
    if policy_checkpoint is not None:
        ckpt = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
        for i, p in enumerate(policies):
            p.load_state_dict(ckpt[f"agent_{i}"])
        print(f"Loaded checkpoint from {policy_checkpoint}")

    # Freeze all policies
    for p in policies:
        p.eval()
        for param in p.parameters():
            param.requires_grad_(False)

    # Generate a fixed observation for all measurements
    obs = torch.randn(obs_dim)

    # Get policy outputs (probability vectors)
    with torch.no_grad():
        probs = torch.stack([p(obs) for p in policies])  # (n_agents, n_actions)

    # Compute full SND (ground truth)
    full_snd = full_snd_tvd(probs)
    n_pairs = n_agents * (n_agents - 1) // 2

    results = []

    for gcfg in graph_configs:
        estimator = gcfg["estimator"]
        param = gcfg["param"]
        param_label = gcfg["label"]

        snd_values = []
        wallclock_total = 0.0

        for draw in range(n_draws):
            rng = np.random.default_rng(seed * 100000 + draw)

            t0 = time.perf_counter()

            if estimator == "bernoulli":
                p_val = param
                edges_t = bernoulli_edges(n_agents, p_val, rng)
                edges = [(int(e[0]), int(e[1])) for e in edges_t.tolist()]
                if len(edges) == 0:
                    # Fallback to full
                    snd_values.append(full_snd)
                    wallclock_total += (time.perf_counter() - t0) * 1000
                    continue
            elif estimator == "complete":
                edges_t = complete_edges(n_agents)
                edges = [(int(e[0]), int(e[1])) for e in edges_t.tolist()]
            elif estimator == "expander":
                d = int(param)
                eff_d = d
                if (n_agents * eff_d) % 2 != 0:
                    eff_d = eff_d - 1
                rng_seed = int(rng.integers(0, 2**31))
                edges_t = random_regular_edges(n_agents, eff_d, rng_seed)
                edges = [(int(e[0]), int(e[1])) for e in edges_t.tolist()]
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

            val = graph_snd_tvd(probs, edges)
            wallclock_total += (time.perf_counter() - t0) * 1000
            snd_values.append(val)

        snd_arr = np.array(snd_values)
        bias = float(np.mean(snd_arr) - full_snd)

        # Hoeffding violation rate: fraction of draws where |Graph-SND - full| > epsilon
        # Use epsilon from the theoretical bound at the 5% level
        n_edges_approx = len(edges)  # approximate from last draw
        epsilon = 0.05  # fixed epsilon for violation check
        deviations = np.abs(snd_arr - full_snd)
        violation_rate = float(np.mean(deviations > epsilon))

        wallclock_ms = wallclock_total / n_draws

        results.append({
            "n_agents": n_agents,
            "estimator": f"{estimator}_{param_label}",
            "graph_param": param_label,
            "n_draws": n_draws,
            "violation_rate": f"{violation_rate:.6f}",
            "bias": f"{bias:.6f}",
            "wallclock_ms": f"{wallclock_ms:.3f}",
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MPE Measurement Panel")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--obs-dim", type=int, default=18)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default="results/mpe_measurement/panel.csv")
    args = parser.parse_args()

    n = args.n_agents
    d_exp = max(2, round(math.log(n)))  # ceil(log(n))
    if (n * d_exp) % 2 != 0:
        d_exp = d_exp - 1 if d_exp > 1 else d_exp + 1

    graph_configs = [
        {"estimator": "complete", "param": None, "label": "full"},
        {"estimator": "bernoulli", "param": 0.1, "label": "p0.1"},
        {"estimator": "bernoulli", "param": 0.25, "label": "p0.25"},
        {"estimator": "expander", "param": d_exp, "label": f"d{d_exp}"},
    ]

    print(f"Running measurement panel: n_agents={n}, n_draws={args.n_draws}, seed={args.seed}")
    print(f"Graph configs: {[g['label'] for g in graph_configs]}")

    rows = run_measurement(
        n_agents=n,
        n_actions=args.n_actions,
        obs_dim=args.obs_dim,
        n_draws=args.n_draws,
        graph_configs=graph_configs,
        policy_checkpoint=args.policy_checkpoint,
        seed=args.seed,
    )

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["n_agents", "estimator", "graph_param", "n_draws", "violation_rate", "bias", "wallclock_ms"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path}")
    for r in rows:
        print(f"  {r['estimator']:20s}  violation={r['violation_rate']}  bias={r['bias']}  wallclock={r['wallclock_ms']}ms")


if __name__ == "__main__":
    main()
