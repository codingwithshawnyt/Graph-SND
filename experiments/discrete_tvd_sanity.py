"""Discrete-action Graph-SND sanity check with total variation distance.

This script is intentionally standalone (no RL training): it samples random
categorical policies for n agents over k discrete actions, computes full SND
from pairwise TVD distances, and verifies Bernoulli-graph Graph-SND estimator
unbiasedness/concentration empirically over many graph draws.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Pair = Tuple[int, int]


def sample_dirichlet(alpha: float, k: int, rng: random.Random) -> List[float]:
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(xs)
    return [x / s for x in xs]


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def all_pairs(n: int) -> List[Pair]:
    return list(itertools.combinations(range(n), 2))


def bernoulli_graph_sample(pairs: Sequence[Pair], prob: float, rng: random.Random) -> List[Pair]:
    return [e for e in pairs if rng.random() < prob]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_cell(
    n_agents: int,
    n_actions: int,
    alpha: float,
    p_edge: float,
    n_draws: int,
    delta: float,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    policies = [sample_dirichlet(alpha=alpha, k=n_actions, rng=rng) for _ in range(n_agents)]
    pairs = all_pairs(n_agents)

    dvals = {(i, j): tvd(policies[i], policies[j]) for (i, j) in pairs}
    snd_full = mean([dvals[e] for e in pairs])
    dmax = 1.0  # TVD is in [0, 1]

    hat_vals: List[float] = []
    violations = 0
    nonempty = 0
    m_values: List[int] = []
    for _ in range(n_draws):
        edges = bernoulli_graph_sample(pairs=pairs, prob=p_edge, rng=rng)
        m = len(edges)
        if m == 0:
            continue
        nonempty += 1
        m_values.append(m)
        hat = mean([dvals[e] for e in edges])
        hat_vals.append(hat)

        bound = dmax * math.sqrt(math.log(2.0 / delta) / (2.0 * m))
        if abs(hat - snd_full) > bound:
            violations += 1

    if not hat_vals:
        raise RuntimeError("No non-empty sampled graphs; increase p_edge or n_draws.")

    mean_hat = mean(hat_vals)
    bias = mean_hat - snd_full
    sd = math.sqrt(mean([(x - mean_hat) ** 2 for x in hat_vals]))
    mean_m = mean(m_values)
    violation_rate = violations / nonempty

    return {
        "n_agents": n_agents,
        "n_actions": n_actions,
        "alpha": alpha,
        "p_edge": p_edge,
        "draws_total": n_draws,
        "draws_nonempty": nonempty,
        "mean_edges": mean_m,
        "snd_full": snd_full,
        "mean_hat": mean_hat,
        "bias": bias,
        "std_hat": sd,
        "violation_rate": violation_rate,
        "delta": delta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-actions", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--p-list", type=str, default="0.1,0.3,0.5")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/discrete_tvd_sanity/discrete_tvd_sanity.csv"),
    )
    args = ap.parse_args()

    p_list = [float(x.strip()) for x in args.p_list.split(",") if x.strip()]
    rows = [
        run_cell(
            n_agents=args.n_agents,
            n_actions=args.n_actions,
            alpha=args.alpha,
            p_edge=p,
            n_draws=args.draws,
            delta=args.delta,
            seed=args.seed + i,
        )
        for i, p in enumerate(p_list)
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out_csv}")
    print("p_edge | mean_edges | snd_full | mean_hat | bias | std_hat | violation_rate")
    for r in rows:
        print(
            f"{r['p_edge']:.2f}   | {r['mean_edges']:.2f}      | {r['snd_full']:.5f}  | "
            f"{r['mean_hat']:.5f} | {r['bias']:+.5f} | {r['std_hat']:.5f} | {r['violation_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
