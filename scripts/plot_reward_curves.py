"""Plot reward curves from Graph-SND DiCo CSV logs.

Reads one or more ``graph_snd_log.csv`` files (produced by
:class:`GraphSNDLoggingCallback`), groups them by estimator, and plots
reward-mean vs iteration with confidence bands (if multiple seeds are
provided for the same estimator).

Usage:
    python scripts/plot_reward_curves.py \
        ippo:outputs/ippo/graph_snd_log.csv \
        full:outputs/full/graph_snd_log.csv \
        knn:outputs/knn/graph_snd_log.csv \
        --output figures/reward_curves.pdf

    # Multiple seeds per estimator:
    python scripts/plot_reward_curves.py \
        "ippo:outputs/seed0_ippo/log.csv,outputs/seed1_ippo/log.csv" \
        "full:outputs/seed0_full/log.csv,outputs/seed1_full/log.csv" \
        --output figures/reward_curves.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    )


LABEL_MAP = {
    "full": "Full SND (Global DiCo)",
    "graph_p01": r"Graph-SND $p{=}0.1$",
    "graph_p025": r"Graph-SND $p{=}0.25$",
    "knn": r"Graph-SND $k$-NN ($k{=}3$)",
}


def _load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _parse_series(
    csv_paths: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    iters_list: List[List[float]] = []
    rewards_list: List[List[float]] = []
    for path in csv_paths:
        rows = _load_csv(path)
        iters_list.append([float(r.get("iter", 0)) for r in rows])
        rewards_list.append([float(r.get("reward_mean", float("nan"))) for r in rows])
    min_len = min(len(x) for x in iters_list)
    if min_len == 0:
        return np.array([]), np.array([])
    iters = np.array(iters_list[0][:min_len])
    rewards = np.array([r[:min_len] for r in rewards_list])
    return iters, rewards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reward curves from Graph-SND DiCo CSV logs."
    )
    parser.add_argument(
        "series",
        nargs="+",
        help='Label:csv_path pairs, e.g. "ippo:log.csv". '
        "Multiple seeds: 'label:path1,path2,path3'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/reward_curves.pdf",
        help="Output figure path.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-mean window for smoothing (1 = no smoothing).",
    )
    args = parser.parse_args()

    all_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spec in args.series:
        label, paths_str = spec.split(":", 1)
        paths = [p.strip() for p in paths_str.split(",")]
        iters, rewards = _parse_series(paths)
        if iters.size == 0:
            print(f"WARNING: no data for {label!r}, skipping.")
            continue
        all_data[label] = (iters, rewards)

    if not all_data:
        raise SystemExit("No data loaded. Check your CSV paths.")

    fig, ax = plt.subplots(figsize=(7, 4))

    for label, (iters, rewards) in all_data.items():
        display_label = LABEL_MAP.get(label, label)
        if rewards.ndim == 1:
            mean = rewards
            std = None
        else:
            mean = np.nanmean(rewards, axis=0)
            std = np.nanstd(rewards, axis=0)

        if args.smooth > 1 and len(mean) >= args.smooth:
            kernel = np.ones(args.smooth) / args.smooth
            mean = np.convolve(mean, kernel, mode="valid")
            iters_plot = iters[: len(mean)]
            if std is not None:
                std = np.convolve(std, kernel, mode="valid")
        else:
            iters_plot = iters

        ax.plot(iters_plot, mean, label=display_label, linewidth=1.5)
        if std is not None:
            ax.fill_between(
                iters_plot,
                mean - std,
                mean + std,
                alpha=0.2,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("VMAS Dispersion: mean reward vs.\ iteration")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
