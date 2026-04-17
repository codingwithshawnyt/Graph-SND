"""Compare DiCo training with full SND vs Graph-SND (p=0.1, p=0.25).

Reads the three per-iteration CSV logs emitted by
``GraphSNDLoggingCallback`` (one per run) and produces a single
two-panel PDF:

- Left panel: ``SND(t)`` vs PPO iteration. Three curves plus a dashed
  horizontal line at ``SND_des`` so convergence is visually obvious.
- Right panel: metric wall-clock per update (ms) on a log-y axis,
  quantifying the Graph-SND speedup.

Styling (figure size, palette, line styles) matches the Graph-SND
paper's existing plots -- see ``experiments/plot_scaling.py`` in the
parent repo.

Example::

    python scripts/plot_graph_dico.py \\
        --full  logs/full/graph_snd_log.csv \\
        --p01   logs/p01/graph_snd_log.csv \\
        --p025  logs/p025/graph_snd_log.csv \\
        --out   graph_dico_comparison.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401  (kept per spec; helpful if the user extends)
import pandas as pd


SERIES = [
    # label, color, linestyle
    ("full",      "#1f77b4", "-"),
    (r"$p = 0.25$", "#ff7f0e", "--"),
    (r"$p = 0.1$",  "#2ca02c", ":"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        type=Path,
        required=True,
        help="CSV log from the full-SND run.",
    )
    parser.add_argument(
        "--p01",
        type=Path,
        required=True,
        help="CSV log from the Graph-SND p=0.1 run.",
    )
    parser.add_argument(
        "--p025",
        type=Path,
        required=True,
        help="CSV log from the Graph-SND p=0.25 run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("graph_dico_comparison.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=16,
        help="Number of agents (used in the left-panel title).",
    )
    parser.add_argument(
        "--snd-des",
        type=float,
        default=0.5,
        help="SND_des reference line on the left panel.",
    )
    return parser.parse_args()


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"iter", "snd_t", "metric_time_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV {path} is missing required columns: {sorted(missing)}"
        )
    return df.sort_values("iter").reset_index(drop=True)


def main() -> None:
    args = _parse_args()

    dataframes = {
        "full": _load(args.full),
        r"$p = 0.25$": _load(args.p025),
        r"$p = 0.1$": _load(args.p01),
    }

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))

    # ---- Left panel: SND(t) vs iter ------------------------------------
    for label, color, ls in SERIES:
        df = dataframes[label]
        ax_left.plot(
            df["iter"], df["snd_t"],
            linestyle=ls, color=color, linewidth=2, label=label,
        )
    ax_left.axhline(
        args.snd_des,
        linestyle="--", color="gray", alpha=0.7,
        label=r"$\mathrm{SND}_{\mathrm{des}}$",
    )
    ax_left.set_xlabel("PPO iteration")
    ax_left.set_ylabel(r"$\mathrm{SND}(t)$")
    ax_left.set_title(
        rf"DiCo convergence to $\mathrm{{SND}}_{{\mathrm{{des}}}}"
        rf"={args.snd_des}$ at $n={args.n_agents}$"
    )
    ax_left.legend(loc="lower right")
    ax_left.grid(alpha=0.3)

    # ---- Right panel: metric wall-clock per update ---------------------
    for label, color, ls in SERIES:
        df = dataframes[label]
        # Graph-SND spends zero time at empty iters; guard against log(0).
        y = df["metric_time_ms"].where(df["metric_time_ms"] > 0)
        ax_right.plot(
            df["iter"], y,
            linestyle=ls, color=color, linewidth=2, label=label,
        )
    ax_right.set_xlabel("PPO iteration")
    ax_right.set_ylabel("metric wall-clock per update (ms)")
    ax_right.set_yscale("log")
    ax_right.set_title("Per-update diversity computation cost")
    ax_right.legend(loc="lower right")
    ax_right.grid(alpha=0.3, which="both")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
