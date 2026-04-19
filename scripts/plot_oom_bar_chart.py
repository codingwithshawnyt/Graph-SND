"""Plot OOM barrier profiler results as a grouped bar chart.

Reads ``oom_profiling_results.csv`` (produced by
``scripts/profile_oom_barrier.py``) and creates a grouped bar chart
with team size on the x-axis, peak VRAM on the y-axis, and one bar
group per estimator (full, graph p=0.01, graph p=0.1).  OOM-crashed
entries are shown as a hashed bar capped at 24 GB (the 4090 limit)
with a red X marker.

Usage:
    python scripts/plot_oom_bar_chart.py \
        --input oom_profiling_results.csv \
        --output figures/oom_barrier.pdf \
        --gpu-vram-gb 24
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    )

ESTIMATOR_ORDER = ["full", "graph_p001", "graph_p01"]
ESTIMATOR_LABELS = {
    "full": "Full SND",
    "graph_p001": r"Graph-SND $p{=}0.01$",
    "graph_p01": r"Graph-SND $p{=}0.1$",
}
ESTIMATOR_COLORS = {
    "full": "#d62728",
    "graph_p001": "#2ca02c",
    "graph_p01": "#1f77b4",
}


def _load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "n_agents": int(row["n_agents"]),
                    "estimator": row["estimator"],
                    "p": float(row["p"]),
                    "vram_used_mb": float(row["vram_used_mb"])
                    if row["vram_used_mb"] not in ("", "nan")
                    else float("nan"),
                    "time_ms": float(row["time_ms"])
                    if row["time_ms"] not in ("", "nan")
                    else float("nan"),
                    "OOM_crashed": row["OOM_crashed"].strip().lower()
                    in ("true", "1"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot OOM barrier profiler results as grouped bar chart."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="oom_profiling_results.csv",
        help="CSV from profile_oom_barrier.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/oom_barrier.pdf",
        help="Output figure path.",
    )
    parser.add_argument(
        "--gpu-vram-gb",
        type=float,
        default=24.0,
        help="GPU VRAM capacity in GB (for the OOM ceiling line).",
    )
    args = parser.parse_args()

    rows = _load_csv(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    team_sizes = sorted({r["n_agents"] for r in rows})
    estimators = [e for e in ESTIMATOR_ORDER if any(r["estimator"] == e for r in rows)]
    if not estimators:
        estimators = sorted({r["estimator"] for r in rows})

    data: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        data[(r["n_agents"], r["estimator"])] = r

    n_groups = len(team_sizes)
    n_bars = len(estimators)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))
    gpu_vram_mb = args.gpu_vram_gb * 1024

    for i, est in enumerate(estimators):
        vram_vals: List[float] = []
        oom_flags: List[bool] = []
        for n in team_sizes:
            entry = data.get((n, est))
            if entry is None:
                vram_vals.append(0.0)
                oom_flags.append(False)
            else:
                mb = entry["vram_used_mb"]
                oom = entry["OOM_crashed"]
                if oom and np.isnan(mb):
                    vram_vals.append(gpu_vram_mb)
                elif np.isnan(mb):
                    vram_vals.append(0.0)
                else:
                    vram_vals.append(mb)
                oom_flags.append(oom)

        vram_gb = [v / 1024 for v in vram_vals]
        offsets = x - 0.4 + bar_width / 2 + i * bar_width

        label = ESTIMATOR_LABELS.get(est, est)
        color = ESTIMATOR_COLORS.get(est, None)
        bars = ax.bar(
            offsets,
            vram_gb,
            width=bar_width * 0.9,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        for j, (bar, oom) in enumerate(zip(bars, oom_flags)):
            if oom:
                bar.set_hatch("//")
                bar.set_edgecolor("#d62728")
                bar.set_linewidth(1.2)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#d62728",
                    fontweight="bold",
                )

    ax.axhline(
        y=args.gpu_vram_gb,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"RTX 4090 ({args.gpu_vram_gb:.0f} GB)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in team_sizes])
    ax.set_xlabel("Number of Agents ($n$)")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title(
        r"Metric evaluation: peak VRAM vs.\ team size ($n$)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
