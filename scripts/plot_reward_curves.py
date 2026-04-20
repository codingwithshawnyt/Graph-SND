"""Plot reward / applied-SND / metric-time curves from Graph-SND DiCo CSV logs.

Reads one or more ``graph_snd_log.csv`` files (produced by
:class:`GraphSNDLoggingCallback`), groups them by estimator, and plots
either a single reward-vs-iteration panel or a 3-panel publication
figure (reward, applied SND, metric wall-clock).

Usage::

    # Single-panel reward-only (backward-compatible default)
    python scripts/plot_reward_curves.py \
        ippo:outputs/ippo/graph_snd_log.csv \
        knn:outputs/knn/graph_snd_log.csv \
        full:outputs/full/graph_snd_log.csv \
        --output Paper/figures/neurips_knn_plot.pdf

    # Multi-seed, 3-panel publication figure
    python scripts/plot_reward_curves.py \
        --figure-type panels \
        "ippo:r/seed0/ippo/log.csv,r/seed1/ippo/log.csv,r/seed2/ippo/log.csv" \
        "knn:r/seed0/knn/log.csv,r/seed1/knn/log.csv,r/seed2/knn/log.csv" \
        "full:r/seed0/full/log.csv,r/seed1/full/log.csv,r/seed2/full/log.csv" \
        --output Paper/figures/neurips_knn_plot.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "ippo": "IPPO Baseline",
    "full": "Full SND",
    "graph_p01": r"Graph-SND $p{=}0.1$",
    "graph_p025": r"Graph-SND $p{=}0.25$",
    "knn": "k-NN Graph-SND",
}

# Fixed palette so independent plots stay visually consistent across the paper.
COLOR_MAP = {
    "ippo": "#1f77b4",
    "knn":  "#ff7f0e",
    "full": "#2ca02c",
    "graph_p01": "#d62728",
    "graph_p025": "#9467bd",
}


def _load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(x: Optional[str]) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def _parse_series(
    csv_paths: List[str],
    columns: Tuple[str, ...],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (iters[min_len], {col: array(n_seeds, min_len) or None}).

    Values missing from the CSV for a given column yield ``None`` so callers
    can skip the corresponding panel.
    """
    per_seed_iters: List[List[float]] = []
    per_seed_cols: Dict[str, List[List[float]]] = {c: [] for c in columns}
    missing_cols: Dict[str, bool] = {c: False for c in columns}

    for path in csv_paths:
        rows = _load_csv(path)
        per_seed_iters.append([_safe_float(r.get("iter", "0")) for r in rows])
        for c in columns:
            if rows and c not in rows[0]:
                missing_cols[c] = True
                per_seed_cols[c].append([float("nan")] * len(rows))
            else:
                per_seed_cols[c].append([_safe_float(r.get(c, "")) for r in rows])

    if not per_seed_iters:
        return np.array([]), {c: None for c in columns}

    min_len = min(len(x) for x in per_seed_iters)
    if min_len == 0:
        return np.array([]), {c: None for c in columns}

    iters = np.array(per_seed_iters[0][:min_len])
    out: Dict[str, np.ndarray] = {}
    for c in columns:
        if missing_cols[c]:
            out[c] = None
            continue
        out[c] = np.array([seed_col[:min_len] for seed_col in per_seed_cols[c]])
    return iters, out


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def _aggregate(
    data: np.ndarray,
    smooth: int,
    aggregator: str = "mean_std",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Aggregate across the leading seed axis, then smooth.

    Returns ``(center, lower_or_None, upper_or_None)``. When
    ``aggregator == "mean_std"`` the band is mean ± std. When
    ``aggregator == "median_iqr"`` the band is [p25, p75]. Median + IQR
    is the right choice for heavy-tailed per-call wall-clock timing:
    mean/std there is dominated by rare long tails (GC, CUDA sync) and
    produces a misleading variance band.
    """
    if data.ndim == 1:
        center = _smooth(data, smooth)
        return center, None, None

    if aggregator == "median_iqr":
        center = np.nanmedian(data, axis=0)
        lower = np.nanpercentile(data, 25, axis=0)
        upper = np.nanpercentile(data, 75, axis=0)
    else:
        center = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        lower = center - std
        upper = center + std

    center = _smooth(center, smooth)
    lower = _smooth(lower, smooth)
    upper = _smooth(upper, smooth)
    return center, lower, upper


def _plot_panel(
    ax,
    all_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    column: str,
    ylabel: str,
    title: str,
    smooth: int,
    log_y: bool = False,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
    aggregator: str = "mean_std",
    annotate_speedup: bool = False,
) -> bool:
    """Draw one panel. Returns True if anything was plotted."""
    any_drawn = False
    per_series_medians: Dict[str, float] = {}
    for label, (iters, data_by_col) in all_data.items():
        raw = data_by_col.get(column)
        if raw is None:
            continue
        if np.all(np.isnan(raw)):
            continue
        center, lower, upper = _aggregate(raw, smooth, aggregator)
        iters_plot = iters[: len(center)]
        display_label = LABEL_MAP.get(label, label)
        color = COLOR_MAP.get(label)
        ax.plot(iters_plot, center, label=display_label, linewidth=1.6, color=color)
        if lower is not None and upper is not None:
            ax.fill_between(
                iters_plot,
                lower,
                upper,
                alpha=0.18,
                color=color,
                linewidth=0,
            )
        any_drawn = True
        if annotate_speedup and raw.ndim > 1:
            per_series_medians[label] = float(np.nanmedian(raw))

    if hline is not None:
        ax.axhline(
            hline,
            linestyle="--",
            color="gray",
            alpha=0.7,
            linewidth=1.0,
            label=hline_label,
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if annotate_speedup and "knn" in per_series_medians and "full" in per_series_medians:
        k = per_series_medians["knn"]
        f = per_series_medians["full"]
        if k > 0 and f > 0:
            ratio = f / k
            ax.text(
                0.97, 0.97,
                rf"$\mathrm{{median\ full/k\text{{-}}NN}} = {ratio:.2f}\times$",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
            )
    return any_drawn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Graph-SND DiCo training curves (single- or multi-seed)."
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
    parser.add_argument(
        "--figure-type",
        choices=("reward", "panels"),
        default="reward",
        help=(
            "'reward' (default, backward-compatible): single reward-vs-iter "
            "panel. 'panels': 3-panel publication figure with reward, applied "
            "SND (vs desired), and metric wall-clock per call."
        ),
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="VMAS Dispersion",
        help="Task name used in panel titles.",
    )
    parser.add_argument(
        "--desired-snd",
        type=float,
        default=0.1,
        help="Desired SND reference line on the applied-SND panel.",
    )
    args = parser.parse_args()

    columns = ("reward_mean", "applied_snd", "metric_time_ms")
    all_data: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
    for spec in args.series:
        label, paths_str = spec.split(":", 1)
        paths = [p.strip() for p in paths_str.split(",") if p.strip()]
        iters, cols = _parse_series(paths, columns)
        if iters.size == 0:
            print(f"WARNING: no data for {label!r}, skipping.")
            continue
        all_data[label] = (iters, cols)

    if not all_data:
        raise SystemExit("No data loaded. Check your CSV paths.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.figure_type == "reward":
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_panel(
            ax,
            all_data,
            column="reward_mean",
            ylabel="Mean reward (per step)",
            title=f"{args.task_name}: mean reward vs. iteration",
            smooth=args.smooth,
        )
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    _plot_panel(
        axes[0],
        all_data,
        column="reward_mean",
        ylabel="Mean reward (per step)",
        title=f"{args.task_name}: mean reward",
        smooth=args.smooth,
    )

    drew_applied = _plot_panel(
        axes[1],
        all_data,
        column="applied_snd",
        ylabel=r"Applied $\mathrm{SND}$",
        title=rf"Applied SND tracking ($\mathrm{{SND}}_{{\mathrm{{des}}}}{{=}}{args.desired_snd:g}$)",
        smooth=args.smooth,
        hline=args.desired_snd,
        hline_label=r"$\mathrm{SND}_{\mathrm{des}}$",
    )
    if not drew_applied:
        axes[1].text(
            0.5, 0.5,
            "applied_snd not present in logs",
            ha="center", va="center",
            transform=axes[1].transAxes, fontsize=9, color="gray",
        )
        axes[1].set_axis_off()

    drew_time = _plot_panel(
        axes[2],
        all_data,
        column="metric_time_ms",
        ylabel="Metric wall-clock per call (ms)",
        title="Per-call diversity cost (median, IQR band)",
        smooth=args.smooth,
        log_y=False,
        aggregator="median_iqr",
        annotate_speedup=True,
    )
    if not drew_time:
        axes[2].text(
            0.5, 0.5,
            "metric_time_ms not present in logs",
            ha="center", va="center",
            transform=axes[2].transAxes, fontsize=9, color="gray",
        )
        axes[2].set_axis_off()

    # Single shared legend above the panels to avoid duplication.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
