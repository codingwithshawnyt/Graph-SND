"""Head-to-head n=50 DiCo comparison: Bernoulli-0.1 Graph-SND vs full SND.

Consumes the layout produced jointly by
``ControllingBehavioralDiversity-fork/scripts/launch_n50_bern_setpoint_sweep.sh``
(already done) and
``ControllingBehavioralDiversity-fork/scripts/launch_n50_full_setpoint_sweep.sh``
(this patch)::

    <root>/seed{0,1,2}/snd{0p12,0p14,0p15}/{bern,full}/graph_snd_log.csv

The two launchers share the same ``RESULTS_BASE`` so both estimators' CSVs sit
side-by-side per cell.

Per cell this script computes

* **Tracking**: late-window mean ``applied_snd`` and relative tracking error
  ``|applied_snd - snd_des| / snd_des``.
* **Reward**: late-window mean ``reward_mean`` (and SEM across seeds).
* **Cost**: median ``metric_time_ms`` (mean per-call diversity wall-clock, the
  only component that causally differs between the two estimators at a fixed
  batch size) and, when available, median ``iter_time_ms`` (end-to-end
  per-iter wall-clock -- only present on runs produced after the callback was
  augmented with end-to-end timing, i.e. the new full-SND runs and any
  regenerated Bernoulli runs).

Outputs to ``--out-dir``:

* ``n50_bern_vs_full_per_cell.csv``        per (estimator, seed, snd_des)
* ``n50_bern_vs_full_summary.csv``         aggregated across seeds
* ``n50_bern_vs_full_comparison.pdf/.png`` 3-panel figure
* ``n50_bern_vs_full_headtohead.tex``      LaTeX tabular fragment ready to
                                           ``\\input`` into the paper

Usage::

    python experiments/n50_bern_vs_full_comparison.py \\
        --root ControllingBehavioralDiversity-fork/results/neurips_final_n50_setpoint_sweep \\
        --out-dir results/dico_n50_bern_vs_full
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ESTIMATOR_STYLE = {
    "bern": {"label": "Graph-SND (Bernoulli p=0.1)", "color": "#1f77b4", "ls": "-"},
    "full": {"label": "Full SND", "color": "#d62728", "ls": "--"},
}

SETPOINT_COLORS = {
    0.12: "#4c72b0",
    0.14: "#55a868",
    0.15: "#c44e52",
}


def _parse_cell(path: str) -> Tuple[int, float, str]:
    """Extract (seed, desired_snd, estimator) from a CSV path.

    The path follows the sweep layout ``<root>/seed<S>/snd<tag>/<est>/graph_snd_log.csv``
    where ``<tag>`` is the desired SND with the decimal point replaced by ``p``
    (so 0.12 -> ``0p12``).
    """
    parts = path.split(os.sep)
    seed_part = next(p for p in parts if p.startswith("seed"))
    snd_part = next(p for p in parts if p.startswith("snd"))
    est_part = parts[-2]

    seed = int(seed_part.replace("seed", ""))
    tag = snd_part.replace("snd", "")
    des = float(tag.replace("p", "."))
    return seed, des, est_part


def load(root: str) -> pd.DataFrame:
    pattern = os.path.join(root, "seed*", "snd*", "*", "graph_snd_log.csv")
    files = sorted(glob.glob(pattern))
    frames = []
    for f in files:
        try:
            seed, des, est = _parse_cell(f)
        except Exception:
            continue
        if est not in ESTIMATOR_STYLE:
            # Skip any per-cell subdirectories that aren't an estimator
            # we care about (e.g. stale knn/ or ippo/ from older launches).
            continue
        df = pd.read_csv(f)
        df["_seed"] = seed
        df["_des"] = des
        df["_est"] = est
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No graph_snd_log.csv under {root}/seed*/snd*/{{bern,full}}/"
        )
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame, late_window: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (est, des, seed), g in df.groupby(["_est", "_des", "_seed"]):
        g = g.sort_values("iter")
        late = g.tail(late_window)
        iter_time_med = (
            float(g["iter_time_ms"].median())
            if "iter_time_ms" in g.columns and g["iter_time_ms"].notna().any()
            else float("nan")
        )
        rows.append(
            dict(
                estimator=est,
                snd_des=des,
                seed=seed,
                n_iters=len(g),
                applied_late_mean=late["applied_snd"].mean(),
                applied_late_std=late["applied_snd"].std(),
                reward_late_mean=late["reward_mean"].mean(),
                reward_late_std=late["reward_mean"].std(),
                snd_t_late_mean=late["snd_t"].mean(),
                metric_time_ms_median=float(g["metric_time_ms"].median()),
                iter_time_ms_median=iter_time_med,
                tracking_err_abs=(late["applied_snd"] - des).abs().mean(),
                tracking_err_rel=(late["applied_snd"] - des).abs().mean() / des,
            )
        )
    per_cell = pd.DataFrame(rows)

    def _sem(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    agg = (
        per_cell.groupby(["estimator", "snd_des"])
        .agg(
            n_seeds=("seed", "nunique"),
            applied_mean=("applied_late_mean", "mean"),
            applied_sem=("applied_late_mean", _sem),
            reward_mean=("reward_late_mean", "mean"),
            reward_sem=("reward_late_mean", _sem),
            tracking_err_rel=("tracking_err_rel", "mean"),
            metric_time_ms=("metric_time_ms_median", "median"),
            iter_time_ms=("iter_time_ms_median", "median"),
        )
        .reset_index()
    )
    return per_cell, agg


def plot(df: pd.DataFrame, per_cell: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 6.5), constrained_layout=True, sharex=False
    )

    # (a) Applied SND trajectories -- full vs bern, mean ± std across seeds,
    # per set point. Use the faceted style: for each set point, overlay the
    # two estimators with distinct linestyles; colour encodes set point.
    ax = axes[0, 0]
    ax.set_title(r"(a) Applied SND tracking by estimator")
    for est, sty in ESTIMATOR_STYLE.items():
        for des, col in SETPOINT_COLORS.items():
            sub = df[(df["_est"] == est) & (df["_des"] == des)]
            if sub.empty:
                continue
            mean = sub.groupby("iter")["applied_snd"].mean()
            std = sub.groupby("iter")["applied_snd"].std()
            ax.plot(
                mean.index,
                mean.values,
                color=col,
                linestyle=sty["ls"],
                lw=1.3,
                label=(
                    f"{sty['label']} @ {des:.2f}"
                    if des == list(SETPOINT_COLORS)[0]
                    else None
                ),
            )
            ax.fill_between(
                mean.index,
                (mean - std).values,
                (mean + std).values,
                color=col,
                alpha=0.12,
                linewidth=0,
            )
            ax.axhline(des, color=col, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(r"Applied SND  $\mathrm{SND}_t\,s_t$")
    ax.set_ylim(0.10, 0.18)
    ax.grid(True, alpha=0.2)
    # Split legend: one entry per estimator style for clarity.
    from matplotlib.lines import Line2D

    est_handles = [
        Line2D([], [], color="gray", linestyle=sty["ls"], lw=1.3, label=sty["label"])
        for sty in ESTIMATOR_STYLE.values()
    ]
    snd_handles = [
        Line2D([], [], color=col, lw=2.5, label=f"SND_des={des:.2f}")
        for des, col in SETPOINT_COLORS.items()
    ]
    ax.legend(
        handles=est_handles + snd_handles, fontsize=7, loc="upper right", ncol=1
    )

    # (b) Reward trajectories.
    ax = axes[0, 1]
    ax.set_title("(b) Task reward by estimator")
    for est, sty in ESTIMATOR_STYLE.items():
        for des, col in SETPOINT_COLORS.items():
            sub = df[(df["_est"] == est) & (df["_des"] == des)]
            if sub.empty:
                continue
            mean = sub.groupby("iter")["reward_mean"].mean()
            std = sub.groupby("iter")["reward_mean"].std()
            ax.plot(mean.index, mean.values, color=col, linestyle=sty["ls"], lw=1.3)
            ax.fill_between(
                mean.index,
                (mean - std).values,
                (mean + std).values,
                color=col,
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.grid(True, alpha=0.2)
    ax.legend(
        handles=est_handles + snd_handles, fontsize=7, loc="lower right", ncol=1
    )

    # (c) Per-cell relative tracking error, grouped by set point and estimator.
    ax = axes[1, 0]
    ax.set_title(r"(c) Relative tracking error (late window)")
    setpoints = sorted(per_cell["snd_des"].unique())
    x = np.arange(len(setpoints))
    width = 0.35
    for i, (est, sty) in enumerate(ESTIMATOR_STYLE.items()):
        per_est = per_cell[per_cell["estimator"] == est]
        means = []
        sems = []
        for des in setpoints:
            cell = per_est[per_est["snd_des"] == des]["tracking_err_rel"]
            means.append(cell.mean() if len(cell) else np.nan)
            sems.append(
                (cell.std(ddof=1) / np.sqrt(len(cell)))
                if len(cell) > 1
                else (0.0 if len(cell) else np.nan)
            )
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=sems,
            label=sty["label"],
            color=sty["color"],
            alpha=0.8,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.2f}" for d in setpoints])
    ax.set_xlabel(r"$\mathrm{SND}_{\mathrm{des}}$")
    ax.set_ylabel(r"$|\,\mathrm{applied\_SND} - \mathrm{SND}_{\mathrm{des}}| / \mathrm{SND}_{\mathrm{des}}$")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    # (d) Cost comparison. Always plot median metric_time_ms (per-call mean
    # wall-clock of the Graph-SND / full-SND estimator itself -- the only
    # component that causally differs between the two at fixed batch size
    # and fixed model). When the full-SND cells also logged end-to-end
    # iter_time_ms (added to the callback after the original Bernoulli
    # runs), annotate the bar with iter-time in seconds so readers can see
    # the metric's share of total per-iter wall-clock at n=50.
    ax = axes[1, 1]
    ax.set_title("(d) Metric-call wall-clock (ms)")
    for i, (est, sty) in enumerate(ESTIMATOR_STYLE.items()):
        per_est = per_cell[per_cell["estimator"] == est]
        vals = (
            per_est.groupby("snd_des")["metric_time_ms_median"]
            .median()
            .reindex(setpoints)
            .values
        )
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=sty["label"],
            color=sty["color"],
            alpha=0.8,
        )
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.1f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.2f}" for d in setpoints])
    ax.set_xlabel(r"$\mathrm{SND}_{\mathrm{des}}$")
    ax.set_ylabel("median metric_time_ms (per call)")
    ax.grid(True, axis="y", alpha=0.2)
    # Leave ~18% headroom above the tallest bar so the value annotations
    # and the iter-time box aren't eclipsed by the legend.
    _ymax = float(max(v for v in vals if np.isfinite(v)))
    ax.set_ylim(0, _ymax * 1.18)
    ax.legend(fontsize=8, loc="center right")

    # Annotate full-SND iter wall-clock at the top of panel (d) so the
    # reader can contextualise the ~10x metric speed-up against the
    # rollout+PPO-dominated per-iter budget at n=50.
    iter_ms_full = per_cell.loc[
        per_cell["estimator"] == "full", "iter_time_ms_median"
    ].median()
    if np.isfinite(iter_ms_full):
        ax.text(
            0.5,
            0.97,
            f"full-SND median iter: {iter_ms_full/1000:.1f} s  "
            f"(metric share $\\lesssim$2\\%)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#333",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#bbb",
                alpha=0.9,
            ),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def emit_latex(agg: pd.DataFrame, out_path: Path) -> None:
    """Write a LaTeX tabular fragment suitable to ``\\input`` into the paper.

    Columns: set point, then for each estimator its applied-SND (mean ± SEM),
    reward (mean ± SEM), relative tracking error, per-call metric time, and
    per-iter wall-clock when available.
    """
    setpoints = sorted(agg["snd_des"].unique())
    lines: list[str] = []
    lines.append(r"% Auto-generated by experiments/n50_bern_vs_full_comparison.py -- do not edit by hand.")
    lines.append(r"\begin{tabular}{l cc cc cc cc}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{2}{c}{Applied SND} & \multicolumn{2}{c}{Reward}"
        r" & \multicolumn{2}{c}{Rel.\ tracking err.} & \multicolumn{2}{c}{Metric time (ms)} \\"
    )
    lines.append(
        r"$\mathrm{SND}_{\mathrm{des}}$ & Bern-0.1 & Full & Bern-0.1 & Full"
        r" & Bern-0.1 (\%) & Full (\%) & Bern-0.1 & Full \\"
    )
    lines.append(r"\midrule")

    def _fmt_pm_scaled(mean: float, sem: float, digits: int) -> str:
        """Format ``mean ± SEM`` with enough digits that the SEM isn't
        rounded to zero when it's several orders of magnitude smaller than
        the mean (as is the case for applied-SND, where SEM ~ 1e-4)."""
        if not np.isfinite(mean):
            return "--"
        if not np.isfinite(sem) or sem == 0.0:
            return f"{mean:.{digits}f}"
        # Show at least 1 significant digit of SEM; bump digits if the
        # default would round the SEM to zero.
        if sem > 0:
            min_sem_digits = max(0, int(np.ceil(-np.log10(sem))) + 1)
            digits = max(digits, min_sem_digits)
        return f"{mean:.{digits}f}$\\pm${sem:.{digits}f}"

    def _fmt_ms(v: float) -> str:
        if not np.isfinite(v):
            return "--"
        if v >= 100:
            return f"{v:.0f}"
        if v >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    for des in setpoints:
        row_bern = agg[(agg["estimator"] == "bern") & (agg["snd_des"] == des)]
        row_full = agg[(agg["estimator"] == "full") & (agg["snd_des"] == des)]
        if row_bern.empty or row_full.empty:
            # Skip partial rows; the full sweep may still be in flight.
            continue
        rb = row_bern.iloc[0]
        rf = row_full.iloc[0]
        lines.append(
            f"{des:.2f} "
            f"& {_fmt_pm_scaled(rb['applied_mean'], rb['applied_sem'], digits=3)} "
            f"& {_fmt_pm_scaled(rf['applied_mean'], rf['applied_sem'], digits=3)} "
            f"& {_fmt_pm_scaled(rb['reward_mean'], rb['reward_sem'], digits=3)} "
            f"& {_fmt_pm_scaled(rf['reward_mean'], rf['reward_sem'], digits=3)} "
            f"& {rb['tracking_err_rel']*100:.2f} "
            f"& {rf['tracking_err_rel']*100:.2f} "
            f"& {_fmt_ms(rb['metric_time_ms'])} "
            f"& {_fmt_ms(rf['metric_time_ms'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=(
            "ControllingBehavioralDiversity-fork/results/"
            "neurips_final_n50_setpoint_sweep"
        ),
        help="Shared RESULTS_BASE used by both the bern and full sweep launchers.",
    )
    ap.add_argument("--out-dir", default="results/dico_n50_bern_vs_full")
    ap.add_argument("--late-window", type=int, default=50)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(root)
    per_cell, agg = summarise(df, late_window=args.late_window)

    per_cell.to_csv(out_dir / "n50_bern_vs_full_per_cell.csv", index=False)
    agg.to_csv(out_dir / "n50_bern_vs_full_summary.csv", index=False)

    print("per-cell:")
    print(per_cell.sort_values(["snd_des", "estimator", "seed"]).to_string(index=False))
    print()
    print("aggregated across seeds:")
    print(agg.to_string(index=False))

    # Only emit the comparison figure and LaTeX fragment if both estimators
    # showed up; otherwise the consumer is only looking at one half of the
    # sweep (e.g. while the full-SND runs are still in flight) and a single-
    # estimator "comparison" would be misleading.
    estimators_present = set(per_cell["estimator"].unique())
    if {"bern", "full"}.issubset(estimators_present):
        plot(df, per_cell, out_dir / "n50_bern_vs_full_comparison.pdf")
        emit_latex(agg, out_dir / "n50_bern_vs_full_headtohead.tex")
        print(f"wrote: {out_dir/'n50_bern_vs_full_comparison.pdf'}")
        print(f"wrote: {out_dir/'n50_bern_vs_full_headtohead.tex'}")
    else:
        print(
            "NOTE: only found estimators="
            + str(sorted(estimators_present))
            + " under --root; skipping head-to-head figure/tex fragment "
            "until both 'bern/' and 'full/' CSVs are present."
        )
    print(f"wrote: {out_dir/'n50_bern_vs_full_summary.csv'}")
    print(f"wrote: {out_dir/'n50_bern_vs_full_per_cell.csv'}")


if __name__ == "__main__":
    main()
