"""Summarize n=50 DiCo post-hoc full-SND validation logs.

Expected input layout from
``ControllingBehavioralDiversity-fork/scripts/launch_n50_posthoc_full_snd_validation.sh``::

    <root>/seed{0,1,2}/snd{0p12,0p14,0p15}/{bern,full}/graph_snd_log.csv

The key column is ``posthoc_full_snd``: complete-graph SND of the scaled
actions sent to the environment. It is intentionally different from
``applied_snd`` for sparse runs, which is the controller signal based on the
sparse estimator.

Outputs:

* ``n50_posthoc_full_snd_per_cell.csv``
* ``n50_posthoc_full_snd_summary.csv``
* ``n50_posthoc_full_snd_validation.pdf/.png``
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
    "bern": {"label": "Bernoulli-0.1 Graph-SND", "color": "#1f77b4", "ls": "-"},
    "full": {"label": "Full-SND controller", "color": "#d62728", "ls": "--"},
}

SETPOINT_COLORS = {
    0.12: "#4c72b0",
    0.14: "#55a868",
    0.15: "#c44e52",
}


def _parse_cell(path: str) -> Tuple[int, float, str]:
    parts = path.split(os.sep)
    seed_part = next(p for p in parts if p.startswith("seed"))
    snd_part = next(p for p in parts if p.startswith("snd"))
    est_part = parts[-2]
    seed = int(seed_part.replace("seed", ""))
    des = float(snd_part.replace("snd", "").replace("p", "."))
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
            continue
        df = pd.read_csv(f)
        if "posthoc_full_snd" not in df.columns:
            raise ValueError(
                f"{f} does not contain posthoc_full_snd; rerun with "
                "graph_snd_posthoc_full_snd_interval > 0."
            )
        df["_seed"] = seed
        df["_des"] = des
        df["_est"] = est
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No validation CSVs found under {root}")
    return pd.concat(frames, ignore_index=True)


def _sem(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize(df: pd.DataFrame, late_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (est, des, seed), g in df.groupby(["_est", "_des", "_seed"]):
        g = g.sort_values("iter")
        late = g.tail(late_window)
        late_posthoc = late[late["posthoc_full_snd"].notna()]
        if late_posthoc.empty:
            late_posthoc = g[g["posthoc_full_snd"].notna()].tail(late_window)

        posthoc_mean = (
            float(late_posthoc["posthoc_full_snd"].mean())
            if not late_posthoc.empty
            else float("nan")
        )
        applied_mean = float(late["applied_snd"].mean())
        rows.append(
            dict(
                estimator=est,
                snd_des=des,
                seed=seed,
                n_iters=int(len(g)),
                n_posthoc=int(g["posthoc_full_snd"].notna().sum()),
                applied_late_mean=applied_mean,
                applied_err_rel=float((late["applied_snd"] - des).abs().mean() / des),
                posthoc_full_snd_late_mean=posthoc_mean,
                posthoc_full_snd_err_rel=(
                    abs(posthoc_mean - des) / des if np.isfinite(posthoc_mean) else float("nan")
                ),
                posthoc_minus_applied=(
                    posthoc_mean - applied_mean
                    if np.isfinite(posthoc_mean) and np.isfinite(applied_mean)
                    else float("nan")
                ),
                reward_late_mean=float(late["reward_mean"].mean()),
                metric_time_ms_median=float(g["metric_time_ms"].median()),
                posthoc_full_snd_time_ms_median=float(
                    g["posthoc_full_snd_time_ms"].dropna().median()
                ),
                iter_time_ms_median=(
                    float(g["iter_time_ms"].dropna().median())
                    if "iter_time_ms" in g.columns and g["iter_time_ms"].notna().any()
                    else float("nan")
                ),
            )
        )
    per_cell = pd.DataFrame(rows)
    agg = (
        per_cell.groupby(["estimator", "snd_des"])
        .agg(
            n_seeds=("seed", "nunique"),
            n_posthoc_median=("n_posthoc", "median"),
            applied_mean=("applied_late_mean", "mean"),
            applied_sem=("applied_late_mean", _sem),
            applied_err_rel=("applied_err_rel", "mean"),
            posthoc_full_snd_mean=("posthoc_full_snd_late_mean", "mean"),
            posthoc_full_snd_sem=("posthoc_full_snd_late_mean", _sem),
            posthoc_full_snd_err_rel=("posthoc_full_snd_err_rel", "mean"),
            posthoc_minus_applied_mean=("posthoc_minus_applied", "mean"),
            posthoc_minus_applied_sem=("posthoc_minus_applied", _sem),
            reward_mean=("reward_late_mean", "mean"),
            reward_sem=("reward_late_mean", _sem),
            metric_time_ms=("metric_time_ms_median", "median"),
            posthoc_full_snd_time_ms=("posthoc_full_snd_time_ms_median", "median"),
            iter_time_ms=("iter_time_ms_median", "median"),
        )
        .reset_index()
    )
    return per_cell, agg


def plot(df: pd.DataFrame, per_cell: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.set_title("(a) Post-hoc full SND trajectories")
    for est, sty in ESTIMATOR_STYLE.items():
        for des, col in SETPOINT_COLORS.items():
            sub = df[(df["_est"] == est) & (df["_des"] == des)]
            sub = sub[sub["posthoc_full_snd"].notna()]
            if sub.empty:
                continue
            mean = sub.groupby("iter")["posthoc_full_snd"].mean()
            std = sub.groupby("iter")["posthoc_full_snd"].std()
            ax.plot(mean.index, mean.values, color=col, linestyle=sty["ls"], lw=1.3)
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
    ax.set_ylabel("post-hoc full SND")
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="gray", linestyle=sty["ls"], lw=1.3, label=sty["label"])
        for sty in ESTIMATOR_STYLE.values()
    ] + [
        Line2D([], [], color=col, lw=2.5, label=f"SND_des={des:.2f}")
        for des, col in SETPOINT_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")

    ax = axes[0, 1]
    ax.set_title("(b) Applied signal vs post-hoc full SND")
    for est, sty in ESTIMATOR_STYLE.items():
        sub = per_cell[per_cell["estimator"] == est]
        ax.scatter(
            sub["applied_late_mean"],
            sub["posthoc_full_snd_late_mean"],
            color=sty["color"],
            label=sty["label"],
            alpha=0.85,
        )
    lo = min(per_cell["applied_late_mean"].min(), per_cell["posthoc_full_snd_late_mean"].min())
    hi = max(per_cell["applied_late_mean"].max(), per_cell["posthoc_full_snd_late_mean"].max())
    ax.plot([lo, hi], [lo, hi], color="#555", linestyle=":", lw=1.0)
    ax.set_xlabel("late applied controller signal")
    ax.set_ylabel("late post-hoc full SND")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.set_title("(c) Post-hoc full-SND target error")
    setpoints = sorted(per_cell["snd_des"].unique())
    x = np.arange(len(setpoints))
    width = 0.35
    for i, (est, sty) in enumerate(ESTIMATOR_STYLE.items()):
        vals = []
        sems = []
        sub = per_cell[per_cell["estimator"] == est]
        for des in setpoints:
            cell = sub[sub["snd_des"] == des]["posthoc_full_snd_err_rel"]
            vals.append(cell.mean() if len(cell) else np.nan)
            sems.append(_sem(cell) if len(cell) else np.nan)
        ax.bar(
            x + (i - 0.5) * width,
            np.asarray(vals) * 100.0,
            width,
            yerr=np.asarray(sems) * 100.0,
            color=sty["color"],
            alpha=0.82,
            capsize=3,
            label=sty["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.2f}" for d in setpoints])
    ax.set_xlabel("SND_des")
    ax.set_ylabel("relative error (%)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.set_title("(d) Diagnostic cost")
    for i, (est, sty) in enumerate(ESTIMATOR_STYLE.items()):
        sub = per_cell[per_cell["estimator"] == est]
        vals = (
            sub.groupby("snd_des")["posthoc_full_snd_time_ms_median"]
            .median()
            .reindex(setpoints)
            .values
        )
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=sty["color"],
            alpha=0.82,
            label=sty["label"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.2f}" for d in setpoints])
    ax.set_xlabel("SND_des")
    ax.set_ylabel("post-hoc full-SND time (ms)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=(
            "ControllingBehavioralDiversity-fork/results/"
            "neurips_final_n50_posthoc_full_snd"
        ),
    )
    ap.add_argument("--out-dir", default="results/dico_n50_posthoc_full_snd")
    ap.add_argument("--late-window", type=int, default=50)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(root)
    per_cell, agg = summarize(df, late_window=args.late_window)
    per_cell.to_csv(out_dir / "n50_posthoc_full_snd_per_cell.csv", index=False)
    agg.to_csv(out_dir / "n50_posthoc_full_snd_summary.csv", index=False)
    plot(df, per_cell, out_dir / "n50_posthoc_full_snd_validation.pdf")

    print("per-cell:")
    print(per_cell.sort_values(["snd_des", "estimator", "seed"]).to_string(index=False))
    print("\naggregated across seeds:")
    print(agg.to_string(index=False))
    print(f"\nwrote: {out_dir/'n50_posthoc_full_snd_per_cell.csv'}")
    print(f"wrote: {out_dir/'n50_posthoc_full_snd_summary.csv'}")
    print(f"wrote: {out_dir/'n50_posthoc_full_snd_validation.pdf'}")


if __name__ == "__main__":
    main()
