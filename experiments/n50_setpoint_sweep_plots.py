"""Plots and summary table for the n=50 Bernoulli-0.1 DiCo set-point sweep.

Expects the directory layout produced by
``ControllingBehavioralDiversity-fork/scripts/launch_n50_bern_setpoint_sweep.sh``::

    <root>/seed{0,1,2}/snd{0p12,0p14,0p15}/bern/graph_snd_log.csv

Produces:

* ``<out_dir>/n50_setpoint_sweep.pdf`` -- two-panel figure (applied-SND
  tracking, reward) shared across seeds and set points.
* ``<out_dir>/n50_setpoint_sweep_summary.csv`` -- per-cell statistics
  (late-window means, relative tracking error, median metric time).

Usage::

    python experiments/n50_setpoint_sweep_plots.py \
        --root ControllingBehavioralDiversity-fork/results/neurips_final_n50_setpoint_sweep \
        --out-dir results/dico_n50_sweep
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETPOINT_COLORS = {
    0.12: "#1f77b4",
    0.14: "#2ca02c",
    0.15: "#d62728",
}


def _parse_cell(path: str) -> tuple[int, float]:
    seed = int(path.split("seed")[1].split(os.sep)[0])
    tag = path.split("snd0p")[1].split(os.sep)[0]
    des = float("0." + tag)
    return seed, des


def load(root: str) -> pd.DataFrame:
    pattern = os.path.join(root, "seed*", "snd*", "bern", "graph_snd_log.csv")
    files = sorted(glob.glob(pattern))
    frames = []
    for f in files:
        seed, des = _parse_cell(f)
        df = pd.read_csv(f)
        df["_seed"] = seed
        df["_des"] = des
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No graph_snd_log.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame, late_window: int = 50) -> pd.DataFrame:
    rows = []
    for (des, seed), g in df.groupby(["_des", "_seed"]):
        g = g.sort_values("iter")
        late = g.tail(late_window)
        rows.append(
            dict(
                des=des,
                seed=seed,
                applied_late_mean=late["applied_snd"].mean(),
                applied_late_std=late["applied_snd"].std(),
                reward_late_mean=late["reward_mean"].mean(),
                reward_late_std=late["reward_mean"].std(),
                snd_t_late_mean=late["snd_t"].mean(),
                scaling_late_mean=late["scaling_ratio_mean"].mean(),
                metric_time_mean=g["metric_time_ms"].mean(),
                metric_time_median=g["metric_time_ms"].median(),
                tracking_err_abs=(late["applied_snd"] - des).abs().mean(),
                tracking_err_rel=(late["applied_snd"] - des).abs().mean() / des,
            )
        )
    per_cell = pd.DataFrame(rows)
    agg = (
        per_cell.groupby("des")
        .agg(
            applied_mean=("applied_late_mean", "mean"),
            applied_sem=(
                "applied_late_mean",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)),
            ),
            reward_mean=("reward_late_mean", "mean"),
            reward_sem=(
                "reward_late_mean",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)),
            ),
            scaling_mean=("scaling_late_mean", "mean"),
            snd_t_mean=("snd_t_late_mean", "mean"),
            metric_time_mean=("metric_time_mean", "mean"),
            metric_time_median=("metric_time_median", "mean"),
            tracking_err_rel_mean=("tracking_err_rel", "mean"),
        )
        .reset_index()
    )
    return per_cell, agg


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.set_title(r"(a) Applied SND tracks $\mathrm{SND}_{\mathrm{des}}$")
    for des, color in SETPOINT_COLORS.items():
        sub = df[df["_des"] == des]
        mean = sub.groupby("iter")["applied_snd"].mean()
        std = sub.groupby("iter")["applied_snd"].std()
        ax.plot(mean.index, mean.values, color=color, label=f"{des:.2f}", lw=1.3)
        ax.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.axhline(des, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(r"Applied SND  $\mathrm{SND}_t \cdot s_t$")
    ax.legend(title=r"$\mathrm{SND}_{\mathrm{des}}$", fontsize=8, loc="center right")
    ax.set_ylim(0.10, 0.18)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.set_title("(b) Task reward by set point")
    for des, color in SETPOINT_COLORS.items():
        sub = df[df["_des"] == des]
        mean = sub.groupby("iter")["reward_mean"].mean()
        std = sub.groupby("iter")["reward_mean"].std()
        ax.plot(mean.index, mean.values, color=color, label=f"{des:.2f}", lw=1.3)
        ax.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.legend(title=r"$\mathrm{SND}_{\mathrm{des}}$", fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

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
            "neurips_final_n50_setpoint_sweep"
        ),
    )
    ap.add_argument("--out-dir", default="results/dico_n50_sweep")
    ap.add_argument("--late-window", type=int, default=50)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(root)
    per_cell, agg = summarise(df, late_window=args.late_window)

    per_cell.to_csv(out_dir / "n50_setpoint_sweep_per_cell.csv", index=False)
    agg.to_csv(out_dir / "n50_setpoint_sweep_summary.csv", index=False)

    print("per-cell:")
    print(per_cell.to_string(index=False))
    print()
    print("aggregated across seeds:")
    print(agg.to_string(index=False))

    plot(df, out_dir / "n50_setpoint_sweep.pdf")
    print(f"wrote: {out_dir/'n50_setpoint_sweep.pdf'}")
    print(f"wrote: {out_dir/'n50_setpoint_sweep_summary.csv'}")
    print(f"wrote: {out_dir/'n50_setpoint_sweep_per_cell.csv'}")


if __name__ == "__main__":
    main()
