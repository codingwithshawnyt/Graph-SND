"""Render the Experiment 1 figures from the CSVs produced by
``exp1_metric_comparison.py``.

Emits four PDFs under the same directory as the CSVs:

- ``recovery.pdf``: a bar plot of full SND vs Graph-SND on K_n for
  n=4 at iter 0 and iter 100. The two bars for each iter should be
  visually indistinguishable.
- ``unbiasedness.pdf``: HT sample mean with 95 percent CI versus the
  true SND across p, annotated with ``bias/se``. The CIs must cover
  the SND line.
- ``concentration.pdf``: empirical absolute error (p95) versus sample
  size m, overlaid with the Hoeffding and Serfling confidence radii.
  The empirical line must lie under both theoretical curves.
- ``timing.pdf``: wall-clock speedup of the sampled Graph-SND path
  over full SND as a function of the Bernoulli inclusion probability
  p, shown on a log-y axis with one line per n.

Run::

    python experiments/exp1_plots.py --results results/exp1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path, **tight_layout_kw: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout_kw:
        fig.tight_layout(**tight_layout_kw)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_recovery(df: pd.DataFrame, out_path: Path) -> None:
    iters = list(df["iter_tag"])
    snd_vals = df["SND_full"].to_numpy()
    graph_vals = df["GraphSND_K4_via_D"].to_numpy()
    errs = df["abs_err_D_path"].abs().to_numpy()

    fig, (ax_bar, ax_err) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    x = np.arange(len(iters))
    width = 0.35
    b1 = ax_bar.bar(x - width / 2, snd_vals, width, label="Full SND", color="#2b7aff")
    b2 = ax_bar.bar(x + width / 2, graph_vals, width, label=r"Graph-SND on $K_n$", color="#ff7f0e")
    ax_bar.set_xticks(x, [t.replace("iter", "iter ") for t in iters])
    ax_bar.set_ylabel("diversity")
    ax_bar.set_title("Proposition 2: recovery at $n=4$")
    ax_bar.legend(loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax_bar.annotate(
                f"{h:.5f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax_err.bar(x, errs, width=0.5, color="#2ca02c")
    ax_err.set_xticks(x, [t.replace("iter", "iter ") for t in iters])
    ax_err.set_ylabel(r"$|\mathrm{SND} - \mathrm{GraphSND}(K_n)|$")
    ax_err.set_title("absolute recovery error")
    ax_err.set_ylim(0, max(1e-12, errs.max() * 2 if errs.max() > 0 else 1e-12))
    for xi, e in zip(x, errs):
        ax_err.annotate(
            f"{e:.1e}",
            xy=(xi, e),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    _save(fig, out_path)


def plot_unbiasedness(df: pd.DataFrame, out_path: Path) -> None:
    tags = df["iter_tag"].unique().tolist()
    fig, axes = plt.subplots(1, len(tags), figsize=(5.5 * len(tags), 4.0), sharey=False)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        sub = df[df["iter_tag"] == tag].sort_values("p")
        p = sub["p"].to_numpy()
        mean = sub["HT_mean"].to_numpy()
        ci = sub["HT_CI95_halfwidth"].to_numpy()
        snd_true = sub["SND_full"].iloc[0]
        bias_se = sub["bias_over_se"].to_numpy()

        ax.axhline(snd_true, color="#888888", linestyle="--", label=f"SND = {snd_true:.5f}")
        ax.errorbar(
            p, mean, yerr=ci, fmt="o-", color="#2b7aff",
            capsize=4, label=r"HT mean $\pm$ 95% CI",
        )
        for xi, m, b in zip(p, mean, bias_se):
            ax.annotate(
                f"bias/se = {b:+.2f}",
                xy=(xi, m),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("inclusion probability $p$")
        ax.set_ylabel(r"$\hat{\mathrm{SND}}_{\mathrm{HT}}$")
        ax.set_title(f"Prop. 7: HT unbiasedness (n=8, {tag})")
        ax.legend(loc="best")

    _save(fig, out_path)


def plot_concentration(df: pd.DataFrame, out_path: Path) -> None:
    tags = df["iter_tag"].unique().tolist()
    fig, axes = plt.subplots(1, len(tags), figsize=(5.5 * len(tags), 4.0))
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        sub = df[df["iter_tag"] == tag].sort_values("m")
        m = sub["m"].to_numpy()
        p95 = sub["p95_abs_err"].to_numpy()
        t_h = sub["hoeffding_t"].to_numpy()
        t_s = sub["serfling_t"].to_numpy()
        tail_h = sub["empirical_tail_hoeff"].to_numpy()
        tail_s = sub["empirical_tail_serfl"].to_numpy()
        delta = float(sub["delta"].iloc[0])
        snd_true = float(sub["SND_full"].iloc[0])

        ax.plot(m, t_h, "-", color="#d62728", label="Hoeffding $t_H$")
        ax.plot(m, t_s, "-", color="#9467bd", label="Serfling $t_S$")
        ax.plot(
            m, p95, "o-", color="#2ca02c",
            label=r"empirical 95th pct. $|\hat{\mathrm{SND}} - \mathrm{SND}|$",
        )
        ax.set_xlabel("sample size $m$")
        ax.set_ylabel("concentration radius")
        ax.set_yscale("log")
        ax.set_title(
            f"Thm. 9 (n=16, {tag}; SND = {snd_true:.4f}, $\\delta$={delta:.2f})"
        )
        ax.legend(loc="upper right", fontsize=9)

    # Extra horizontal padding between the two iter panels (Figure 3).
    _save(fig, out_path, w_pad=2.0)


def plot_timing(df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_speed, ax_abs) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    for n, grp in df.groupby("n"):
        sub = grp.sort_values("p")
        ax_speed.plot(
            sub["p"].to_numpy(), sub["speedup"].to_numpy(),
            "o-", label=f"n = {int(n)}"
        )
    ax_speed.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_speed.set_yscale("log")
    ax_speed.set_xlabel("Bernoulli inclusion probability $p$")
    ax_speed.set_ylabel(r"speedup = $T_{\mathrm{full}} / T_{\mathrm{graph}}$")
    ax_speed.set_title("Graph-SND wall-clock speedup")
    ax_speed.legend()

    for n, grp in df.groupby("n"):
        sub = grp.sort_values("p")
        p = sub["p"].to_numpy()
        ax_abs.errorbar(
            p, sub["sample_time_mean"].to_numpy() * 1e3,
            yerr=sub["sample_time_std"].to_numpy() * 1e3,
            fmt="o-", label=f"n = {int(n)} (sampled)", capsize=3,
        )
        full_mean = sub["full_time_mean"].iloc[0] * 1e3
        ax_abs.axhline(full_mean, linestyle="--", alpha=0.4,
                       label=f"n = {int(n)} (full, {full_mean:.1f} ms)")
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel("Bernoulli inclusion probability $p$")
    ax_abs.set_ylabel("wall-clock per estimate (ms)")
    ax_abs.set_title("absolute times")
    ax_abs.legend(fontsize=7, ncol=2, loc="lower right")

    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Experiment 1 figures from CSVs."
    )
    parser.add_argument(
        "--results", type=str, default="results/exp1",
        help="Directory containing recovery.csv, unbiasedness.csv, "
             "concentration.csv, timing.csv.",
    )
    args = parser.parse_args()
    results_dir = Path(args.results)

    df_recov = pd.read_csv(results_dir / "recovery.csv")
    df_unbias = pd.read_csv(results_dir / "unbiasedness.csv")
    df_conc = pd.read_csv(results_dir / "concentration.csv")
    df_timing = pd.read_csv(results_dir / "timing.csv")

    plot_recovery(df_recov, results_dir / "recovery.pdf")
    plot_unbiasedness(df_unbias, results_dir / "unbiasedness.pdf")
    plot_concentration(df_conc, results_dir / "concentration.pdf")
    plot_timing(df_timing, results_dir / "timing.pdf")

    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print("summary_numbers:")
        for k, v in summary.get("summary_numbers", {}).items():
            print(f"  {k:40s}: {v}")
    print(f"\nFigures written to {results_dir}/")
    for name in ("recovery.pdf", "unbiasedness.pdf", "concentration.pdf", "timing.pdf"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
