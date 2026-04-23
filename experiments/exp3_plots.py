"""Plots for Experiment 3: expander sparsification ablation.

Expects a CSV produced by ``exp3_expander_distortion.py``.

Usage::

    python experiments/exp3_plots.py --csv results/exp3/expander_distortion.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_STYLES = {
    "random_regular": {"color": "#1f77b4", "marker": "o", "label": "d-regular expander"},
    "bernoulli": {"color": "#ff7f0e", "marker": "s", "label": "Bernoulli (matched |E|)"},
    "uniform": {"color": "#2ca02c", "marker": "^", "label": "Uniform (matched |E|)"},
    "knn": {"color": "#d62728", "marker": "D", "label": "k-NN (matched |E|)"},
}


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["n", "d", "graph_family", "data_source"]).agg(
        ratio_mean=("ratio", "mean"),
        ratio_std=("ratio", "std"),
        ratio_recip_mean=("ratio_reciprocal", "mean"),
        ratio_recip_std=("ratio_reciprocal", "std"),
        abs_distortion_mean=("abs_distortion", "mean"),
        num_edges_mean=("num_edges", "mean"),
        edge_fraction=("edge_fraction", "first"),
        lambda_2_mean=("lambda_2", "mean"),
        spectral_gap_mean=("spectral_gap", "mean"),
        pi_G_mean=("pi_G", "mean"),
        upper_bound_mean=("upper_bound", "mean"),
        time_full_ms=("time_full_ms", "first"),
        time_graph_ms=("time_graph_ms", "mean"),
    ).reset_index()
    return grp


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    for gf, style in FAMILY_STYLES.items():
        sub = df[(df["graph_family"] == gf) & (df["data_source"] == "synthetic")]
        if sub.empty:
            continue
        ax.errorbar(
            sub["edge_fraction"],
            sub["ratio_recip_mean"],
            yerr=sub["ratio_recip_std"].fillna(0),
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markersize=5,
            capsize=2,
            linestyle="none",
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    lim_lo = df["edge_fraction"].min() * 0.5
    lim_hi = df["edge_fraction"].max() * 1.2
    ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        color="gray",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label="lower bound |E|/|P|",
    )
    ax.set_xlabel("Edge fraction  |E| / |P|")
    ax.set_ylabel(r"$\mathrm{SND}_G^{\mathrm{u}} \; / \; \mathrm{SND}$")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(a) Distortion ratio vs edge density")


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[df["graph_family"] == "random_regular"]
    if sub.empty:
        return
    for n_val, grp in sub.groupby("n"):
        ax.scatter(
            grp["d"],
            grp["pi_G_mean"],
            marker="o",
            s=30,
            label=f"n={int(n_val)}",
        )
    d_range = np.linspace(df["d"].min(), df["d"].max(), 100)
    for n_val in sub["n"].unique():
        predicted = float(n_val) * np.log2(float(n_val)) / d_range
        ax.plot(
            d_range,
            predicted,
            linestyle="--",
            alpha=0.4,
            linewidth=1,
        )
    ax.set_xlabel("Degree d")
    ax.set_ylabel(r"$\pi(G)$ (measured)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title(r"(b) Forwarding index $\pi(G)$ vs degree")


def plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    for gf, style in FAMILY_STYLES.items():
        sub = df[(df["graph_family"] == gf) & (df["data_source"] == "synthetic")]
        if sub.empty:
            continue
        ax.scatter(
            sub["spectral_gap_mean"],
            sub["ratio_recip_mean"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            s=30,
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Spectral gap $1 - \lambda_2 / d$")
    ax.set_ylabel(r"$\mathrm{SND}_G^{\mathrm{u}} \; / \; \mathrm{SND}$")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(c) Distortion vs spectral gap")


def plot_panel_d(ax: plt.Axes, df: pd.DataFrame) -> None:
    for gf, style in FAMILY_STYLES.items():
        sub = df[(df["graph_family"] == gf) & (df["data_source"] == "synthetic")]
        if sub.empty:
            continue
        ax.scatter(
            sub["time_graph_ms"],
            sub["abs_distortion_mean"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            s=30,
        )
    ax.set_xlabel("Graph-SND time (ms)")
    ax.set_ylabel(r"$|\mathrm{SND} - \mathrm{SND}_G^{\mathrm{u}}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(d) Cost-accuracy Pareto")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df_raw = pd.read_csv(args.csv)
    df = _aggregate(df_raw)

    out_dir = Path(args.csv).parent if args.out is None else Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.csv).stem

    fig_main, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))
    plot_panel_a(ax_a, df)
    plot_panel_b(ax_b, df)
    fig_main.tight_layout()
    main_path = out_dir / f"{stem}_main.pdf"
    fig_main.savefig(main_path, dpi=150, bbox_inches="tight")
    print(f"Main figure -> {main_path}")

    fig_app, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(10, 4))
    plot_panel_c(ax_c, df)
    plot_panel_d(ax_d, df)
    fig_app.tight_layout()
    app_path = out_dir / f"{stem}_appendix.pdf"
    fig_app.savefig(app_path, dpi=150, bbox_inches="tight")
    print(f"Appendix figure -> {app_path}")

    # Optional: when trained-policy checkpoint rows are present (n=100),
    # emit a small side-by-side panel to compare synthetic vs checkpoint.
    ckpt = df_raw[df_raw["data_source"] == "checkpoint"].copy()
    if not ckpt.empty:
        synth = df_raw[
            (df_raw["data_source"] == "synthetic")
            & (df_raw["n"] == 100)
            & (df_raw["graph_seed"] == df_raw["graph_seed"].min())
        ].copy()
        fig_ck, axes = plt.subplots(1, 2, figsize=(10, 4))
        for gf, style in FAMILY_STYLES.items():
            s1 = synth[synth["graph_family"] == gf]
            s2 = ckpt[ckpt["graph_family"] == gf]
            if not s1.empty:
                axes[0].plot(
                    s1["edge_fraction"],
                    s1["ratio_reciprocal"],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle="-",
                    alpha=0.6,
                    label=f"{style['label']} (synthetic)",
                )
            if not s2.empty:
                axes[0].plot(
                    s2["edge_fraction"],
                    s2["ratio_reciprocal"],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle="--",
                    label=f"{style['label']} (checkpoint)",
                )
                axes[1].scatter(
                    s2["time_graph_ms"],
                    s2["abs_distortion"],
                    marker=style["marker"],
                    color=style["color"],
                    s=40,
                    label=style["label"],
                )
        axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Edge fraction  |E| / |P|")
        axes[0].set_ylabel(r"$\mathrm{SND}_G^{\mathrm{u}} \; / \; \mathrm{SND}$")
        axes[0].set_title("(a) n=100 synthetic vs checkpoint")
        axes[0].legend(fontsize=6, loc="lower right")

        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Graph-SND time (ms)")
        axes[1].set_ylabel(r"$|\mathrm{SND} - \mathrm{SND}_G^{\mathrm{u}}|$")
        axes[1].set_title("(b) n=100 checkpoint cost-accuracy")
        axes[1].legend(fontsize=7, loc="upper right")
        fig_ck.tight_layout()
        ckpt_path = out_dir / f"{stem}_n100_checkpoint.pdf"
        fig_ck.savefig(ckpt_path, dpi=150, bbox_inches="tight")
        print(f"Checkpoint panel -> {ckpt_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
