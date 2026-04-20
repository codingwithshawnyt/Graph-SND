"""Plot the n=500 timing sweep alongside prior-scale speedup anchors.

Produces ``figures/timing_n500.pdf`` with two panels:

- Left: wall-clock speedup vs Bernoulli inclusion probability ``p`` at
  ``n=500`` (from ``results/exp2/timing_*.csv``), with the structural
  prediction ``1/p`` overlaid.
- Right: per-call speedup at ``p=0.1`` as a function of ``n``,
  combining the CPU Experiment-1 sweep (``results/exp1/timing.csv``,
  ``n in {4, 8, 16}``), the online n=100 scaling log
  (``results/scaling/n100_overnight_snd_log.csv``), and the new frozen-
  init n=500 point. The dashed gray line marks the
  ``binom(n, 2) / E[|E(G_0.1)|] = 10x`` prediction.

Sources are kept as separate CSV lookups because each was collected
under a different timing regime (CPU frozen-checkpoint at small n, GPU
online at n=100, GPU frozen-init at n=500); the right panel
deliberately mixes those regimes because Proposition 6's structural
prediction should be regime-invariant.

CLI::

    python experiments/plot_timing_n500.py \
        --exp2-csv results/exp2/timing_n500.csv \
        --exp1-csv results/exp1/timing.csv \
        --n100-csv results/scaling/n100_overnight_snd_log.csv \
        --out Paper/figures/timing_n500.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _filter_n500(df_exp2: pd.DataFrame, target_n: int) -> pd.DataFrame:
    sub = df_exp2[df_exp2["n"] == target_n].copy()
    sub = sub[~sub["speedup"].isna()]
    sub = sub.sort_values("p")
    return sub


def _scaling_at_p01_from_exp1(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[np.isclose(df["p"], 0.1)][["n", "speedup"]].copy()
    sub["regime"] = "CPU, frozen-ckpt (App. C)"
    return sub


def _scaling_at_p01_from_n100(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    sub = df[np.isclose(df["p"], 0.1)][["n_agents", "speedup"]].copy()
    sub = sub.rename(columns={"n_agents": "n"})
    sub = (
        sub.groupby("n", as_index=False)["speedup"]
        .median()
        .sort_values("n")
    )
    sub["regime"] = "GPU, online training (Sec. 6.4)"
    return sub


def _scaling_at_p01_from_exp2(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    sub = df[np.isclose(df["p"], 0.1) & ~df["speedup"].isna()][["n", "speedup"]].copy()
    if sub.empty:
        return None
    sub["regime"] = "GPU, frozen-init (this work)"
    return sub


def plot(
    df_exp2: pd.DataFrame,
    df_exp1: Optional[pd.DataFrame],
    df_n100: Optional[pd.DataFrame],
    out_path: Path,
    target_n: int = 500,
) -> None:
    fig, (ax_p, ax_n) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    sub = _filter_n500(df_exp2, target_n)
    if sub.empty:
        raise RuntimeError(
            f"no non-NaN rows at n={target_n} in exp2 CSV; did the run OOM?"
        )
    p_grid = sub["p"].to_numpy()
    speedup = sub["speedup"].to_numpy()
    speedup_med = sub["speedup_median"].to_numpy() if "speedup_median" in sub.columns else speedup

    ax_p.plot(
        p_grid, speedup, "o-",
        color="#1f77b4", linewidth=2.2, markersize=7,
        label=f"measured (mean of trials)",
    )
    ax_p.plot(
        p_grid, speedup_med, "s--",
        color="#2ca02c", linewidth=1.2, markersize=5, alpha=0.8,
        label="measured (median)",
    )
    p_dense = np.linspace(max(p_grid.min(), 0.05), 1.0, 100)
    ax_p.plot(
        p_dense, 1.0 / p_dense,
        color="gray", linestyle=":", linewidth=1.5,
        label=r"structural $1/p$",
    )
    ax_p.axhline(1.0, color="lightgray", linestyle="-", linewidth=0.8)
    ax_p.set_xlabel(r"Bernoulli inclusion probability $p$")
    ax_p.set_ylabel(r"wall-clock speedup $T_{\mathrm{full}} / T_{\mathrm{sample}}$")
    n_dtype_label = (
        f"GPU {sub['device'].iloc[0]}, "
        f"{sub['dtype'].iloc[0]}, "
        f"{int(sub['timing_trials'].iloc[0])} trials"
    )
    ax_p.set_title(
        rf"Frozen-init speedup at $n={target_n}$ "
        rf"({n_dtype_label})"
    )
    ax_p.set_yscale("log")
    ax_p.legend(loc="upper right")
    ax_p.grid(alpha=0.3, which="both")

    pieces = []
    if df_exp1 is not None and not df_exp1.empty:
        pieces.append(_scaling_at_p01_from_exp1(df_exp1))
    if df_n100 is not None:
        piece_n100 = _scaling_at_p01_from_n100(df_n100)
        if piece_n100 is not None:
            pieces.append(piece_n100)
    piece_exp2 = _scaling_at_p01_from_exp2(df_exp2)
    if piece_exp2 is not None:
        pieces.append(piece_exp2)

    if not pieces:
        raise RuntimeError(
            "no data at p=0.1 across exp1/n100/exp2 CSVs; cannot draw scaling panel"
        )
    regime_styles = {
        "CPU, frozen-ckpt (App. C)": {"marker": "o", "color": "#d62728"},
        "GPU, online training (Sec. 6.4)": {"marker": "s", "color": "#ff7f0e"},
        "GPU, frozen-init (this work)": {"marker": "D", "color": "#1f77b4"},
    }
    for piece in pieces:
        regime = piece["regime"].iloc[0]
        style = regime_styles.get(regime, {"marker": "x", "color": "black"})
        ax_n.scatter(
            piece["n"], piece["speedup"],
            s=70, marker=style["marker"], color=style["color"],
            edgecolors="black", linewidths=0.6, label=regime, zorder=3,
        )

    n_ref = np.unique(
        np.concatenate([piece["n"].to_numpy() for piece in pieces])
    )
    ax_n.axhline(
        10.0, color="gray", linestyle=":", linewidth=1.5,
        label=r"structural $1/p = 10\times$",
    )
    ax_n.set_xscale("log")
    ax_n.set_xlabel(r"team size $n$")
    ax_n.set_ylabel(r"wall-clock speedup at $p{=}0.1$")
    ax_n.set_title(
        r"Speedup at $p{=}0.1$ across scale: $n{\in}\{4, 8, 16, 100, 250, 500\}$"
    )
    ax_n.grid(alpha=0.3, which="both")

    ymin = 0.0
    ymax = max(
        15.0,
        float(np.nanmax([piece["speedup"].max() for piece in pieces])) * 1.15,
    )
    ax_n.set_ylim(ymin, ymax)
    ax_n.set_xticks(n_ref)
    ax_n.set_xticklabels([str(int(n)) for n in n_ref])
    ax_n.legend(loc="lower right", fontsize=9)

    _save(fig, out_path)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the n=500 timing sweep with prior-scale anchors."
    )
    parser.add_argument(
        "--exp2-csv", type=str, default="results/exp2/timing_n500.csv",
        help="Output CSV from experiments/exp2_timing_scaling.py.",
    )
    parser.add_argument(
        "--exp1-csv", type=str, default="results/exp1/timing.csv",
        help="CPU timing CSV from experiments/exp1_metric_comparison.py.",
    )
    parser.add_argument(
        "--n100-csv", type=str,
        default="results/scaling/n100_overnight_snd_log.csv",
        help="Online n=100 training SND log.",
    )
    parser.add_argument(
        "--out", type=str, default="Paper/figures/timing_n500.pdf",
    )
    parser.add_argument("--target-n", type=int, default=500)
    args = parser.parse_args()

    df_exp2 = pd.read_csv(args.exp2_csv)
    df_exp1 = pd.read_csv(args.exp1_csv) if Path(args.exp1_csv).exists() else None
    df_n100 = pd.read_csv(args.n100_csv) if Path(args.n100_csv).exists() else None

    plot(
        df_exp2=df_exp2,
        df_exp1=df_exp1,
        df_n100=df_n100,
        out_path=Path(args.out),
        target_n=args.target_n,
    )


if __name__ == "__main__":
    main()
