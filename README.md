# Graph-SND: Scalable System Neural Diversity

Scalable System Neural Diversity via graph-based local aggregation and sampling-based estimators. This repository contains the official companion code and experimental pipeline for the paper extending the original *System Neural Diversity* metric (Bettini, Shankar, Prorok; JMLR 2025).

## Overview

System Neural Diversity (SND) is a rigorous metric for quantifying behavioral heterogeneity in MARL, but its uniform pairwise aggregation scales quadratically, $\mathcal{O}(n^2)$. This package implements **Graph-SND**, extending the metric to scale linearly, $\mathcal{O}(|E|)$, while preserving its exact closed-form properties on continuous Gaussian policies.

### Core implementations

- **Full SND** (Eq. 2): The uniform mean of Wasserstein behavioral distances over all unique agent pairs.
- **Graph-SND** (Eq. 4): A weighted mean of behavioral distances strictly over the edges of a specified graph $G$. Recovers full SND exactly on the complete graph with unit weights.
- **Horvitz–Thompson estimator**: An unbiased sampling estimator of full SND obtained by including each pair independently with probability $p$, then reweighting by $1/p$.
- **Uniform-weight variant**: A sample mean over a fixed-size edge sample, subject to the Hoeffding-type concentration bounds proven in the paper.

---

## Installation

We recommend using a virtual environment. The package can be installed locally along with its testing and experimental dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
```

On Windows, use `.\.venv\Scripts\activate` instead of `source .venv/bin/activate`. The quotes around `'.[test]'` avoid shell glob expansion with `zsh`.

---

## Reproducing the experiments

The `experiments/` directory contains the scripts used to generate the validation data and plots for **Section 6** of the paper.

### 1. Train the policies

Train the heterogeneous, per-agent Gaussian policies on the VMAS *Multi-Agent Goal Navigation* environment. This script saves the `iter_0` (random initialization) and `iter_100` (trained) checkpoints for each team size under `checkpoints/` (large `.pt` files are gitignored by default; run training locally to produce them).

```bash
python training/train_navigation.py --n-agents 4 --iters 100
python training/train_navigation.py --n-agents 8 --iters 100
python training/train_navigation.py --n-agents 16 --iters 100
```

### 2. Run the metric validations

Run the core metric-comparison experiment. This computes the recovery, unbiasedness, concentration, and timing datasets, saving the raw `.csv` files and `summary.json` to `results/exp1/`.

```bash
python experiments/exp1_metric_comparison.py \
    --checkpoint-dir checkpoints \
    --out results/exp1
```

### 3. Generate the paper figures

Render the four plots as in the PDF (recovery, unbiasedness, concentration, and timing).

```bash
python experiments/exp1_plots.py --results results/exp1
```

### 4. Run the test suite

Validate the mathematical identities (Wasserstein closed-form correctness, metric properties, graph logic) via `pytest`.

```bash
pytest tests/ -v
```

---

## Experiment 1: headline results

The following table summarizes the automated validation of the paper’s theoretical claims (values from `results/exp1/summary.json` on the committed seed-42 run):

| Theoretical claim | Experimental check | Observed result |
| :--- | :--- | :--- |
| **Prop. 2** (recovery on $K_n$) | Max absolute gap between full SND and Graph-SND on the complete graph $K_n$, $n=4$ | `0.0` exactly (iter 0 and iter 100) |
| **Prop. 7** (HT unbiasedness) | Max $\lvert \mathrm{bias}\rvert / \mathrm{SE}$ for HT mean at $n=8$, 2000 draws per $p$ | Max `1.42` over $p \in \{0.1, 0.25, 0.5, 0.75\}$ |
| **Thm. 9** (concentration) | Empirical $\mathbb{P}\bigl(\lvert \widehat{\mathrm{SND}} - \mathrm{SND}\rvert \geq t_H\bigr)$ at $n=16$, 2000 draws per $m$, $\delta = 0.1$ | `0.00` across all $m$ (up to 80% of pairs) |
| **Prop. 6** (complexity) | Wall-clock ratio: full SND / sampled Graph-SND | Up to `13.4×` at $p=0.1$; $\sim 1.0\times$ at $p=1.0$ |

---

## Repository layout

```text
.
├── graphsnd/                 # core package
│   ├── wasserstein.py        # closed-form W₂ between Gaussians (diag & full cov)
│   ├── metrics.py            # SND, Graph-SND, HT estimator, uniform estimator
│   ├── graphs.py             # complete, Bernoulli, uniform-size, and k-NN graph generators
│   ├── policies.py           # GaussianMLPPolicy (independent per-agent, no parameter sharing)
│   └── rollouts.py           # evaluation rollout collection
├── training/
│   └── train_navigation.py   # minimal IPPO training loop for VMAS navigation
├── experiments/
│   ├── exp1_metric_comparison.py  # validation data for Props. 2, 6, 7 and Thm. 9
│   └── exp1_plots.py         # Matplotlib rendering for the four paper PDFs
└── tests/
    ├── test_wasserstein.py   # distance identities
    ├── test_metrics.py       # metric properties and bounds
    └── test_graphs.py        # graph generators
```

## Citation and reference

Please refer to the companion paper for the formal definitions, proofs, and discussion of Graph-SND’s graph-based aggregation and sampling semantics.
