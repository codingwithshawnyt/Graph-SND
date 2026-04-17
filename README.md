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

## Scaling experiment: training at $n = 50, 100$

The Experiment-1 pipeline (`training/train_navigation.py`) instantiates one `nn.Linear` per agent per layer, which is fine up to $n = 16$ but becomes Python-loop bound at $n \geq 50$. For the scaling experiment we ship a second trainer, `training/train_navigation_batched.py`, that stacks the $n$ per-agent MLPs into a single `BatchedGaussianMLPPolicy` whose forward pass is one `torch.bmm` call per layer. Parameters remain fully independent across agents, so the gradient for agent $i$ still depends only on agent $i$'s loss (see `tests/test_batched_policies.py::test_gradient_is_independent_across_agents`).

Two additional features matter for long runs:

- **Online Graph-SND cost logging.** At every `--snd-every` iterations the trainer measures both full SND and Graph-SND on the current rollout and appends the values *and wall-clock costs* to `results/scaling/n{n}_{tag}_snd_log.csv`. At $n = 100$ that is $\binom{100}{2} = 4{,}950$ Wasserstein evaluations for full SND versus roughly $500$ at $p = 0.1$, so the operational speedup from Prop. 6 becomes visible during training, not just at the end.
- **Periodic checkpointing and resume.** Checkpoints are written every `--ckpt-every` iterations, the latest is duplicated to `checkpoints/n{n}_{tag}_latest.pt`, and `--resume` restores policy, value net, optimizer, and RNGs so an overnight run can survive a crash or reboot.

### Run overnight on 2 GPUs

Preflight first — this runs the test suite, probes every visible CUDA device, and completes a 2-iteration smoke training on each one. It exits in under two minutes if anything is wrong with the setup:

```bash
bash scripts/preflight.sh
```

Then launch the overnight run. The scripts detach via `nohup`, stream logs to `logs/`, and write the pid to `logs/n{n}_{tag}.pid` so you can close the terminal:

```bash
bash scripts/run_overnight_n100.sh                    # cuda:0, n=100, 5000 iters
DEVICE=cuda:1 bash scripts/run_overnight_n50.sh       # cuda:1, n=50, 5000 iters
```

Override any parameter via environment variables, e.g. `ITERS=10000 bash scripts/run_overnight_n100.sh`. To resume, pass `RESUME=checkpoints/n100_overnight_latest.pt`. During the run, follow progress with `tail -f logs/n100_overnight_*.log`; the SND / speedup line is printed every `SND_EVERY` iterations (default 50):

```
[SND] full=0.1542 (4950 pairs, 412.7ms)  Graph-SND(p=0.10)=0.1535 (498 pairs, 42.1ms)  speedup=9.80x
```

### Loading a scaling checkpoint

Every training checkpoint ships with a sidecar `.metric.pt` written in the per-agent format, so the existing Experiment-1 evaluator consumes it without changes:

```python
from graphsnd.batched_policies import load_batched_checkpoint
from graphsnd.policies import load_checkpoint

batched_policy, batched_value, _ = load_batched_checkpoint(
    "checkpoints/n100_overnight_iter5000.metric.pt"
)
per_agent_policies, per_agent_values, _ = load_checkpoint(
    "checkpoints/n100_overnight_iter5000.metric.pt"
)
```

Both calls return objects that produce bit-identical forward outputs (this is asserted in the batched-policy test file).

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
├── graphsnd/                     # core package
│   ├── wasserstein.py            # closed-form W₂ between Gaussians (diag & full cov)
│   ├── metrics.py                # SND, Graph-SND, HT estimator, uniform estimator
│   ├── graphs.py                 # complete, Bernoulli, uniform-size, and k-NN graph generators
│   ├── policies.py               # GaussianMLPPolicy (independent per-agent, no parameter sharing)
│   ├── batched_policies.py       # BatchedGaussianMLPPolicy for scalable training (n >= 50)
│   └── rollouts.py               # evaluation rollout collection
├── training/
│   ├── train_navigation.py       # minimal IPPO for Exp. 1 (n in {4, 8, 16})
│   └── train_navigation_batched.py  # scalable batched IPPO for the scaling experiment
├── experiments/
│   ├── exp1_metric_comparison.py # validation data for Props. 2, 6, 7 and Thm. 9
│   └── exp1_plots.py             # Matplotlib rendering for the four paper PDFs
├── scripts/
│   ├── preflight.sh              # preflight check before overnight runs
│   ├── run_overnight_n100.sh     # detach overnight n=100 training on cuda:0
│   └── run_overnight_n50.sh      # detach overnight n=50 training on cuda:1
├── ControllingBehavioralDiversity-fork/  # vendored DiCo + Graph-SND integration (Section 6.7)
│   ├── GRAPH_SND_CHANGES.md      # list of modifications vs upstream DiCo
│   └── LICENSE                   # DiCo / ProrokLab terms (see Third-party code)
└── tests/
    ├── test_wasserstein.py       # distance identities
    ├── test_metrics.py           # metric properties and bounds
    ├── test_graphs.py            # graph generators
    └── test_batched_policies.py  # batched ↔ per-agent equivalence
```

## Third-party code

This repository vendors a fork of [DiCo](https://github.com/proroklab/ControllingBehavioralDiversity)
at `ControllingBehavioralDiversity-fork/` for the Graph-DiCo integration experiment (Section 6.7 of the
paper). **If you use this code in research, cite DiCo** as requested upstream: Bettini, M.,
Kortvelesy, R., & Prorok, A. (2024). *Controlling Behavioral Diversity in Multi-Agent Reinforcement
Learning* [Conference paper]. Forty-first International Conference on Machine Learning. For BibTeX
and machine-readable metadata, use `ControllingBehavioralDiversity-fork/CITATION.cff` (the
`preferred-citation` block matches the sentence above). The fork is used under DiCo's original terms
(see `ControllingBehavioralDiversity-fork/LICENSE`). Modifications to enable Graph-SND as a diversity
control target are documented in `ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md`.

## Citation and reference

Please refer to the companion paper for the formal definitions, proofs, and discussion of Graph-SND's graph-based aggregation and sampling semantics.
