# Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning

Graph-based local aggregation and sampling-based estimation of the
System Neural Diversity (SND) metric for multi-agent reinforcement
learning. Companion code and experimental pipeline for the paper
**Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent
Reinforcement Learning**, which extends the original *System Neural
Diversity* metric (Bettini, Shankar, Prorok; JMLR 2025).

## Overview

SND is a rigorous metric for behavioural heterogeneity in MARL, but its
uniform pairwise aggregation scales as $\mathcal{O}(n^2)$. This
repository implements **Graph-SND**, which evaluates pairwise
behavioural distances only on the edges of a user-specified graph $G$,
and aggregates them as a normalised weighted mean.

Two interpretations fall out of the same definition:

- **Fixed $G$** (communication graph, $k$-NN in observation space, a
  user prior) — a **localised diversity measure** with cost
  $\mathcal{O}(|E|)$ that recovers full SND exactly on $G = K_n$.
- **Random $G$** (Bernoulli-$p$ with Horvitz–Thompson weights) — an
  **unbiased sampling estimator** of full SND with a Hoeffding-type
  $\mathcal{O}(1/\sqrt{m})$ bound in the number of sampled edges $m$,
  independent of $n$.

Every SND consumer that treats the metric as a scalar can consume
Graph-SND as a drop-in replacement. Most importantly, the
[DiCo diversity controller](https://github.com/proroklab/ControllingBehavioralDiversity)
accepts Graph-SND without any architectural change (see
Section 6.7 of the paper).

### Core implementations

- **Full SND** (`graphsnd/metrics.py`): uniform mean of Wasserstein
  behavioural distances over all $\binom{n}{2}$ unique agent pairs.
- **Graph-SND** (`graphsnd/metrics.py`): weighted mean over the edges
  of a specified graph. Recovers full SND exactly on $K_n$.
- **Horvitz–Thompson estimator** (`graphsnd/metrics.py`): unbiased
  sampling estimator of full SND via Bernoulli-$p$ inclusion and
  $1/p$ reweighting.
- **Uniform-weight variant**: sample mean over a fixed-size edge
  subsample, subject to the Hoeffding- and Serfling-type
  concentration bounds proven in the paper.
- **Graph generators** (`graphsnd/graphs.py`): complete, Bernoulli,
  uniform-size, and $k$-nearest-neighbour graphs.

---

## Installation

We recommend a clean virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
```

On Windows, use `.\.venv\Scripts\activate` instead of
`source .venv/bin/activate`. The quotes around `'.[test]'` avoid
`zsh` glob expansion.

---

## Reproducing the experiments

The `experiments/` directory holds every script used to generate the
validation data and figures for Section 6 of the paper. All random
seeds are fixed at $42$ unless otherwise noted.

### 1. Train the benchmark policies ($n \in \{4, 8, 16\}$)

```bash
python training/train_navigation.py --n-agents 4  --iters 100
python training/train_navigation.py --n-agents 8  --iters 100
python training/train_navigation.py --n-agents 16 --iters 100
```

Both iter-0 and iter-100 checkpoints are saved under `checkpoints/`.
Binary `.pt` files are `.gitignore`d; run the training locally to
materialise them.

### 2. Run the four core validations

Computes recovery, unbiasedness, concentration, and CPU-timing
datasets at $n \in \{4, 8, 16\}$, and writes them to
`results/exp1/`.

```bash
python experiments/exp1_metric_comparison.py \
    --checkpoint-dir checkpoints \
    --out results/exp1
```

Render the four paper PDFs:

```bash
python experiments/exp1_plots.py --results results/exp1
```

### 3. GPU frozen-init timing sweep at $n \in \{100, 250, 500\}$

Verifies that Proposition 6's $\mathcal{O}(n^2) \to \mathcal{O}(|E|)$
speedup continues to hold at the scale where full SND is
computationally meaningful on modern GPUs (Section 6.5.1).

```bash
python experiments/exp2_timing_scaling.py \
    --n-agents "100,250,500" \
    --device cuda:0 \
    --out results/exp2/timing_n100_250_500.csv
```

Every call uses `torch.cuda.synchronize` on both sides of a
`time.perf_counter` measurement, with 20 trials per cell after 3
warm-ups. The companion figure is
`Paper/figures/timing_n500.pdf`.

### 4. Run the test suite

Validates the closed-form Wasserstein identities, metric properties,
graph generators, and batched-↔-per-agent policy equivalence.

```bash
pytest tests/ -v
```

---

## $n = 100$ overnight PPO scaling run

The Experiment 1 pipeline (`training/train_navigation.py`)
instantiates one `nn.Linear` per agent per layer, which becomes
Python-loop-bound at $n \geq 50$. `training/train_navigation_batched.py`
stacks the $n$ per-agent MLPs into a single
`BatchedGaussianMLPPolicy` whose forward pass is one `torch.bmm`
call per layer; parameters stay fully independent across agents
(`tests/test_batched_policies.py::test_gradient_is_independent_across_agents`).

Two additional features matter for long runs:

- **Online Graph-SND cost logging.** Every `--snd-every` iterations
  the trainer measures both full SND and Graph-SND on the current
  rollout and appends the values *and wall-clock costs* to
  `results/scaling/n{n}_{tag}_snd_log.csv`. At $n = 100$ that is
  $\binom{100}{2} = 4{,}950$ Wasserstein evaluations for full SND
  versus roughly $500$ at $p = 0.1$, so the Proposition 6 speedup
  is visible during training, not just at the end.
- **Periodic checkpointing and resume.** Checkpoints are written
  every `--ckpt-every` iterations, the latest is duplicated to
  `checkpoints/n{n}_{tag}_latest.pt`, and `--resume` restores
  policy, value net, optimiser, and RNGs so an overnight run can
  survive a crash or reboot.

### Launch

```bash
bash scripts/preflight.sh                                   # 2-min sanity check
bash scripts/run_overnight_n100.sh                          # cuda:0, n=100
DEVICE=cuda:1 bash scripts/run_overnight_n50.sh             # optional cuda:1
```

Override any parameter via environment variables, e.g.
`ITERS=10000 bash scripts/run_overnight_n100.sh`. To resume, pass
`RESUME=checkpoints/n100_overnight_latest.pt`. During the run,
follow progress with `tail -f logs/n100_overnight_*.log`; the
SND / speedup line prints every `SND_EVERY` iterations (default 50):

```
[SND] full=0.1542 (4950 pairs, 412.7ms)  Graph-SND(p=0.10)=0.1535 (498 pairs, 42.1ms)  speedup=9.80x
```

### Loading a scaling checkpoint

Every training checkpoint ships with a sidecar `.metric.pt` written
in the per-agent format, so the Experiment 1 evaluator consumes it
unchanged:

```python
from graphsnd.batched_policies import load_batched_checkpoint
from graphsnd.policies import load_checkpoint

batched_policy, batched_value, _ = load_batched_checkpoint(
    "checkpoints/n100_overnight_iter500.metric.pt"
)
per_agent_policies, per_agent_values, _ = load_checkpoint(
    "checkpoints/n100_overnight_iter500.metric.pt"
)
```

Both calls return objects with bit-identical forward outputs (asserted
in the batched-policy test file).

---

## DiCo + Graph-SND on VMAS Dispersion (Section 6.7)

`ControllingBehavioralDiversity-fork/` vendors the upstream
[DiCo](https://github.com/proroklab/ControllingBehavioralDiversity)
codebase with a `GRAPH_SND_CHANGES.md` summary of the modifications
needed to plug Graph-SND into the DiCo diversity controller. The
closed-loop experiment compares four configurations on VMAS
Dispersion ($n{=}10$, ten food targets, shared team reward,
$\SND_{\mathrm{des}} = 0.1$, three seeds):

| configuration                    | graph $G$                         | diversity estimator |
| :------------------------------- | :-------------------------------- | :------------------ |
| IPPO baseline (control disabled) | –                                 | –                   |
| DiCo + full SND                  | $K_{10}$                          | full                |
| DiCo + $k$-NN Graph-SND          | dynamic $k{=}3$ per env           | `graph_knn`         |
| DiCo + Bernoulli-$0.1$ Graph-SND | random, $p = 0.1$                 | `graph_p01`         |

Results live in `results/dico/` (CSV + summary PDF) and are rendered
by the DiCo fork's plot script. The multi-panel figure is
`Paper/figures/neurips_knn_plot.pdf`.

### Optional: DiCo + Bernoulli-0.1 feasibility at $n{=}50$ (riddle)

`ControllingBehavioralDiversity-fork/scripts/launch_dico_n50_feasibility.sh`
reproduces a single-seed, two-run feasibility demonstration at
${\sim}5\times$ the team size of the main Dispersion experiment: it
launches an IPPO baseline on GPU 0 and DiCo with Bernoulli-$0.1$
Graph-SND on GPU 1 in parallel, each for 167 PPO iterations with
`on_policy_n_envs_per_worker=120` and
`on_policy_collected_frames_per_batch=12000` (smaller than the
$n{=}10$ defaults to keep the rollout within a single 4090's memory
budget at $5\times$ more per-agent tensors).

On riddle:

```bash
cd ~/shawnr/Graph-SND  # or wherever the repo lives on riddle
source .venv/bin/activate
cd ControllingBehavioralDiversity-fork
bash scripts/launch_dico_n50_feasibility.sh
# Monitor (paths are under the fork, not the parent repo):
#   tail -f ControllingBehavioralDiversity-fork/logs/neurips_n50_seed0_*.log
# Artefacts:
#   results/neurips_final_n50/seed0/ippo/graph_snd_log.csv
#   results/neurips_final_n50/seed0/bern/graph_snd_log.csv
```

Then from the **local** repo root, copy the two CSVs. **Do not** rsync
both remote paths into one flat directory: each run is named
`graph_snd_log.csv`, so the second transfer would overwrite the
first. Create `ippo/` and `bern/` and pull each file separately
(replace `HOST` and the remote home path if riddle differs):

```bash
cd ~/Documents/Research/Graph-SND
mkdir -p ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/ippo \
         ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/bern

rsync -avz shawnr@HOST:~/shawnr/Graph-SND/ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/ippo/graph_snd_log.csv \
  ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/ippo/

rsync -avz shawnr@HOST:~/shawnr/Graph-SND/ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/bern/graph_snd_log.csv \
  ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/bern/
```

If you already ran a bad rsync, remove any stray
`ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/graph_snd_log.csv`
(single file directly under `seed0/`) and re-run the two commands
above. Check with:

```bash
wc -l ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/ippo/graph_snd_log.csv \
     ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/bern/graph_snd_log.csv
```

Then regenerate the appendix companion figure:

```bash
python3 scripts/plot_reward_curves.py \
    --figure-type panels --smooth 5 \
    "ippo:ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/ippo/graph_snd_log.csv" \
    "graph_p01:ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/bern/graph_snd_log.csv" \
    --desired-snd 0.1 \
    --task-name "VMAS Dispersion (n=50, one seed)" \
    --output Paper/figures/neurips_n50_feasibility.pdf
```

Finally flip the single Boolean flag at the bottom of
`Paper/main.tex` (`\hasneurisfiftyresultsfalse` →
`\hasneurisfiftyresultstrue`) and recompile to activate
Appendix~D. The flag is dormant by default, so a failed or skipped
run leaves the submitted PDF unchanged.

---

## Headline validation results

From the committed seed-42 runs (values match
`results/exp1/summary.json`):

| Theoretical claim                   | Experimental check                                                                                                               | Observed result                              |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------- |
| **Prop. 2** (recovery on $K_n$)     | Max absolute gap between full SND and Graph-SND on $K_n$, $n=4$                                                                  | `0.0` exactly (iter 0 and iter 100)          |
| **Prop. 7** (HT unbiasedness)       | Max $\lvert \text{bias} \rvert / \text{SE}$ for HT mean at $n=8$, 2000 draws per $p$                                             | `1.42` over $p \in \{0.1, 0.25, 0.5, 0.75\}$ |
| **Thm. 9** (concentration)          | Empirical $\mathbb{P}(\lvert \widehat{\text{SND}} - \text{SND}\rvert \geq t_H)$ at $n=16$, 2000 draws per $m$, $\delta = 0.1$    | `0.00` across all $m$                        |
| **Prop. 6** (complexity, CPU)       | Wall-clock ratio full / sampled at $n \in \{4, 8, 16\}$                                                                          | up to `13.4×` at $p=0.1$; `~1×` at $p=1.0$   |
| **Prop. 6** (complexity, GPU scale) | Mean speedup at $p=0.1$, $n \in \{100, 250, 500\}$, RTX 4090, 20 trials                                                          | `9.77×` / `10.04×` / `9.98×`                 |
| **DiCo + Graph-SND**                | Applied SND vs. $\SND_{\text{des}} = 0.1$ over 3 seeds on VMAS Dispersion, $n = 10$                                              | `0.1009–0.1010` (full / $k$-NN / Bernoulli)  |

---

## Repository layout

```text
.
├── graphsnd/                          # core package
│   ├── wasserstein.py                 # closed-form W₂ between Gaussians
│   ├── metrics.py                     # SND, Graph-SND, HT and uniform estimators
│   ├── graphs.py                      # complete, Bernoulli, uniform-size, k-NN graphs
│   ├── policies.py                    # GaussianMLPPolicy (one per agent)
│   ├── batched_policies.py            # BatchedGaussianMLPPolicy (stacked per-agent)
│   └── rollouts.py                    # rollout collection
├── training/
│   ├── train_navigation.py            # IPPO for Exp. 1 (n ∈ {4, 8, 16})
│   └── train_navigation_batched.py    # scalable IPPO for the n = 100 run
├── experiments/
│   ├── exp1_metric_comparison.py      # data for Props. 2, 6, 7 and Thm. 9 (CPU, n ≤ 16)
│   ├── exp1_plots.py                  # renders the four paper PDFs
│   └── exp2_timing_scaling.py         # GPU frozen-init timing at n ∈ {100, 250, 500}
├── scripts/
│   ├── preflight.sh                   # smoke-test every visible CUDA device
│   ├── run_overnight_n100.sh          # overnight n = 100 PPO on cuda:0
│   └── run_overnight_n50.sh           # overnight n = 50 PPO on cuda:1
├── Paper/
│   ├── main.tex                       # paper source (NeurIPS 2026 style)
│   ├── references.bib                 # all citations
│   ├── checklist.tex                  # filled NeurIPS reproducibility checklist
│   ├── neurips_2026.sty               # NeurIPS style file (vendored)
│   ├── figures/                       # generated figure PDFs
│   └── Formatting_Instructions_For_NeurIPS_2026/  # upstream NeurIPS template
├── ControllingBehavioralDiversity-fork/   # vendored DiCo fork (Section 6.7 integration)
│   ├── GRAPH_SND_CHANGES.md           # diff vs. upstream DiCo
│   └── LICENSE                        # DiCo / ProrokLab terms
├── results/
│   ├── exp1/                          # CSV + summary.json for the four small-n claims
│   ├── exp2/                          # timing_n100_250_500.csv + summary.json
│   ├── scaling/                       # n = 100 overnight logs and scaling_n100.pdf
│   └── dico/                          # VMAS Dispersion DiCo + Graph-SND CSVs and PDF
└── tests/
    ├── test_wasserstein.py            # distance identities
    ├── test_metrics.py                # metric properties and bounds
    ├── test_graphs.py                 # graph generators
    └── test_batched_policies.py       # batched ↔ per-agent equivalence
```

---

## Paper compilation

The paper is written against the NeurIPS 2026 style file. The current
draft is configured in **preprint** mode (non-anonymous, loads
reference PDF locally):

```latex
\usepackage[preprint]{neurips_2026}
```

To produce each of the three target artefacts:

| target                                | `\usepackage{neurips_2026}` options | author block | notes                                    |
| :------------------------------------ | :---------------------------------- | :----------- | :--------------------------------------- |
| local preprint / arXiv                | `[preprint]`                        | shown        | current `main.tex` default               |
| **double-blind submission**           | `[main]` (or no option)             | anonymised   | style file replaces authors; see below   |
| camera-ready (post-acceptance)        | `[main, final]`                     | shown        | line numbers off, track name in footer   |

**Double-blind toggle.** Switching the `\usepackage` option is
usually sufficient: the NeurIPS style file auto-replaces the author
block with `Anonymous Author(s) / Affiliation`, and `main.tex`
defines a `\codelocation` macro that auto-rewrites the code-release
sentence in the Conclusion from the GitHub URL (preprint / final) to
`Code will be released upon acceptance.` (double-blind). The named
acknowledgement in the `\begin{ack}` block is also hidden
automatically by the style file in anonymous mode, so no manual
edits are normally required — just change the `\usepackage` option,
recompile, and confirm no residual `Shawn Ray`, `CMU`, or
`codingwithshawnyt` strings appear in the PDF (e.g., with
`pdftotext main.pdf - | grep -Ei 'shawn|cmu|codingwith'`).

Compile the paper with either `pdflatex` or
[`tectonic`](https://github.com/tectonic-typesetting/tectonic):

```bash
cd Paper
# Option A: classic TeXLive
pdflatex main && bibtex main && pdflatex main && pdflatex main
# Option B: tectonic (single-command, self-bootstrapping)
tectonic -X compile main.tex
```

---

## Third-party code

This repository vendors a fork of
[DiCo](https://github.com/proroklab/ControllingBehavioralDiversity) in
`ControllingBehavioralDiversity-fork/` for the DiCo + Graph-SND
integration (Section 6.7 of the paper). **If you use this code in
research, cite DiCo** as requested upstream:

> Bettini, M., Kortvelesy, R., & Prorok, A. (2024). *Controlling
> Behavioral Diversity in Multi-Agent Reinforcement Learning.*
> Forty-first International Conference on Machine Learning.

The BibTeX and machine-readable metadata live in
`ControllingBehavioralDiversity-fork/CITATION.cff` (the
`preferred-citation` block matches the sentence above). The fork is
used under DiCo's original terms
(`ControllingBehavioralDiversity-fork/LICENSE`). All modifications
to enable Graph-SND as a diversity-control target are documented in
`ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md`.

## Citation

Please refer to the companion paper (`Paper/main.tex`) for the
formal definitions, proofs, and full experimental discussion of
Graph-SND.
