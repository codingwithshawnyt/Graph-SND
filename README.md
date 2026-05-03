# Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning

This repository contains the anonymous code supplement for the NeurIPS
submission **Graph-SND: Sparse Aggregation for Behavioral Diversity in
Multi-Agent Reinforcement Learning**.

Graph-SND is a sparse aggregation layer for System Neural Diversity (SND).
Instead of averaging behavioral distances over all
$\binom{n}{2}$ agent pairs, it averages over the edges of a graph $G$.
The same implementation supports:

- complete-graph recovery of full SND;
- fixed-graph localized diversity measurement;
- Bernoulli and fixed-size random edge sampling for scalable SND
  estimation;
- random-regular expander aggregation for deterministic sparse
  approximations.

The supplement is organized so reviewers can quickly verify the paper's
code-backed claims from source, tests, and committed result summaries.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test,plot]'
python -m pytest tests -q
```

The core tests cover graph generators, SND/Graph-SND estimators,
Wasserstein identities, concentration radii, and the batched policy code
used for the $n=100$ PPO scaling run.

## Core Package

- `graphsnd/metrics.py`: full SND, Graph-SND, Horvitz-Thompson
  estimation, finite-population sample means, and concentration radii.
- `graphsnd/graphs.py`: complete graphs, Bernoulli samples, fixed-size
  samples, $k$-nearest-neighbor graphs, random regular graphs, and
  spectral diagnostics.
- `graphsnd/wasserstein.py` and `graphsnd/tvd.py`: behavioral distances
  used in the Gaussian/Wasserstein and categorical/TVD experiments.
- `graphsnd/batched_policies.py`: stacked independent per-agent policies
  used to make the $n=100$ PPO run practical without changing the policy
  factorization.

## Reproducing Paper Evidence

The committed CSV/JSON summaries and plotting scripts let reviewers
inspect the exact numeric evidence without rerunning all GPU experiments.
The commands below regenerate the main lightweight figures and tables.

```bash
# Core recovery, unbiasedness, concentration, and small-n timing plots.
python experiments/exp1_plots.py --results results/exp1

# GPU timing figure from committed timing CSV.
python experiments/plot_timing_n500.py \
  --csv results/exp2/timing_n100_250_500.csv \
  --output results/exp2/timing_n100_250_500.pdf

# Expander sparsification figures from committed CSVs.
python experiments/exp3_plots.py \
  --csv results/exp3/expander_distortion.csv \
  --output-main results/exp3/expander_distortion_main.pdf \
  --output-appendix results/exp3/expander_distortion_appendix.pdf

# DiCo n=50 Bernoulli-vs-full summary table and figure.
python experiments/n50_bern_vs_full_comparison.py \
  --results-base results/dico_n50_bern_vs_full
```

Several experiments require a CUDA GPU to rerun from scratch. The
committed summaries are included so the numerical claims remain
inspectable on CPU-only machines.

## Claim-to-Code Map

| Paper claim | Code / artifact |
| :--- | :--- |
| Complete graph recovers SND exactly | `tests/test_metrics.py`, `experiments/exp1_metric_comparison.py`, `results/exp1/recovery.csv` |
| Horvitz-Thompson estimator is unbiased | `graphsnd/metrics.py::ht_estimator`, `tests/test_metrics.py`, `results/exp1/unbiasedness.csv` |
| Uniform samples concentrate with $O(1/\sqrt{m})$ radius | `graphsnd/metrics.py::hoeffding_bound`, `tests/test_metrics.py`, `results/exp1/concentration.csv` |
| Runtime scales with $|E|$ rather than $\binom{n}{2}$ | `graphsnd/metrics.py::graph_snd_from_rollouts`, `experiments/exp2_timing_scaling.py`, `results/exp2/timing_n100_250_500.csv` |
| Random regular graphs give accurate sparse aggregators | `graphsnd/graphs.py::random_regular_edges`, `experiments/exp3_expander_distortion.py`, `results/exp3/expander_distortion.csv` |
| Graph-SND can replace full SND in DiCo | `ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md`, selected DiCo integration files, `results/dico_n50_bern_vs_full/` |

## DiCo Integration

`ControllingBehavioralDiversity-fork/` contains a focused fork of the
DiCo code path used for the closed-loop diversity-control experiments.
The supplement packager includes only the Graph-SND integration files
needed to inspect or reapply the modification. The upstream DiCo notice
and license are preserved in that bundle.

For the exact integration summary, see
`ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md`.

## Building the Anonymous Supplement Zip

```bash
python scripts/build_neurips_supplement.py --check
```

The builder writes `dist/graph_snd_neurips_supplement.zip` and fails if:

- the archive exceeds the NeurIPS supplementary material size limit;
- files outside the curated whitelist are included;
- private paths, hostnames, or author-identifying strings appear in
  included text files.

See `SUPPLEMENT_MANIFEST.md` for the exact inclusion/exclusion policy.

