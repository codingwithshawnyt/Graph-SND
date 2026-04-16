# Graph-SND

Scalable System Neural Diversity via graph-based local aggregation and
sampling-based estimators. Companion code for the paper extending
*System Neural Diversity* (Bettini, Shankar, Prorok; JMLR 2025).

## What it implements

- **Full SND** (eq. 2 of the paper): the uniform mean of Wasserstein
  behavioral distances over all agent pairs.
- **Graph-SND** (eq. 4 of this paper): a weighted mean of behavioral
  distances over the edges of a graph `G`. Recovers full SND exactly on
  the complete graph with unit weights.
- **Horvitz-Thompson sampled estimator**: unbiased estimator of full SND
  obtained by including each pair in `E` independently with probability
  `p`, then reweighting by `1/p`.
- **Uniform-weight variant** (Remark 2, Theorem 6): sample mean over a
  fixed-size edge sample. Subject to Hoeffding-type concentration.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[test]
```

## Run experiment 1 (metric comparison)

```bash
# 1. Train heterogeneous per-agent Gaussian policies briefly on VMAS
#    Multi-Agent Goal Navigation. Saves iter_0 and iter_100 checkpoints
#    for each n.
python training/train_navigation.py --n-agents 4  --iters 100
python training/train_navigation.py --n-agents 8  --iters 100
python training/train_navigation.py --n-agents 16 --iters 100

# 2. Run the metric-comparison experiment. Produces CSVs + PDFs
#    under results/exp1/.
python experiments/exp1_metric_comparison.py \
    --checkpoint-dir checkpoints \
    --out results/exp1

# 3. Unit tests (Prop 1, Prop 2, Prop 5, Thm 6, Wasserstein identities)
pytest tests/ -v
```

## Layout

```
graphsnd/
  wasserstein.py   closed-form W_2 between Gaussians (diag + full cov)
  metrics.py       snd, graph_snd, ht_estimator, uniform_sample_estimator,
                   pairwise_behavioral_distance
  graphs.py        complete_edges, bernoulli_edges, uniform_size_edges, knn_edges
  policies.py      GaussianMLPPolicy (per-agent, no param sharing)
  rollouts.py      collect_rollouts -> observations Tensor[T, n, obs_dim]

training/
  train_navigation.py   minimal IPPO on raw VMAS navigation

experiments/
  exp1_metric_comparison.py   Prop 1 @ n=4, Prop 5 @ n=8, Thm 6 @ n=16, timing sweep

tests/
  test_wasserstein.py  test_metrics.py  test_graphs.py
```

## Paper

See the companion LaTeX draft for the formal statements of the
propositions and theorems this code validates.
