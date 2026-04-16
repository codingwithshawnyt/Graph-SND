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

# 2. Run the metric-comparison experiment. Produces CSVs + summary.json
#    under results/exp1/.
python experiments/exp1_metric_comparison.py \
    --checkpoint-dir checkpoints \
    --out results/exp1

# 3. Render the four paper figures from the CSVs.
python experiments/exp1_plots.py --results results/exp1

# 4. Unit tests (Prop 1, Prop 2, Prop 5, Thm 6, Wasserstein identities).
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
  exp1_plots.py               renders recovery / unbiasedness / concentration / timing PDFs

tests/
  test_wasserstein.py  test_metrics.py  test_graphs.py
```

## Experiment 1 headline numbers

From `results/exp1/summary.json` on the committed run (seed 42):

| Claim | What we checked | Observed |
|-------|-----------------|----------|
| Prop 1 (K_n recovery) | `|SND - Graph-SND(K_n)|` at n=4 | `0.0` exactly (both iter 0 and iter 100) |
| Prop 5 (HT unbiased) | `|bias| / SE` of HT mean at n=8, 2000 draws per `p` | max `1.42` across all `p in {0.1, 0.25, 0.5, 0.75}` |
| Thm 6 (concentration) | empirical `P(|est - SND| >= t_Hoeffding)` at n=16, 2000 draws per `m`, `delta = 0.1` | `0.00` across all m (max 80% of pairs) |
| Timing | full-SND / sampled-Graph-SND wall-clock ratio | up to `13.4x` at p=0.1; about `1.0x` at p=1.0 |

## Paper

See the companion LaTeX draft for the formal statements of the
propositions and theorems this code validates.
