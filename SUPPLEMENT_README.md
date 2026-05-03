# Anonymous NeurIPS Code Supplement: Graph-SND

This supplement supports the submission **Graph-SND: Sparse Aggregation
for Behavioral Diversity in Multi-Agent Reinforcement Learning**. It is
designed for double-blind review: author identities, private hostnames,
private paths, and non-anonymous repository URLs are excluded by the
supplement builder.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test,plot]'
python -m pytest tests -q
```

The core package has no dependency on the DiCo fork. DiCo integration
experiments require the optional `dico` extra and a CUDA-capable
environment:

```bash
pip install -e '.[dico]'
```

Some upstream DiCo dependencies may need platform-specific CUDA wheels.
The committed CSV/JSON summaries are included so the paper's numerical
claims can still be inspected without rerunning long GPU jobs.

## Fast Verification

```bash
python -m pytest tests -q
python experiments/exp1_plots.py --results results/exp1
python experiments/exp3_plots.py \
  --csv results/exp3/expander_distortion.csv \
  --output-main results/exp3/expander_distortion_main.pdf \
  --output-appendix results/exp3/expander_distortion_appendix.pdf
```

Expected result: all unit tests pass, and the plotting commands recreate
the committed result figures from CSV/JSON data.

## Paper Claim Verification

| Claim | Minimal verification path |
| :--- | :--- |
| Complete-graph Graph-SND equals full SND | `python -m pytest tests/test_metrics.py -q`; inspect `results/exp1/recovery.csv` |
| Bernoulli Horvitz-Thompson estimator is unbiased | `graphsnd.metrics.ht_estimator`; inspect `results/exp1/unbiasedness.csv` |
| Uniform edge samples concentrate as a function of sampled edge count | `graphsnd.metrics.hoeffding_bound`; inspect `results/exp1/concentration.csv` |
| Sparse runtime follows $\binom{n}{2}/|E|$ | inspect `results/exp2/timing_n100_250_500.csv` |
| Expander sparse aggregation is accurate and has spectral diagnostics | run `experiments/exp3_plots.py`; inspect `results/exp3/expander_distortion.csv` |
| DiCo can use Graph-SND as a control signal | inspect `ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md` and `results/dico_n50_bern_vs_full/` |

## Hardware Notes

- Unit tests and figure regeneration from committed CSVs run on CPU.
- VMAS training and timing sweeps are faster on CUDA GPUs.
- The DiCo $n=50$ validation runs are GPU experiments; the supplement
  includes logs/summaries sufficient for review without rerunning them.

