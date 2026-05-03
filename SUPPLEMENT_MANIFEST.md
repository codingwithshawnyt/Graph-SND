# Supplement Manifest

The NeurIPS supplement is built by:

```bash
python scripts/build_neurips_supplement.py --check
```

The archive is intentionally curated instead of being a raw repository
zip. This keeps review focused on the code and artifacts that directly
support the paper.

## Included

- `graphsnd/`: core Graph-SND package.
- `tests/`: unit tests for graphs, metrics, Wasserstein distances, and
  batched policy equivalence.
- `training/`: IPPO training scripts used for VMAS navigation runs.
- `experiments/`: scripts used to produce result CSVs, summaries, and
  plots.
- selected `scripts/`: reproducibility launchers and supplement builder.
- selected `results/`: committed CSV/JSON/PDF artifacts used by the
  paper.
- `checkpoints/*_meta.json`: checkpoint metadata without binary model
  weights.
- selected `ControllingBehavioralDiversity-fork/` files: Graph-SND
  integration code, relevant configs, tests, launch scripts, and the
  upstream DiCo notice/license.

## Excluded

- `.git`, virtual environments, caches, local editor state, and logs.
- `Paper/` source and compiled paper PDFs; the paper is uploaded
  separately in OpenReview.
- binary training checkpoints (`*.pt`) and heavyweight raw Hydra trees.
- local machine/cluster helper scripts with private hostnames or paths.
- untracked scratch outputs and exploratory result dumps.

## Anonymization Checks

The builder scans included text files for author identifiers, private
paths, private hostnames, and non-anonymous repository URLs. The build
fails if any such string appears in the archive.

