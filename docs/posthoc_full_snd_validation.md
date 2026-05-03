# n=50 DiCo Post-Hoc Full-SND Validation Runbook

This run validates that Bernoulli-0.1 Graph-SND DiCo does more than track its
own sparse controller signal. During training it logs `posthoc_full_snd`, the
complete-graph SND of the scaled actions actually sent to the environment.

The launcher runs the same grid as the current n=50 head-to-head:

- seeds: `0 1 2`
- desired SND values: `0.12 0.14 0.15`
- estimator arms: `bern full`
- total jobs: `18`
- default GPUs: physical `0 1`

## On the 2x4090 Machine

```bash
ssh shawnr@172.24.170.204

# Use your existing checkout if it is already there; otherwise clone it.
cd ~/Graph-SND 2>/dev/null || git clone git@github.com:codingwithshawnyt/Graph-SND.git ~/Graph-SND
cd ~/Graph-SND
git fetch origin
git checkout main
git pull --ff-only origin main

# Activate the environment you normally use for these DiCo runs.
# If it does not exist yet, create it and install the repo package.
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'

# The DiCo fork also needs the same VMAS/TensorDict/TorchRL/BenchMARL stack used
# by the previous DiCo runs. If that environment already exists, keep using it.
cd ControllingBehavioralDiversity-fork

# Sanity-check the launch plan without starting jobs.
DRY_RUN=1 bash scripts/launch_n50_posthoc_full_snd_validation.sh

# Start the two-GPU sweep inside tmux.
tmux new -s n50-posthoc
GPUS="0 1" \
POSTHOC_INTERVAL=1 \
POSTHOC_SUBSAMPLE=4096 \
bash scripts/launch_n50_posthoc_full_snd_validation.sh
```

Detach from tmux with `Ctrl-b d`.

## Monitoring

```bash
cd ~/Graph-SND/ControllingBehavioralDiversity-fork
tail -f logs/n50_posthoc_*.log

find results/neurips_final_n50_posthoc_full_snd \
  -name graph_snd_log.csv -print -exec wc -l {} \;
```

A complete default run should produce 18 CSV files, each with roughly 168 lines
(header plus 167 PPO iterations).

## Summarize On The GPU Machine

From the repo root:

```bash
cd ~/Graph-SND
source .venv/bin/activate
python experiments/n50_posthoc_full_snd_validation.py \
  --root ControllingBehavioralDiversity-fork/results/neurips_final_n50_posthoc_full_snd \
  --out-dir results/dico_n50_posthoc_full_snd
```

## Copy Results Back To The Local Machine

Run this from the local machine:

```bash
cd /Users/shawn/Documents/Research/Graph-SND

rsync -avz \
  shawnr@172.24.170.204:~/Graph-SND/ControllingBehavioralDiversity-fork/results/neurips_final_n50_posthoc_full_snd/ \
  ControllingBehavioralDiversity-fork/results/neurips_final_n50_posthoc_full_snd/

rsync -avz \
  shawnr@172.24.170.204:~/Graph-SND/results/dico_n50_posthoc_full_snd/ \
  results/dico_n50_posthoc_full_snd/
```

Then regenerate the summary locally if needed:

```bash
source .venv/bin/activate
python experiments/n50_posthoc_full_snd_validation.py \
  --root ControllingBehavioralDiversity-fork/results/neurips_final_n50_posthoc_full_snd \
  --out-dir results/dico_n50_posthoc_full_snd
```
