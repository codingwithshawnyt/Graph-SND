#!/usr/bin/env bash
# DiCo-validated sanity baseline: Dispersion n=4 share_rew=True MADDPG,
# estimator="full", 30 iters, ONE GPU, no Graph-SND variants.
#
# This script deliberately does NOT modify the existing
# run_graph_dico_two_gpus_then_third.sh orchestrator. It's a
# single-job, single-GPU diagnostic: "does this fork learn anything at
# n=4 on a DiCo-paper-validated task (Dispersion n=4 share_rew=True)
# in a 30-iter budget?"
#
# After this exits, inspect the CSV (graph_snd_log.csv in the Hydra
# output dir) and decide which branch of DIAGNOSIS.md Postmortem #2
# applies:
#
#   - reward grows + applied_snd tracks desired_snd  => fork is healthy,
#     the n=4 Navigation divergence is task-extrapolation specific.
#   - reward stays near zero on Dispersion too       => fork has a broader
#     regression vs upstream DiCo; next plan is a dep / config diff.
#
# Usage (from repo fork root):
#   # Foreground (prints to stdout, easy to tail):
#   bash scripts/run_dispersion_sanity.sh
#
#   # Background over ssh:
#   nohup bash scripts/run_dispersion_sanity.sh \
#       > logs/dispersion_sanity.log 2>&1 &
#
# Override defaults:
#   GPU=1 MAX_ITERS=50 DESIRED_SND=0.3 bash scripts/run_dispersion_sanity.sh
#
# Budget: ~15-20 min on a modern single GPU at MAX_ITERS=30 with the
# default off-policy settings (6k frames/batch * 30 iters = 180k frames,
# 30k optimizer steps total).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

GPU="${GPU:-0}"
MAX_ITERS="${MAX_ITERS:-30}"
DESIRED_SND="${DESIRED_SND:-0.1}"
LOG_PATH="logs/dispersion_sanity_n4_snd${DESIRED_SND}.log"

echo "[$(date -Is)] Dispersion MADDPG sanity (n=4, share_rew=True, estimator=full)"
echo "[$(date -Is)]   GPU=${GPU}  MAX_ITERS=${MAX_ITERS}  DESIRED_SND=${DESIRED_SND}"
echo "[$(date -Is)]   log: ${LOG_PATH}"

# One logical GPU visible to the process; pass cuda:0 for train/sampling
# devices so the Hydra/TorchRL pipeline doesn't try to index a missing GPU.
CUDA_VISIBLE_DEVICES="${GPU}" python het_control/run_scripts/run_dispersion_maddpg_full.py \
  --config-name dispersion_maddpg_full_config \
  experiment.loggers=[] \
  experiment.max_n_iters="${MAX_ITERS}" \
  experiment.train_device=cuda:0 \
  experiment.sampling_device=cuda:0 \
  model.desired_snd="${DESIRED_SND}" \
  > "${LOG_PATH}" 2>&1

echo "[$(date -Is)] Run finished. Hydra output dir holds graph_snd_log.csv:"
grep -Eo '/[^ ]*graph_snd_log.csv' "${LOG_PATH}" | tail -n 1 || true
echo "[$(date -Is)] Tail of ${LOG_PATH}:"
tail -n 20 "${LOG_PATH}" || true
