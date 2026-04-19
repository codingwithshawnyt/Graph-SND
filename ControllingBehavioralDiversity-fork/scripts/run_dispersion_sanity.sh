#!/usr/bin/env bash
# DiCo-validated sanity baseline: Dispersion n=4 share_rew=True MADDPG,
# estimator="full", ONE GPU per process (see "Multi-GPU" below).
#
# This script deliberately does NOT modify the existing
# run_graph_dico_two_gpus_then_third.sh orchestrator.
#
# After this exits, inspect graph_snd_log.csv in the Hydra output dir
# (see DIAGNOSIS.md Postmortem #2).
#
# --- Riddle / cluster: use the machine well ---
#
# * One Python process here uses a single logical GPU (cuda:0 inside the
#   process after CUDA_VISIBLE_DEVICES is set). BenchMARL MADDPG in this
#   fork does not fan one experiment across multiple GPUs; to use every
#   GPU on the node, launch multiple independent runs in parallel with
#   different GPU= values (tmux panes, GNU parallel, or a small for-loop).
# * CPU: if OMP_NUM_THREADS / MKL_NUM_THREADS are unset, this script sets
#   them to the number of logical cores so PyTorch / MKL / NumPy do not
#   accidentally run single-threaded on a 64-core box.
# * For a stronger MADDPG learning signal than a 30-iter / 180k-frame run,
#   default MAX_ITERS is 200 (~1.2M frames); override MAX_ITERS=30 for a
#   quick crash/smoke. Short runs still have high exploration epsilon and
#   an empty replay buffer at the start; use HYDRA_EXTRA to shorten the
#   exploration anneal and add replay warmup, e.g.
#     HYDRA_EXTRA="experiment.exploration_anneal_frames=180000 experiment.off_policy_init_random_frames=5000"
#
# Usage (from repo fork root):
#   bash scripts/run_dispersion_sanity.sh
#
#   nohup bash scripts/run_dispersion_sanity.sh \
#       > logs/dispersion_sanity.log 2>&1 &
#
# Override defaults:
#   GPU=1 MAX_ITERS=200 DESIRED_SND=0.1 bash scripts/run_dispersion_sanity.sh
#
# Extra Hydra overrides (space-separated):
#   HYDRA_EXTRA="experiment.off_policy_init_random_frames=5000 experiment.exploration_anneal_frames=180000" \
#     GPU=0 MAX_ITERS=30 bash scripts/run_dispersion_sanity.sh
#
# Turn periodic evaluation back on (slower):
#   EVALUATION=true bash scripts/run_dispersion_sanity.sh
#
# Log path includes iter count and desired SND to avoid clobbering.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

_default_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null || echo 8
  else
    echo 8
  fi
}

# Use all CPU cores for BLAS / OpenMP unless the user already set these
# (e.g. in Slurm: export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK).
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS="$(_default_cores)"
fi
if [[ -z "${MKL_NUM_THREADS:-}" ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi
if [[ -z "${NUMEXPR_NUM_THREADS:-}" ]]; then
  export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
fi

GPU="${GPU:-0}"
# MADDPG needs buffer fill + exploration decay; 30 iters @ 6k frames is
# often misleading (see DIAGNOSIS.md / team notes). Default 200 iters.
MAX_ITERS="${MAX_ITERS:-200}"
DESIRED_SND="${DESIRED_SND:-0.1}"
EVALUATION="${EVALUATION:-false}"
LOG_PATH="logs/dispersion_sanity_n4_i${MAX_ITERS}_snd${DESIRED_SND}.log"

# Optional space-separated Hydra overrides (split on whitespace only;
# do not put spaces inside values).
HYDRA_EXTRA_ARGS=()
if [[ -n "${HYDRA_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a HYDRA_EXTRA_ARGS <<< "${HYDRA_EXTRA}"
fi

echo "[$(date -Is)] Dispersion MADDPG sanity (n=4, share_rew=True, estimator=full)"
echo "[$(date -Is)]   GPU=${GPU}  MAX_ITERS=${MAX_ITERS}  DESIRED_SND=${DESIRED_SND}"
echo "[$(date -Is)]   OMP_NUM_THREADS=${OMP_NUM_THREADS}  MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[$(date -Is)]   EVALUATION=${EVALUATION}"
if [[ ${#HYDRA_EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "[$(date -Is)]   HYDRA_EXTRA (${#HYDRA_EXTRA_ARGS[@]} args): ${HYDRA_EXTRA_ARGS[*]}"
fi
echo "[$(date -Is)]   log: ${LOG_PATH}"

# One logical GPU visible to the process; pass cuda:0 for train/sampling
# so the pipeline does not index a missing GPU.
CMD=(
  python het_control/run_scripts/run_dispersion_maddpg_full.py
  --config-name dispersion_maddpg_full_config
  experiment.loggers=[]
  "experiment.max_n_iters=${MAX_ITERS}"
  experiment.train_device=cuda:0
  experiment.sampling_device=cuda:0
  "model.desired_snd=${DESIRED_SND}"
  "experiment.evaluation=${EVALUATION}"
  experiment.create_json=false
)
if [[ ${#HYDRA_EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${HYDRA_EXTRA_ARGS[@]}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_PATH}" 2>&1

echo "[$(date -Is)] Run finished. Hydra output dir holds graph_snd_log.csv:"
grep -Eo '/[^ ]*graph_snd_log.csv' "${LOG_PATH}" | tail -n 1 || true
echo "[$(date -Is)] Tail of ${LOG_PATH}:"
tail -n 20 "${LOG_PATH}" || true
