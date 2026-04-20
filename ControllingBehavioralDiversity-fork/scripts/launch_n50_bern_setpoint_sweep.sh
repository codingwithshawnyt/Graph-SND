#!/usr/bin/env bash
# n=50 Dispersion: Bernoulli-0.1 DiCo sweep over DESIRED_SND × seeds.
#
# Uses the same memory-related Hydra overrides as launch_dico_n50_feasibility.sh
# (ENV_N, FRAMES_PER_BATCH). Do NOT use launch_bern_dico.sh at n=50 without these.
#
# IMPORTANT: never capture the launcher with "$(run_bern ...)" — that runs in a
# subshell and breaks `wait` ("pid is not a child of this shell"); jobs may
# orphan but keep training.
#
# Usage (from fork root, tmux recommended):
#   bash scripts/launch_n50_bern_setpoint_sweep.sh
#   FORCE=1 bash scripts/launch_n50_bern_setpoint_sweep.sh   # overwrite existing CSV dirs
#
# Optional env:
#   SNDS="0.12 0.14 0.15"   SEEDS="0 1 2"   RESULTS_BASE=...   MAX_ITERS=167

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

if [[ -f "$ROOT/../.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/../.venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)
LOGGERS='experiment.loggers=[]'
N_AGENTS="${N_AGENTS:-50}"
MAX_ITERS="${MAX_ITERS:-167}"
P="${P:-0.1}"
ENV_N="${ENV_N:-120}"
FRAMES_PER_BATCH="${FRAMES_PER_BATCH:-$((ENV_N * 100))}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final_n50_setpoint_sweep}"
FORCE="${FORCE:-0}"

case "${P}" in
  0.1|0.10|1e-1) ESTIMATOR=graph_p01 ;;
  0.25|0.250) ESTIMATOR=graph_p025 ;;
  *)
    echo "ERROR: P=${P} not supported (use 0.1 or 0.25)." >&2
    exit 2
    ;;
esac

run_bern() {
  local SEED="$1" DESIRED_SND="$2" GPU="$3"
  local TAG="${DESIRED_SND//./p}"
  local OUT="${RESULTS_BASE}/seed${SEED}/snd${TAG}/bern"
  local LOG="${ROOT}/logs/n50_seed${SEED}_snd${TAG}_bern.log"

  mkdir -p "$(dirname "$OUT")"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$OUT"
  fi
  mkdir -p "$OUT"

  echo "[$(date -Is)] start seed=${SEED} DESIRED_SND=${DESIRED_SND} GPU=${GPU} -> ${OUT}"
  # Must run in THIS shell (no command substitution) so $! is waitable.
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${LOGGERS}" \
    "seed=${SEED}" \
    "task.n_agents=${N_AGENTS}" \
    "task.n_food=${N_FOOD}" \
    "task.share_reward=true" \
    "experiment.max_n_iters=${MAX_ITERS}" \
    "experiment.on_policy_n_envs_per_worker=${ENV_N}" \
    "experiment.on_policy_collected_frames_per_batch=${FRAMES_PER_BATCH}" \
    "experiment.render=false" \
    "experiment.train_device=cuda:0" \
    "experiment.sampling_device=cuda:0" \
    "experiment.buffer_device=cpu" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=${ESTIMATOR}" \
    "model.diversity_p=${P}" \
    "hydra.run.dir=${OUT}" \
    >"${LOG}" 2>&1 &
}

health() {
  local pid="$1" log="$2"
  sleep 20
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: pid ${pid} died early; tail ${log}" >&2
    tail -n 120 "$log" >&2 || true
    exit 1
  fi
}

read -r -a SNDS <<<"${SNDS:-0.12 0.14 0.15}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"

pids=()
logs=()

flush() {
  if ((${#pids[@]} == 0)); then return 0; fi
  local i pid
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    health "$pid" "${logs[$i]}"
  done
  for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
  pids=()
  logs=()
}

for SND in "${SNDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    gpu=$((${#pids[@]} % 2))
    TAG="${SND//./p}"
    LOG="${ROOT}/logs/n50_seed${SEED}_snd${TAG}_bern.log"

    if [[ -s "${RESULTS_BASE}/seed${SEED}/snd${TAG}/bern/graph_snd_log.csv" && "$FORCE" != "1" ]]; then
      continue
    fi

    run_bern "$SEED" "$SND" "$gpu"
    # Last background PID from this shell (not from a subshell).
    pids+=("$!")
    logs+=("$LOG")

    if ((${#pids[@]} == 2)); then
      flush
    fi
  done
done
flush

echo "[$(date -Is)] done. Outputs under: ${RESULTS_BASE}/"
