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
# OOM handling: if the run exits with CUDA OOM (detected in the log), the same
# cell is retried on the SAME physical GPU after a short sleep, indefinitely
# by default (MAX_OOM_RETRIES=0). Optionally shrink ENV_N / FRAMES_PER_BATCH
# each attempt (OOM_SHRINK=1, default). Set MAX_OOM_RETRIES=N (N>0) to cap.
#
# Runs one training job at a time (sequential) so two sweeps never fight for
# the same GPU memory on shared hosts; alternate GPUs with SWEEP_ALTERNATE_GPUS=1.
#
# Usage (from fork root, tmux recommended):
#   bash scripts/launch_n50_bern_setpoint_sweep.sh
#   FORCE=1 bash scripts/launch_n50_bern_setpoint_sweep.sh   # overwrite existing CSV dirs
#
# Optional env:
#   SNDS="0.12 0.14 0.15"   SEEDS="0 1 2"   RESULTS_BASE=...   MAX_ITERS=167
#   MAX_OOM_RETRIES=0       # default 0 = unlimited OOM retries; set N>0 to cap at N attempts/cell
#   OOM_RETRY_SLEEP_SEC=20   OOM_SHRINK=1   OOM_MIN_ENV_N=32
#   SWEEP_GPU=0             # pin every cell to this GPU (ignore alternation)
#   SWEEP_ALTERNATE_GPUS=1  # default: round-robin 0,1,0,1,... (set 0 to always use SWEEP_GPU)

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

# 0 = unlimited OOM retries (same GPU); N>0 = stop after N attempts for that cell.
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-0}"
OOM_RETRY_SLEEP_SEC="${OOM_RETRY_SLEEP_SEC:-20}"
OOM_SHRINK="${OOM_SHRINK:-1}"
OOM_MIN_ENV_N="${OOM_MIN_ENV_N:-32}"
MIN_CSV_LINES="${MIN_CSV_LINES:-150}"
SWEEP_ALTERNATE_GPUS="${SWEEP_ALTERNATE_GPUS:-1}"
SWEEP_GPU="${SWEEP_GPU:-0}"

case "${P}" in
  0.1|0.10|1e-1) ESTIMATOR=graph_p01 ;;
  0.25|0.250) ESTIMATOR=graph_p025 ;;
  *)
    echo "ERROR: P=${P} not supported (use 0.1 or 0.25)." >&2
    exit 2
    ;;
esac

log_is_oom() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  grep -qE 'OutOfMemoryError|CUDA out of memory' "$log"
}

csv_sufficient() {
  local csv="$1"
  [[ -s "$csv" ]] || return 1
  local n
  n=$(wc -l <"$csv" | tr -d ' ')
  ((n >= MIN_CSV_LINES))
}

run_bern() {
  local SEED="$1" DESIRED_SND="$2" GPU="$3" eff_n="$4" eff_batch="$5"
  local TAG="${DESIRED_SND//./p}"
  local OUT="${RESULTS_BASE}/seed${SEED}/snd${TAG}/bern"
  local LOG="${ROOT}/logs/n50_seed${SEED}_snd${TAG}_bern.log"

  mkdir -p "$(dirname "$OUT")"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$OUT"
  fi
  mkdir -p "$OUT"

  echo "[$(date -Is)] start seed=${SEED} DESIRED_SND=${DESIRED_SND} GPU=${GPU} env_n=${eff_n} batch=${eff_batch} -> ${OUT}"
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${LOGGERS}" \
    "seed=${SEED}" \
    "task.n_agents=${N_AGENTS}" \
    "task.n_food=${N_FOOD}" \
    "task.share_reward=true" \
    "experiment.max_n_iters=${MAX_ITERS}" \
    "experiment.on_policy_n_envs_per_worker=${eff_n}" \
    "experiment.on_policy_collected_frames_per_batch=${eff_batch}" \
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
    echo "[$(date -Is)] WARN: pid ${pid} gone before 20s; check ${log}" >&2
    return 1
  fi
  return 0
}

# Run one (SEED, SND, GPU) cell until success or non-OOM failure.
# On OOM: same GPU, clear output dir, optional shrink env/batch, sleep, retry forever
# unless MAX_OOM_RETRIES>0 (then stop after that many attempts for this cell).
run_cell_with_oom_retries() {
  local SEED="$1" DESIRED_SND="$2" GPU="$3"
  local TAG="${DESIRED_SND//./p}"
  local OUT="${RESULTS_BASE}/seed${SEED}/snd${TAG}/bern"
  local LOG="${ROOT}/logs/n50_seed${SEED}_snd${TAG}_bern.log"
  local csv="${OUT}/graph_snd_log.csv"

  if [[ -s "$csv" && "$FORCE" != "1" ]] && csv_sufficient "$csv"; then
    echo "[$(date -Is)] skip (complete): $csv"
    return 0
  fi

  local attempt=1
  local eff_n="${ENV_N}"
  local eff_batch="${FRAMES_PER_BATCH}"

  while true; do
    rm -rf "$OUT"
    mkdir -p "$OUT"
    : >"${LOG}"

    run_bern "$SEED" "$DESIRED_SND" "$GPU" "$eff_n" "$eff_batch"
    local pid=$!
    health "$pid" "$LOG" || true

    local rc=0
    wait "$pid" || rc=$?

    if [[ "$rc" -eq 0 ]] && csv_sufficient "$csv"; then
      echo "[$(date -Is)] ok seed=${SEED} snd=${DESIRED_SND} gpu=${GPU} attempt=${attempt} lines=$(wc -l <"$csv" | tr -d ' ')"
      return 0
    fi

    if log_is_oom "$LOG"; then
      local cap_msg="unlimited"
      if ((MAX_OOM_RETRIES > 0)); then
        cap_msg="${MAX_OOM_RETRIES}"
        if ((attempt >= MAX_OOM_RETRIES)); then
          echo "ERROR: OOM retries exhausted (MAX_OOM_RETRIES=${MAX_OOM_RETRIES}) seed=${SEED} snd=${DESIRED_SND} gpu=${GPU}" >&2
          tail -n 60 "$LOG" >&2 || true
          return 1
        fi
      fi
      echo "[$(date -Is)] OOM seed=${SEED} snd=${DESIRED_SND} gpu=${GPU} attempt=${attempt} (cap=${cap_msg}) env_n=${eff_n} batch=${eff_batch}" >&2
      if [[ "$OOM_SHRINK" == "1" ]]; then
        eff_n=$((eff_n * 3 / 4))
        ((eff_n < OOM_MIN_ENV_N)) && eff_n=${OOM_MIN_ENV_N}
        eff_batch=$((eff_n * 100))
        echo "[$(date -Is)] shrink -> env_n=${eff_n} batch=${eff_batch}" >&2
      fi
      sleep "${OOM_RETRY_SLEEP_SEC}"
      attempt=$((attempt + 1))
      continue
    fi

    echo "ERROR: seed=${SEED} snd=${DESIRED_SND} gpu=${GPU} failed (rc=${rc}), not classified as OOM. Tail ${LOG}" >&2
    tail -n 80 "$LOG" >&2 || true
    return 1
  done
}

read -r -a SNDS <<<"${SNDS:-0.12 0.14 0.15}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"

idx=0
for SND in "${SNDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    if [[ "${SWEEP_ALTERNATE_GPUS}" == "1" ]]; then
      gpu=$((idx % 2))
    else
      gpu=${SWEEP_GPU}
    fi
    idx=$((idx + 1))

    run_cell_with_oom_retries "$SEED" "$SND" "$gpu"
  done
done

echo "[$(date -Is)] done. Outputs under: ${RESULTS_BASE}/"
