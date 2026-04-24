#!/usr/bin/env bash
# n=50 Dispersion: FULL-SND DiCo sweep over DESIRED_SND × seeds, head-to-head
# with the existing Bernoulli-0.1 sweep produced by
# launch_n50_bern_setpoint_sweep.sh.
#
# Output layout (same RESULTS_BASE as the Bernoulli sweep so full/ and bern/
# sit side-by-side per cell, which lets n50_bern_vs_full_comparison.py glob
# both estimators in one pass):
#
#   RESULTS_BASE/
#     seed0/
#       snd0p12/{bern,full}/graph_snd_log.csv
#       snd0p14/{bern,full}/graph_snd_log.csv
#       snd0p15/{bern,full}/graph_snd_log.csv
#     seed1/...
#     seed2/...
#
# Memory envelope: same ENV_N=120, FRAMES_PER_BATCH=12000 as the Bernoulli
# sweep. Full SND at n=50 does 25x more pairwise Wasserstein calls per PPO
# forward (1225 pairs vs ~123 for p=0.1), but each pair is a scalar-valued
# op on Gaussian parameters, so the added memory footprint is negligible
# compared to the rollout buffer and PPO grad graph. OOM is handled the
# same way as the Bernoulli launcher (detect in log, retry on same GPU,
# optionally shrink env_n/frames_per_batch each attempt).
#
# Concurrency: two sequential worker pipelines (one per GPU) run in
# parallel. Within a GPU, cells run one at a time so full-SND at n=50
# never fights itself for HBM. Cells are dealt round-robin across GPUs so
# both cards are saturated for the full duration.
#
# IMPORTANT: like launch_n50_bern_setpoint_sweep.sh, we never wrap the
# runner in "$(...)" — subshell capture breaks `wait` with "pid is not a
# child of this shell". Instead each worker is a shell *function*
# backgrounded with `&` so $! inside the worker is a real child of that
# worker's subshell.
#
# Usage (from fork root, tmux strongly recommended — see RUNBOOK note at
# bottom of this file):
#   bash scripts/launch_n50_full_setpoint_sweep.sh
#   FORCE=1 bash scripts/launch_n50_full_setpoint_sweep.sh   # overwrite existing CSV dirs
#
# Optional env:
#   SNDS="0.12 0.14 0.15"     SEEDS="0 1 2"    RESULTS_BASE=...   MAX_ITERS=167
#   MAX_OOM_RETRIES=0   # 0=unlimited, N>0=cap attempts per cell at N
#   OOM_RETRY_SLEEP_SEC=20   OOM_SHRINK=1   OOM_MIN_ENV_N=32
#   ENV_N=120   FRAMES_PER_BATCH=12000
#   DRY_RUN=1   # print the per-cell plan without launching anything

set -uo pipefail

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
ENV_N="${ENV_N:-120}"
FRAMES_PER_BATCH="${FRAMES_PER_BATCH:-$((ENV_N * 100))}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final_n50_setpoint_sweep}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-0}"
OOM_RETRY_SLEEP_SEC="${OOM_RETRY_SLEEP_SEC:-20}"
OOM_SHRINK="${OOM_SHRINK:-1}"
OOM_MIN_ENV_N="${OOM_MIN_ENV_N:-32}"
MIN_CSV_LINES="${MIN_CSV_LINES:-150}"

read -r -a SNDS <<<"${SNDS:-0.12 0.14 0.15}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"

# ---------------------------------------------------------------------------
# Preflight: announce the plan and (optionally) exit without launching.
# ---------------------------------------------------------------------------

echo "[$(date -Is)] host=$(hostname) user=$(whoami) cwd=${ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(date -Is)] nvidia-smi --query-gpu=index,name,memory.total --format=csv:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
else
  echo "[$(date -Is)] WARN: nvidia-smi not found on PATH — GPU visibility may be misconfigured." >&2
fi

echo "[$(date -Is)] plan: n=${N_AGENTS}  iters=${MAX_ITERS}  env_n=${ENV_N}  batch=${FRAMES_PER_BATCH}"
echo "[$(date -Is)] plan: seeds=(${SEEDS[*]})  snds=(${SNDS[*]})"
echo "[$(date -Is)] plan: results_base=${RESULTS_BASE}"

# Build the cell list up front (deterministic order: for each SND, for each
# SEED). Dealing them round-robin to 2 GPUs gives a roughly balanced slate
# (5 cells on GPU 0, 4 on GPU 1 for the default 3x3 grid).
CELLS=()
for SND in "${SNDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CELLS+=("${SEED}:${SND}")
  done
done

CELLS_GPU0=()
CELLS_GPU1=()
for (( i=0; i<${#CELLS[@]}; i++ )); do
  if (( i % 2 == 0 )); then
    CELLS_GPU0+=("${CELLS[$i]}")
  else
    CELLS_GPU1+=("${CELLS[$i]}")
  fi
done

echo "[$(date -Is)] plan: GPU 0 cells (${#CELLS_GPU0[@]}): ${CELLS_GPU0[*]}"
echo "[$(date -Is)] plan: GPU 1 cells (${#CELLS_GPU1[@]}): ${CELLS_GPU1[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(date -Is)] DRY_RUN=1 — plan printed above, exiting without launching any training."
  exit 0
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

run_full() {
  # Launch a single full-SND training process in the background, pinned to
  # the given GPU. Returns $! in the caller's scope.
  local SEED="$1" DESIRED_SND="$2" GPU="$3" eff_n="$4" eff_batch="$5"
  local TAG="${DESIRED_SND//./p}"
  local OUT="${RESULTS_BASE}/seed${SEED}/snd${TAG}/full"
  local LOG="${ROOT}/logs/n50_seed${SEED}_snd${TAG}_full.log"

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
    "model.diversity_estimator=full" \
    "model.diversity_p=1.0" \
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

run_cell_with_oom_retries() {
  local SEED="$1" DESIRED_SND="$2" GPU="$3"
  local TAG="${DESIRED_SND//./p}"
  local OUT="${RESULTS_BASE}/seed${SEED}/snd${TAG}/full"
  local LOG="${ROOT}/logs/n50_seed${SEED}_snd${TAG}_full.log"
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

    run_full "$SEED" "$DESIRED_SND" "$GPU" "$eff_n" "$eff_batch"
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

worker_for_gpu() {
  # Runs all cells for a single GPU sequentially, each with its own
  # OOM-retry loop. Tracks per-worker aggregate status but always
  # continues to the next cell on failure so one bad cell doesn't block
  # the other 8.
  local GPU="$1"
  shift
  local CELLS=("$@")
  local failures=0
  for cell in "${CELLS[@]}"; do
    local SEED="${cell%%:*}"
    local SND="${cell##*:}"
    if run_cell_with_oom_retries "$SEED" "$SND" "$GPU"; then
      :
    else
      failures=$((failures + 1))
      echo "[$(date -Is)] worker GPU=${GPU} continuing past failed cell seed=${SEED} snd=${SND}" >&2
    fi
  done
  echo "[$(date -Is)] worker GPU=${GPU} done, failures=${failures}"
  return "$failures"
}

# ---------------------------------------------------------------------------
# Launch: one worker per GPU, concurrent, parent waits on both.
# ---------------------------------------------------------------------------

worker_for_gpu 0 "${CELLS_GPU0[@]}" &
PID_W0=$!
worker_for_gpu 1 "${CELLS_GPU1[@]}" &
PID_W1=$!

echo "[$(date -Is)] spawned worker PIDs: GPU 0 -> ${PID_W0}, GPU 1 -> ${PID_W1}"

rc0=0
rc1=0
wait "$PID_W0" || rc0=$?
wait "$PID_W1" || rc1=$?

echo
echo "[$(date -Is)] done. worker exit codes: GPU 0 -> ${rc0}, GPU 1 -> ${rc1}"
echo "[$(date -Is)] outputs under: ${RESULTS_BASE}/"
echo "[$(date -Is)] next step on local machine:"
echo "  rsync -av --include='**/graph_snd_log.csv' --filter='-! */' \\"
echo "      <user>@<riddle>:${RESULTS_BASE}/ \\"
echo "      ControllingBehavioralDiversity-fork/results/neurips_final_n50_setpoint_sweep/"
echo "  python experiments/n50_bern_vs_full_comparison.py"

if (( rc0 != 0 || rc1 != 0 )); then
  exit 1
fi
exit 0
