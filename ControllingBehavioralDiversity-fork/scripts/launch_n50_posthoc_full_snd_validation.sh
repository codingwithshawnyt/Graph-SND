#!/usr/bin/env bash
# n=50 Dispersion post-hoc full-SND validation.
#
# Runs the Bernoulli-0.1 Graph-SND controller and the full-SND controller on
# the same 3 seeds x 3 desired-SND grid, with an extra CSV diagnostic enabled:
# posthoc_full_snd = complete-graph SND of the *scaled* actions sent to the
# environment. This directly tests whether sparse DiCo produces policies whose
# actual full-SND diversity matches the target, rather than only tracking its
# own sparse feedback signal.
#
# Output layout:
#   RESULTS_BASE/
#     seed0/snd0p12/{bern,full}/graph_snd_log.csv
#     seed0/snd0p14/{bern,full}/graph_snd_log.csv
#     ...
#
# Usage from ControllingBehavioralDiversity-fork/:
#   bash scripts/launch_n50_posthoc_full_snd_validation.sh
#   DRY_RUN=1 bash scripts/launch_n50_posthoc_full_snd_validation.sh
#   FORCE=1 bash scripts/launch_n50_posthoc_full_snd_validation.sh
#
# Useful env:
#   GPUS="0 1"                  physical GPUs to use
#   SNDS="0.12 0.14 0.15"       desired SND values
#   SEEDS="0 1 2"               seeds
#   ESTIMATORS="bern full"      run both arms by default
#   RESULTS_BASE=...            output root
#   POSTHOC_INTERVAL=1          compute diagnostic every N PPO iterations
#   POSTHOC_SUBSAMPLE=4096      max rollout observations, all C(n,2) pairs
#   MAX_ITERS=167 ENV_N=120 FRAMES_PER_BATCH=12000
#   MAX_OOM_RETRIES=0           0 = unlimited retries per cell

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
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final_n50_posthoc_full_snd}"
POSTHOC_INTERVAL="${POSTHOC_INTERVAL:-1}"
POSTHOC_SUBSAMPLE="${POSTHOC_SUBSAMPLE:-4096}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-0}"
OOM_RETRY_SLEEP_SEC="${OOM_RETRY_SLEEP_SEC:-20}"
OOM_SHRINK="${OOM_SHRINK:-1}"
OOM_MIN_ENV_N="${OOM_MIN_ENV_N:-32}"
MIN_CSV_LINES="${MIN_CSV_LINES:-150}"

read -r -a GPUS <<<"${GPUS:-0 1}"
read -r -a SNDS <<<"${SNDS:-0.12 0.14 0.15}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"
read -r -a ESTIMATORS <<<"${ESTIMATORS:-bern full}"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

slot_for_index() {
  local idx="$1"
  local n_gpus="${#GPUS[@]}"
  local n_estimators="${#ESTIMATORS[@]}"
  if ((n_estimators > 1)); then
    echo $(((idx + idx / n_estimators) % n_gpus))
  else
    echo $((idx % n_gpus))
  fi
}

if ((${#GPUS[@]} == 0)); then
  echo "ERROR: GPUS is empty." >&2
  exit 2
fi

echo "[$(ts)] host=$(hostname) user=$(whoami) cwd=${ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
else
  echo "[$(ts)] WARN: nvidia-smi not found." >&2
fi

echo "[$(ts)] plan: n=${N_AGENTS} iters=${MAX_ITERS} env_n=${ENV_N} batch=${FRAMES_PER_BATCH}"
echo "[$(ts)] plan: seeds=(${SEEDS[*]}) snds=(${SNDS[*]}) estimators=(${ESTIMATORS[*]})"
echo "[$(ts)] plan: posthoc interval=${POSTHOC_INTERVAL} subsample=${POSTHOC_SUBSAMPLE}"
echo "[$(ts)] plan: results_base=${RESULTS_BASE}"

CELLS=()
for SND in "${SNDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for EST in "${ESTIMATORS[@]}"; do
      case "$EST" in
        bern|full) CELLS+=("${SEED}:${SND}:${EST}") ;;
        *)
          echo "ERROR: unknown estimator arm ${EST}; expected bern or full." >&2
          exit 2
          ;;
      esac
    done
  done
done

for ((i = 0; i < ${#GPUS[@]}; i++)); do
  assigned=()
  for ((j = 0; j < ${#CELLS[@]}; j++)); do
    if (($(slot_for_index "$j") == i)); then
      assigned+=("${CELLS[$j]}")
    fi
  done
  echo "[$(ts)] plan: GPU ${GPUS[$i]} cells (${#assigned[@]}): ${assigned[*]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$(ts)] DRY_RUN=1; exiting without launching training."
  exit 0
fi

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

estimator_overrides() {
  local est="$1"
  if [[ "$est" == "bern" ]]; then
    echo "model.diversity_estimator=graph_p01 model.diversity_p=0.1"
  else
    echo "model.diversity_estimator=full model.diversity_p=1.0"
  fi
}

run_one() {
  local seed="$1" desired_snd="$2" est="$3" gpu="$4" eff_n="$5" eff_batch="$6"
  local tag="${desired_snd//./p}"
  local out="${RESULTS_BASE}/seed${seed}/snd${tag}/${est}"
  local log="${ROOT}/logs/n50_posthoc_seed${seed}_snd${tag}_${est}.log"
  local overrides

  overrides="$(estimator_overrides "$est")"
  mkdir -p "$(dirname "$out")"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$out"
  fi
  mkdir -p "$out"

  echo "[$(ts)] start seed=${seed} snd=${desired_snd} est=${est} gpu=${gpu} env_n=${eff_n} batch=${eff_batch} -> ${out}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${LOGGERS}" \
    "seed=${seed}" \
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
    "model.desired_snd=${desired_snd}" \
    ${overrides} \
    "graph_snd_posthoc_full_snd_interval=${POSTHOC_INTERVAL}" \
    "graph_snd_posthoc_full_snd_subsample=${POSTHOC_SUBSAMPLE}" \
    "hydra.run.dir=${out}" \
    >"${log}" 2>&1 &
}

run_cell_with_oom_retries() {
  local seed="$1" desired_snd="$2" est="$3" gpu="$4"
  local tag="${desired_snd//./p}"
  local out="${RESULTS_BASE}/seed${seed}/snd${tag}/${est}"
  local log="${ROOT}/logs/n50_posthoc_seed${seed}_snd${tag}_${est}.log"
  local csv="${out}/graph_snd_log.csv"

  if [[ -s "$csv" && "$FORCE" != "1" ]] && csv_sufficient "$csv"; then
    echo "[$(ts)] skip complete: $csv"
    return 0
  fi

  local attempt=1
  local eff_n="${ENV_N}"
  local eff_batch="${FRAMES_PER_BATCH}"
  while true; do
    rm -rf "$out"
    mkdir -p "$out"
    : >"$log"

    run_one "$seed" "$desired_snd" "$est" "$gpu" "$eff_n" "$eff_batch"
    local pid=$!
    sleep 20
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$(ts)] WARN: pid ${pid} exited before 20s; check ${log}" >&2
    fi

    local rc=0
    wait "$pid" || rc=$?
    if [[ "$rc" -eq 0 ]] && csv_sufficient "$csv"; then
      echo "[$(ts)] ok seed=${seed} snd=${desired_snd} est=${est} gpu=${gpu} attempt=${attempt} lines=$(wc -l <"$csv" | tr -d ' ')"
      return 0
    fi

    if log_is_oom "$log"; then
      if ((MAX_OOM_RETRIES > 0 && attempt >= MAX_OOM_RETRIES)); then
        echo "ERROR: OOM retries exhausted for seed=${seed} snd=${desired_snd} est=${est}" >&2
        tail -n 80 "$log" >&2 || true
        return 1
      fi
      echo "[$(ts)] OOM seed=${seed} snd=${desired_snd} est=${est} gpu=${gpu} attempt=${attempt}" >&2
      if [[ "$OOM_SHRINK" == "1" ]]; then
        eff_n=$((eff_n * 3 / 4))
        ((eff_n < OOM_MIN_ENV_N)) && eff_n=${OOM_MIN_ENV_N}
        eff_batch=$((eff_n * 100))
        echo "[$(ts)] shrink -> env_n=${eff_n} batch=${eff_batch}" >&2
      fi
      sleep "$OOM_RETRY_SLEEP_SEC"
      attempt=$((attempt + 1))
      continue
    fi

    echo "ERROR: seed=${seed} snd=${desired_snd} est=${est} failed rc=${rc}; tail ${log}" >&2
    tail -n 80 "$log" >&2 || true
    return 1
  done
}

worker_for_gpu() {
  local gpu="$1"
  shift
  local cells=("$@")
  local failures=0
  for cell in "${cells[@]}"; do
    local seed="${cell%%:*}"
    local rest="${cell#*:}"
    local snd="${rest%%:*}"
    local est="${rest##*:}"
    if ! run_cell_with_oom_retries "$seed" "$snd" "$est" "$gpu"; then
      failures=$((failures + 1))
      echo "[$(ts)] worker gpu=${gpu} continuing after failed cell ${cell}" >&2
    fi
  done
  echo "[$(ts)] worker gpu=${gpu} done failures=${failures}"
  return "$failures"
}

PIDS=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  assigned=()
  for ((j = 0; j < ${#CELLS[@]}; j++)); do
    if (($(slot_for_index "$j") == i)); then
      assigned+=("${CELLS[$j]}")
    fi
  done
  worker_for_gpu "${GPUS[$i]}" "${assigned[@]}" &
  PIDS+=("$!")
done

failures=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

echo "[$(ts)] done. worker_failures=${failures}"
echo "[$(ts)] outputs under: ${RESULTS_BASE}/"
echo "[$(ts)] summarize with:"
echo "  python experiments/n50_posthoc_full_snd_validation.py --root ${RESULTS_BASE}"

if ((failures > 0)); then
  exit 1
fi
exit 0
