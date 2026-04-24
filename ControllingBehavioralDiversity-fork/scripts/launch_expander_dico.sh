#!/usr/bin/env bash
# Expander-GG DiCo on VMAS Dispersion — single seed, single GPU.
#
# Runs DiCo with a random d-regular (expander) graph as the Graph-SND
# estimator. The expander graph is resampled each estimate_snd call via
# the same per-iteration reseed scheme as Bernoulli.
#
# Output:
#   results/neurips_final/seed${SEED}/expander/graph_snd_log.csv
#
# Usage:
#   SEED=0 bash scripts/launch_expander_dico.sh
#   SEED=1 GPU=1 bash scripts/launch_expander_dico.sh
#
# Multi-seed driver:
#   SEEDS="0 1 2" bash scripts/launch_expander_dico_seeds.sh
#
# Env vars:
#   SEED          (default 0)
#   GPU           (default 0)
#   N_AGENTS      (default 10)
#   MAX_ITERS     (default 167)
#   DESIRED_SND   (default 0.1)
#   RR_D          (default 3)   — expander degree; n*d must be even
#   N_FOOD        (default N_AGENTS)
#   FORCE         (default 0)
#   RESULTS_BASE  (default $ROOT/results/neurips_final)
#
# Auto-abort: kills the process if iteration 10 is not reached within
# 2 hours (polls CSV row count every 5 min).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

if [[ -f "$ROOT/../.venv/bin/activate" ]]; then
    source "$ROOT/../.venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

SEED="${SEED:-0}"
GPU="${GPU:-0}"
N_AGENTS="${N_AGENTS:-10}"
MAX_ITERS="${MAX_ITERS:-167}"
DESIRED_SND="${DESIRED_SND:-0.1}"
RR_D="${RR_D:-3}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
FORCE="${FORCE:-0}"
LOGGERS='experiment.loggers=[]'

RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final}"
RESULTS_DIR="${RESULTS_BASE}/seed${SEED}"
EXP_DIR="${RESULTS_DIR}/expander"

if [[ -s "${EXP_DIR}/graph_snd_log.csv" && "$FORCE" != "1" ]]; then
    echo "ERROR: ${EXP_DIR} already contains graph_snd_log.csv." >&2
    echo "       Re-run with FORCE=1 to wipe and replace." >&2
    exit 1
fi
if [[ "$FORCE" == "1" ]]; then
    rm -rf "$EXP_DIR"
fi
mkdir -p "$EXP_DIR"

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)

HYDRA_COMMON=(
    "${LOGGERS}"
    "seed=${SEED}"
    "task.n_agents=${N_AGENTS}"
    "task.n_food=${N_FOOD}"
    "task.share_reward=true"
    "experiment.max_n_iters=${MAX_ITERS}"
    "experiment.render=false"
    "experiment.train_device=cuda:0"
    "experiment.sampling_device=cuda:0"
    "experiment.buffer_device=cpu"
)

EXP_LOG="logs/neurips_final_seed${SEED}_expander_d${RR_D}.log"

echo "[$(date -Is)] seed=${SEED} gpu=${GPU} estimator=expander d=${RR_D} -> ${EXP_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=expander" \
    "model.diversity_expander_d=${RR_D}" \
    "hydra.run.dir=${EXP_DIR}" \
    > "${EXP_LOG}" 2>&1 &
PID=$!

echo "[$(date -Is)] Spawned PID=${PID} (GPU ${GPU}); first health check in 10 s..."
sleep 10

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: Expander DiCo (PID ${PID}) exited immediately. Tail:" >&2
    tail -n 80 "${EXP_LOG}" >&2 || true
    exit 1
fi

# --- Auto-abort: kill if iteration 10 not reached within 2 hours ---
ABORT_TIMEOUT_SEC=7200   # 2 hours
POLL_INTERVAL_SEC=300    # 5 minutes
MIN_ITERS=10
ELAPSED=0

while kill -0 "$PID" 2>/dev/null; do
    sleep "${POLL_INTERVAL_SEC}"
    ELAPSED=$((ELAPSED + POLL_INTERVAL_SEC))

    CSV="${EXP_DIR}/graph_snd_log.csv"
    if [[ -s "$CSV" ]]; then
        ROWS=$(wc -l < "$CSV" | tr -d ' ')
        # Subtract 1 for header
        ITERS=$((ROWS - 1))
        if [[ "$ITERS" -ge "$MIN_ITERS" ]]; then
            # Progress is fine; disable abort and just wait
            echo "[$(date -Is)] Reached iter ${ITERS} >= ${MIN_ITERS}; auto-abort disabled. Waiting for completion..."
            break
        fi
    fi

    if [[ "$ELAPSED" -ge "$ABORT_TIMEOUT_SEC" ]]; then
        echo "[$(date -Is)] AUTO-ABORT: iteration ${MIN_ITERS} not reached within $((ABORT_TIMEOUT_SEC / 3600))h. Killing PID ${PID}." >&2
        kill "$PID" 2>/dev/null || true
        sleep 2
        kill -9 "$PID" 2>/dev/null || true
        echo "[$(date -Is)] Tail of log:" >&2
        tail -n 40 "${EXP_LOG}" >&2 || true
        exit 1
    fi
done

echo "[$(date -Is)] Waiting for PID=${PID} ..."
wait "$PID" 2>/dev/null
rc=$?

if [[ "$rc" -ne 0 ]]; then
    echo "[$(date -Is)] ERROR: Expander DiCo exited with rc=${rc}. Tail:" >&2
    tail -n 80 "${EXP_LOG}" >&2 || true
    exit "$rc"
fi

if [[ ! -s "${EXP_DIR}/graph_snd_log.csv" ]]; then
    echo "[$(date -Is)] ERROR: ${EXP_DIR}/graph_snd_log.csv missing or empty." >&2
    exit 3
fi

rows=$(wc -l < "${EXP_DIR}/graph_snd_log.csv" | tr -d ' ')
echo "[$(date -Is)] seed=${SEED} done. ${EXP_DIR}/graph_snd_log.csv = ${rows} lines."
