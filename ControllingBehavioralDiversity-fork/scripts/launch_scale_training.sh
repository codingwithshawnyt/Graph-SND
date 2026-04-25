#!/usr/bin/env bash
# Limitation 1 closer: IPPO + passive Graph-SND logging at n=250 (and n=500 stretch).
#
# NOT DiCo — just IPPO with GraphSNDLoggingCallback writing the standard CSV.
# The claim: "Graph-SND remains a stable, low-overhead measurement signal
# during in-loop training at n=250."
#
# Phase 1: n=250, single seed, 200 iters
#   GPU 0: IPPO + full SND passive logging
#   GPU 1: IPPO + Bernoulli-0.1 Graph-SND passive logging
#   Both run desired_snd=-1 (no DiCo control, just measurement).
#
# Phase 2 (stretch, auto if Phase 1 succeeds): n=500, single seed, 100 iters
#   Same layout: GPU 0 = full, GPU 1 = Bernoulli-0.1
#   Even 50-100 iters with stable tracking is enough for a feasibility sentence.
#
# Memory scaling (from empirical data):
#   n=10:  ENV_N=600 (default)
#   n=50:  ENV_N=120 (5x reduction)
#   n=100: ENV_N=60  (10x reduction, from overnight run)
#   n=250: ENV_N=24  (25x reduction, conservative)
#   n=500: ENV_N=10  (60x reduction, very conservative)
#
# Usage (from fork root, in tmux):
#   bash scripts/launch_scale_training.sh
#   FORCE=1 bash scripts/launch_scale_training.sh
#   SKIP_N500=1 bash scripts/launch_scale_training.sh   # n=250 only
#
# Env vars (all optional):
#   SEED            (default 0)
#   N250_ITERS      (default 200)
#   N500_ITERS      (default 100)
#   N250_ENV_N      (default 12)
#   N500_ENV_N      (default 6)
#   ONP_MINIBATCH_ITERS (default 5)    — lower than config default 45 for scale runs
#   ONP_MINIBATCH_SIZE  (default 2048) — lower than config default 4096 for scale runs
#   CUDA_ALLOC_CONF     (default expandable_segments:True)
#   SKIP_N500       (default 0)   — set 1 to skip the n=500 stretch
#   FORCE           (default 0)
#   RESULTS_BASE    (default $ROOT/results/scale_training)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

if [[ -f "$ROOT/../.venv/bin/activate" ]]; then
    source "$ROOT/../.venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

SEED="${SEED:-0}"
N250_ITERS="${N250_ITERS:-200}"
N500_ITERS="${N500_ITERS:-100}"
N250_ENV_N="${N250_ENV_N:-12}"
N500_ENV_N="${N500_ENV_N:-6}"
ONP_MINIBATCH_ITERS="${ONP_MINIBATCH_ITERS:-5}"
ONP_MINIBATCH_SIZE="${ONP_MINIBATCH_SIZE:-2048}"
CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True}"
SKIP_N500="${SKIP_N500:-0}"
FORCE="${FORCE:-0}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/scale_training}"
LOGGERS='experiment.loggers=[]'

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)

# ---------------------------------------------------------------------------
# Helper: run one (n, estimator, GPU) cell
# ---------------------------------------------------------------------------
run_cell() {
    local N_AGENTS="$1"
    local ESTIMATOR="$2"  # "full" or "graph_p01"
    local GPU="$3"
    local MAX_ITERS="$4"
    local ENV_N="$5"
    local TAG="$6"         # output subdirectory name

    local N_FOOD="${N_AGENTS}"
    local FRAMES_PER_BATCH=$(( ENV_N * 100 ))
    local OUT_DIR="${RESULTS_BASE}/n${N_AGENTS}/seed${SEED}/${TAG}"
    local LOG="${ROOT}/logs/scale_n${N_AGENTS}_seed${SEED}_${TAG}.log"

    if [[ -s "${OUT_DIR}/graph_snd_log.csv" && "$FORCE" != "1" ]]; then
        echo "[$(date -Is)] n=${N_AGENTS} ${TAG}: CSV exists, skipping (FORCE=1 to overwrite)" >&2
        RUN_CELL_PID="SKIP"
        return 0
    fi
    if [[ "$FORCE" == "1" ]]; then
        rm -rf "$OUT_DIR"
    fi
    mkdir -p "$OUT_DIR"

    local P_VAL="1.0"
    local EXTRA_OVERRIDES=()
    if [[ "$ESTIMATOR" == "graph_p01" ]]; then
        P_VAL="0.1"
        EXTRA_OVERRIDES=("model.diversity_p=0.1")
    fi

    echo "[$(date -Is)] n=${N_AGENTS} ${TAG} (${ESTIMATOR}): GPU ${GPU}, ${MAX_ITERS} iters, ENV_N=${ENV_N} -> ${OUT_DIR}" >&2

    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTORCH_CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF}" \
    nohup "${RUNNER[@]}" \
        --config-name dispersion_ippo_knn_config \
        "${LOGGERS}" \
        "seed=${SEED}" \
        "task.n_agents=${N_AGENTS}" \
        "task.n_food=${N_FOOD}" \
        "task.share_reward=true" \
        "experiment.max_n_iters=${MAX_ITERS}" \
        "experiment.on_policy_n_envs_per_worker=${ENV_N}" \
        "experiment.on_policy_collected_frames_per_batch=${FRAMES_PER_BATCH}" \
        "experiment.on_policy_n_minibatch_iters=${ONP_MINIBATCH_ITERS}" \
        "experiment.on_policy_minibatch_size=${ONP_MINIBATCH_SIZE}" \
        "experiment.render=false" \
        "experiment.train_device=cuda:0" \
        "experiment.sampling_device=cuda:0" \
        "experiment.buffer_device=cpu" \
        "model.desired_snd=-1" \
        "model.diversity_estimator=${ESTIMATOR}" \
        "${EXTRA_OVERRIDES[@]}" \
        "hydra.run.dir=${OUT_DIR}" \
        > "${LOG}" 2>&1 &

    RUN_CELL_PID="$!"
    return 0
}

# ---------------------------------------------------------------------------
# Helper: wait for two PIDs with health check
# ---------------------------------------------------------------------------
wait_pair() {
    local PID0="$1"
    local PID1="$2"
    local LOG0="$3"
    local LOG1="$4"
    local LABEL="$5"

    if [[ "$PID0" == "SKIP" && "$PID1" == "SKIP" ]]; then
        echo "[$(date -Is)] ${LABEL}: both jobs skipped."
        return 0
    fi
    if [[ "$PID0" == "SKIP" || "$PID1" == "SKIP" ]]; then
        echo "[$(date -Is)] ${LABEL}: inconsistent skip state (one skipped, one not). Use FORCE=1 or clear outputs." >&2
        return 1
    fi
    if ! [[ "$PID0" =~ ^[0-9]+$ && "$PID1" =~ ^[0-9]+$ ]]; then
        echo "[$(date -Is)] ${LABEL}: invalid PID(s): PID0='${PID0}' PID1='${PID1}'" >&2
        return 1
    fi

    echo "[$(date -Is)] ${LABEL}: spawned PID ${PID0} (GPU 0), PID ${PID1} (GPU 1). Health check in 20s..."
    sleep 20

    if ! kill -0 "$PID0" 2>/dev/null; then
        echo "[$(date -Is)] ERROR: ${LABEL} GPU 0 (PID ${PID0}) exited immediately. Tail:" >&2
        tail -n 60 "${LOG0}" >&2 || true
    fi
    if ! kill -0 "$PID1" 2>/dev/null; then
        echo "[$(date -Is)] ERROR: ${LABEL} GPU 1 (PID ${PID1}) exited immediately. Tail:" >&2
        tail -n 60 "${LOG1}" >&2 || true
    fi

    echo "[$(date -Is)] ${LABEL}: waiting for both jobs..."
    local rc0=0 rc1=0
    wait "$PID0" 2>/dev/null || rc0=$?
    wait "$PID1" 2>/dev/null || rc1=$?

    echo "[$(date -Is)] ${LABEL}: exit codes GPU0=${rc0}, GPU1=${rc1}"

    # Report CSVs
    for tag_dir in "$5"_*; do
        : # handled by caller
    done

    if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
        return 1
    fi
    return 0
}

report_csvs() {
    local N_AGENTS="$1"
    echo "  Artefacts for n=${N_AGENTS}:"
    for tag in full bern_p01; do
        local csv="${RESULTS_BASE}/n${N_AGENTS}/seed${SEED}/${tag}/graph_snd_log.csv"
        if [[ -s "$csv" ]]; then
            local rows
            rows=$(wc -l < "$csv" | tr -d ' ')
            echo "    ${tag}: ${csv} (${rows} rows)"
        else
            echo "    ${tag}: ${csv} MISSING OR EMPTY"
        fi
    done
}

# ===================================================================
echo "============================================================"
echo "[$(date -Is)] SCALE TRAINING: IPPO + passive Graph-SND logging"
echo "  Seed: ${SEED}"
echo "  Phase 1: n=250, ${N250_ITERS} iters, ENV_N=${N250_ENV_N}"
echo "  Phase 2: n=500, ${N500_ITERS} iters, ENV_N=${N500_ENV_N} (skip=${SKIP_N500})"
echo "  PPO inner loop: minibatch_iters=${ONP_MINIBATCH_ITERS}, minibatch_size=${ONP_MINIBATCH_SIZE}"
echo "  CUDA alloc conf: ${CUDA_ALLOC_CONF}"
echo "  Results: ${RESULTS_BASE}"
echo "============================================================"

# ===================================================================
# PHASE 1: n=250
# ===================================================================
echo
echo "[$(date -Is)] ===== PHASE 1: n=250 ====="

run_cell 250 "full" 0 "${N250_ITERS}" "${N250_ENV_N}" "full"
PID_FULL="$RUN_CELL_PID"
run_cell 250 "graph_p01" 1 "${N250_ITERS}" "${N250_ENV_N}" "bern_p01"
PID_BERN="$RUN_CELL_PID"

LOG_FULL="${ROOT}/logs/scale_n250_seed${SEED}_full.log"
LOG_BERN="${ROOT}/logs/scale_n250_seed${SEED}_bern_p01.log"

wait_pair "$PID_FULL" "$PID_BERN" "$LOG_FULL" "$LOG_BERN" "n=250"
RC_250=$?

report_csvs 250

if [[ "$RC_250" -ne 0 ]]; then
    echo
    echo "[$(date -Is)] PHASE 1 (n=250) had failures. Check logs above." >&2
    echo "[$(date -Is)] Continuing to Phase 2 anyway (stretch goal)..." >&2
fi

# ===================================================================
# PHASE 2: n=500 (stretch)
# ===================================================================
if [[ "$SKIP_N500" == "1" ]]; then
    echo
    echo "[$(date -Is)] PHASE 2 (n=500) skipped (SKIP_N500=1)."
else
    echo
    echo "[$(date -Is)] ===== PHASE 2: n=500 (stretch) ====="

    run_cell 500 "full" 0 "${N500_ITERS}" "${N500_ENV_N}" "full"
    PID_FULL_500="$RUN_CELL_PID"
    run_cell 500 "graph_p01" 1 "${N500_ITERS}" "${N500_ENV_N}" "bern_p01"
    PID_BERN_500="$RUN_CELL_PID"

    LOG_FULL_500="${ROOT}/logs/scale_n500_seed${SEED}_full.log"
    LOG_BERN_500="${ROOT}/logs/scale_n500_seed${SEED}_bern_p01.log"

    wait_pair "$PID_FULL_500" "$PID_BERN_500" "$LOG_FULL_500" "$LOG_BERN_500" "n=500"
    RC_500=$?

    report_csvs 500

    if [[ "$RC_500" -ne 0 ]]; then
        echo "[$(date -Is)] PHASE 2 (n=500) had failures. This is a stretch goal — partial data is still useful." >&2
    fi
fi

# ===================================================================
# SUMMARY
# ===================================================================
echo
echo "============================================================"
echo "[$(date -Is)] SCALE TRAINING COMPLETE"
echo "============================================================"
echo
report_csvs 250
if [[ "$SKIP_N500" != "1" ]]; then
    report_csvs 500
fi
echo
echo "Monitor progress:"
echo "  tail -f ${ROOT}/logs/scale_n250_seed${SEED}_full.log"
echo "  tail -f ${ROOT}/logs/scale_n250_seed${SEED}_bern_p01.log"
echo "  wc -l ${RESULTS_BASE}/n250/seed${SEED}/*/graph_snd_log.csv"
