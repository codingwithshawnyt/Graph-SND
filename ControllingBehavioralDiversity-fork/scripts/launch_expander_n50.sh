#!/usr/bin/env bash
# Feasibility run: Expander-GG DiCo on Dispersion at n=50, single seed.
#
# Two runs, one per GPU, in parallel:
#   GPU 0: IPPO baseline (desired_snd=-1)
#   GPU 1: DiCo + expander (d=4, since log(50)≈3.9; 50*4=200 is even)
#
# Single seed is by design: this is a feasibility demonstration consistent
# with the n=50 Bernoulli sweep's three-seed result, not an independent
# statistical claim.
#
# Usage (from repo fork root):
#   bash scripts/launch_expander_n50.sh
#   SEED=1 FORCE=1 bash scripts/launch_expander_n50.sh
#
# Env vars:
#   SEED          (default 0)
#   N_AGENTS      (default 50)
#   MAX_ITERS     (default 167)
#   DESIRED_SND   (default 0.1)
#   RR_D          (default 4)
#   ENV_N         (default 120)
#   FORCE         (default 0)
#   RESULTS_BASE  (default $ROOT/results/neurips_final_n50)

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
N_AGENTS="${N_AGENTS:-50}"
MAX_ITERS="${MAX_ITERS:-167}"
DESIRED_SND="${DESIRED_SND:-0.1}"
RR_D="${RR_D:-4}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
ENV_N="${ENV_N:-120}"
FRAMES_PER_BATCH="${FRAMES_PER_BATCH:-$(( ENV_N * 100 ))}"
FORCE="${FORCE:-0}"
LOGGERS='experiment.loggers=[]'

RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final_n50}"
RESULTS_DIR="${RESULTS_BASE}/seed${SEED}"
IPPO_DIR="${RESULTS_DIR}/ippo"
EXP_DIR="${RESULTS_DIR}/expander"

for d in "$IPPO_DIR" "$EXP_DIR"; do
    if [[ -s "$d/graph_snd_log.csv" && "$FORCE" != "1" ]]; then
        echo "ERROR: $d already contains graph_snd_log.csv." >&2
        echo "       Re-run with FORCE=1 to wipe and replace." >&2
        exit 1
    fi
    if [[ "$FORCE" == "1" ]]; then
        rm -rf "$d"
    fi
    mkdir -p "$d"
done

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)

HYDRA_COMMON=(
    "${LOGGERS}"
    "seed=${SEED}"
    "task.n_agents=${N_AGENTS}"
    "task.n_food=${N_FOOD}"
    "task.share_reward=true"
    "experiment.max_n_iters=${MAX_ITERS}"
    "experiment.on_policy_n_envs_per_worker=${ENV_N}"
    "experiment.on_policy_collected_frames_per_batch=${FRAMES_PER_BATCH}"
    "experiment.render=false"
    "experiment.train_device=cuda:0"
    "experiment.sampling_device=cuda:0"
    "experiment.buffer_device=cpu"
)

IPPO_LOG="logs/neurips_n50_seed${SEED}_ippo.log"
EXP_LOG="logs/neurips_n50_seed${SEED}_expander_d${RR_D}.log"

echo "[$(date -Is)] n=${N_AGENTS} seed=${SEED} iters=${MAX_ITERS} envs/worker=${ENV_N} -> ${RESULTS_DIR}"

echo "[$(date -Is)] GPU 0: IPPO baseline (desired_snd=-1) -> ${IPPO_DIR}"
CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=-1" \
    "model.diversity_estimator=full" \
    "hydra.run.dir=${IPPO_DIR}" \
    > "${IPPO_LOG}" 2>&1 &
PID_IPPO=$!

echo "[$(date -Is)] GPU 1: Expander DiCo (d=${RR_D}) -> ${EXP_DIR}"
CUDA_VISIBLE_DEVICES=1 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=expander" \
    "model.diversity_expander_d=${RR_D}" \
    "hydra.run.dir=${EXP_DIR}" \
    > "${EXP_LOG}" 2>&1 &
PID_EXP=$!

echo "[$(date -Is)] Spawned PID_IPPO=${PID_IPPO} (GPU 0), PID_EXP=${PID_EXP} (GPU 1); health check in 15 s..."
sleep 15

if ! kill -0 "$PID_IPPO" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: IPPO baseline exited immediately. Tail:" >&2
    tail -n 80 "${IPPO_LOG}" >&2 || true
fi
if ! kill -0 "$PID_EXP" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: Expander DiCo exited immediately. Tail:" >&2
    tail -n 80 "${EXP_LOG}" >&2 || true
fi

# --- Auto-abort for expander: kill if iter 5 not reached within 4 hours ---
ABORT_TIMEOUT_SEC=14400
POLL_INTERVAL_SEC=300
MIN_ITERS=5
ELAPSED=0
ABORT_TRIGGERED=0

while kill -0 "$PID_EXP" 2>/dev/null; do
    sleep "${POLL_INTERVAL_SEC}"
    ELAPSED=$((ELAPSED + POLL_INTERVAL_SEC))

    CSV="${EXP_DIR}/graph_snd_log.csv"
    if [[ -s "$CSV" ]]; then
        ROWS=$(wc -l < "$CSV" | tr -d ' ')
        ITERS=$((ROWS - 1))
        if [[ "$ITERS" -ge "$MIN_ITERS" ]]; then
            echo "[$(date -Is)] Expander reached iter ${ITERS} >= ${MIN_ITERS}; auto-abort disabled."
            break
        fi
    fi

    if [[ "$ELAPSED" -ge "$ABORT_TIMEOUT_SEC" ]]; then
        echo "[$(date -Is)] AUTO-ABORT: expander iter ${MIN_ITERS} not reached within $((ABORT_TIMEOUT_SEC / 3600))h. Killing PID ${PID_EXP}." >&2
        kill "$PID_EXP" 2>/dev/null || true
        sleep 2
        kill -9 "$PID_EXP" 2>/dev/null || true
        ABORT_TRIGGERED=1
        break
    fi
done

echo "[$(date -Is)] Waiting on PID_IPPO=${PID_IPPO} PID_EXP=${PID_EXP} ..."
wait "$PID_IPPO" 2>/dev/null
rc_ippo=$?
wait "$PID_EXP" 2>/dev/null
rc_exp=$?

echo
echo "[$(date -Is)] n=${N_AGENTS} seed=${SEED} exit codes: IPPO=${rc_ippo}, Expander=${rc_exp}"
echo "Artefacts:"
for tag in ippo expander; do
    csv="${RESULTS_DIR}/${tag}/graph_snd_log.csv"
    if [[ -s "$csv" ]]; then
        rows=$(wc -l < "$csv" | tr -d ' ')
        echo "  ${tag}: ${csv} (${rows} rows)"
    else
        echo "  ${tag}: ${csv} MISSING OR EMPTY (check logs/)"
    fi
done

if [[ "$ABORT_TRIGGERED" -eq 1 || "$rc_ippo" -ne 0 || "$rc_exp" -ne 0 ]]; then
    exit 1
fi
exit 0
