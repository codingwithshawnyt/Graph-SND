#!/usr/bin/env bash
# NeurIPS final Dispersion runs.
#
# Launches three IPPO runs on riddle's two RTX 4090s and pins each run's
# Hydra output directory so the CSV lands exactly where the paper pipeline
# expects it:
#
#   results/neurips_final/ippo/graph_snd_log.csv   <- IPPO baseline (desired_snd=-1)
#   results/neurips_final/knn/graph_snd_log.csv    <- k-NN Graph-SND (k=3, desired_snd=0.1)
#   results/neurips_final/full/graph_snd_log.csv   <- Full SND DiCo (desired_snd=0.1)
#
# GPU assignments (hardcoded, no polling):
#   Run A (IPPO baseline)     -> physical GPU 0
#   Run C (k-NN Graph-SND)    -> physical GPU 1 (parallel with A)
#   Run B (Full SND DiCo)     -> physical GPU 0 (after A finishes)
#
# Task: VMAS Dispersion. Dispersion requires agents to spread out and cover
# food landmarks -- local spatial diversity pressure (k-NN) should beat
# global diversity (full SND) because only nearby neighbours compete for
# the same food. This is also the DiCo-validated task regime per
# DIAGNOSIS.md Postmortem #2.
#
# Usage (from repo fork root):
#   bash scripts/launch_knn_dico.sh
#
# Override defaults:
#   N_AGENTS=10 MAX_ITERS=167 DESIRED_SND=0.1 bash scripts/launch_knn_dico.sh
#
# Force-clean pre-existing run dirs (otherwise the script refuses to
# overwrite a populated target):
#   FORCE=1 bash scripts/launch_knn_dico.sh

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

N_AGENTS="${N_AGENTS:-10}"
MAX_ITERS="${MAX_ITERS:-167}"
DESIRED_SND="${DESIRED_SND:-0.1}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
FORCE="${FORCE:-0}"
LOGGERS='experiment.loggers=[]'

RESULTS_DIR="${ROOT}/results/neurips_final"
IPPO_DIR="${RESULTS_DIR}/ippo"
KNN_DIR="${RESULTS_DIR}/knn"
FULL_DIR="${RESULTS_DIR}/full"

for d in "$IPPO_DIR" "$KNN_DIR" "$FULL_DIR"; do
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
    "task.n_agents=${N_AGENTS}"
    "task.n_food=${N_FOOD}"
    "task.share_reward=true"
    "experiment.max_n_iters=${MAX_ITERS}"
    "experiment.render=false"
    "experiment.train_device=cuda:0"
    "experiment.sampling_device=cuda:0"
    "experiment.buffer_device=cpu"
)

# --- Run A: IPPO baseline (desired_snd=-1 -> scaling_ratio=1.0, pure IPPO) ---
echo "[$(date -Is)] Starting IPPO baseline on physical GPU 0 -> ${IPPO_DIR}"
CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=-1" \
    "model.diversity_estimator=full" \
    "hydra.run.dir=${IPPO_DIR}" \
    > "logs/neurips_final_ippo.log" 2>&1 &
PID_A=$!

# --- Run C: k-NN Graph-SND DiCo (k=3, desired_snd=0.1) on GPU 1 ---
echo "[$(date -Is)] Starting k-NN Graph-SND (k=3) on physical GPU 1 -> ${KNN_DIR}"
CUDA_VISIBLE_DEVICES=1 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=knn" \
    "model.diversity_knn_k=3" \
    "hydra.run.dir=${KNN_DIR}" \
    > "logs/neurips_final_knn.log" 2>&1 &
PID_C=$!

echo "[$(date -Is)] Spawned PID_A=$PID_A (GPU 0) PID_C=$PID_C (GPU 1); checking health..."
sleep 10

if ! kill -0 "$PID_A" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: IPPO baseline (PID_A) exited immediately. Tail:" >&2
    tail -n 80 "logs/neurips_final_ippo.log" >&2 || true
    exit 1
fi
if ! kill -0 "$PID_C" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: k-NN DiCo (PID_C) exited immediately. Tail:" >&2
    tail -n 80 "logs/neurips_final_knn.log" >&2 || true
    exit 1
fi

# --- Wait for Run A to finish, then launch Run B on GPU 0 ---
echo "[$(date -Is)] Waiting for IPPO baseline (PID_A=$PID_A) to finish..."
wait "$PID_A" 2>/dev/null || true
echo "[$(date -Is)] IPPO baseline done. Starting Full SND DiCo on physical GPU 0 -> ${FULL_DIR}"

CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=full" \
    "model.diversity_p=1.0" \
    "hydra.run.dir=${FULL_DIR}" \
    > "logs/neurips_final_full.log" 2>&1 &
PID_B=$!

echo "[$(date -Is)] Waiting for remaining jobs PID_B=$PID_B PID_C=$PID_C ..."
wait "$PID_B" 2>/dev/null || true
wait "$PID_C" 2>/dev/null || true

echo
echo "[$(date -Is)] All three runs finished. CSVs:"
echo "  IPPO baseline:   ${IPPO_DIR}/graph_snd_log.csv"
echo "  k-NN Graph-SND:  ${KNN_DIR}/graph_snd_log.csv"
echo "  Full SND DiCo:   ${FULL_DIR}/graph_snd_log.csv"
echo
echo "Plot with:"
echo "  python ../scripts/plot_reward_curves.py \\"
echo "      \"IPPO Baseline:${IPPO_DIR}/graph_snd_log.csv\" \\"
echo "      \"k-NN Graph-SND:${KNN_DIR}/graph_snd_log.csv\" \\"
echo "      \"Full SND:${FULL_DIR}/graph_snd_log.csv\" \\"
echo "      --output ${ROOT}/../Paper/figures/neurips_knn_plot.pdf"
