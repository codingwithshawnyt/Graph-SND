#!/usr/bin/env bash
# Kill-Shot 1: Localised Diversity via Dynamic k-NN Graphs — Dispersion task.
#
# Three IPPO + DiCo runs on riddle's two RTX 4090s.
# GPU assignments are hardcoded (no polling):
# Run A (IPPO baseline, desired_snd=-1): physical GPU 0
# Run C (k-NN Graph-SND, k=3): physical GPU 1 (parallel with A)
# Run B (Full SND DiCo): physical GPU 0 (after A finishes)
#
# Task: VMAS Dispersion (share_rew=True).  Dispersion requires agents
# to spread out and cover food landmarks — local spatial diversity
# pressure (k-NN) should beat global diversity (full SND) because only
# nearby neighbours compete for the same food.
#
# Usage (from repo fork root):
# bash scripts/launch_knn_dico.sh
#
# Override defaults:
# N_AGENTS=10 MAX_ITERS=300 DESIRED_SND=0.1 bash scripts/launch_knn_dico.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
fi

N_AGENTS="${N_AGENTS:-10}"
MAX_ITERS="${MAX_ITERS:-300}"
DESIRED_SND="${DESIRED_SND:-0.1}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
LOGGERS='experiment.loggers=[]'

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)

HYDRA_COMMON=(
    "${LOGGERS}"
    "task.n_agents=${N_AGENTS}"
    "task.n_food=${N_FOOD}"
    "task.share_rew=true"
    "experiment.max_n_iters=${MAX_ITERS}"
    "experiment.render=false"
)

# --- Run A: IPPO baseline (desired_snd=-1 → scaling_ratio=1.0, pure IPPO) ---
echo "[$(date -Is)] Starting IPPO baseline on physical GPU 0 -> logs/knn_ippo_disp_n${N_AGENTS}.log"
CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=-1" \
    "model.diversity_estimator=full" \
    experiment.train_device=cuda:0 \
    experiment.sampling_device=cuda:0 \
    > "logs/knn_ippo_disp_n${N_AGENTS}.log" 2>&1 &
PID_A=$!

# --- Run C: k-NN Graph-SND DiCo (k=3, desired_snd=0.1) on GPU 1 ---
echo "[$(date -Is)] Starting k-NN Graph-SND (k=3) on physical GPU 1 -> logs/knn_dico_knn_disp_n${N_AGENTS}.log"
CUDA_VISIBLE_DEVICES=1 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=knn" \
    "model.diversity_knn_k=3" \
    experiment.train_device=cuda:0 \
    experiment.sampling_device=cuda:0 \
    > "logs/knn_dico_knn_disp_n${N_AGENTS}.log" 2>&1 &
PID_C=$!

echo "[$(date -Is)] Spawned PID_A=$PID_A (GPU 0) PID_C=$PID_C (GPU 1); checking health..."
sleep 5

if ! kill -0 "$PID_A" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: IPPO baseline (PID_A) exited immediately. Tail:" >&2
    tail -n 80 "logs/knn_ippo_disp_n${N_AGENTS}.log" >&2 || true
    exit 1
fi
if ! kill -0 "$PID_C" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: k-NN DiCo (PID_C) exited immediately. Tail:" >&2
    tail -n 80 "logs/knn_dico_knn_disp_n${N_AGENTS}.log" >&2 || true
    exit 1
fi

# --- Wait for Run A to finish, then launch Run B on GPU 0 ---
echo "[$(date -Is)] Waiting for IPPO baseline (PID_A=$PID_A) to finish..."
wait "$PID_A" 2>/dev/null || true
echo "[$(date -Is)] IPPO baseline done. Starting Full SND DiCo on physical GPU 0 -> logs/knn_dico_full_disp_n${N_AGENTS}.log"

CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=full" \
    "model.diversity_p=1.0" \
    experiment.train_device=cuda:0 \
    experiment.sampling_device=cuda:0 \
    > "logs/knn_dico_full_disp_n${N_AGENTS}.log" 2>&1 &
PID_B=$!

echo "[$(date -Is)] Waiting for remaining jobs PID_B=$PID_B PID_C=$PID_C ..."
wait "$PID_B" 2>/dev/null || true
wait "$PID_C" 2>/dev/null || true

echo "[$(date -Is)] All three runs finished. Check logs/ and Hydra outputs/ for CSV paths."
