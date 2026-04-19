#!/usr/bin/env bash
# Kill-Shot 2: OOM Barrier Profiler.
#
# Runs the profiling script on physical GPU 1 (the GPU not used by
# the k-NN experiments if they are running concurrently).  The script
# systematically probes team sizes [50, 100, 250, 500, 1000] and
# measures VRAM + wall-clock for Full SND vs Graph-SND at p=0.01
# and p=0.1, catching CUDA OOM errors gracefully.
#
# Output: oom_profiling_results.csv in the repo root.
#
# Usage (from repo root):
#   bash scripts/run_oom_profiler.sh
#
# Override GPU:
#   GPU=0 bash scripts/run_oom_profiler.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-1}"

PY_BIN=".venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
    PY_BIN="python"
fi

OUTPUT="${OUTPUT:-oom_profiling_results.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "[$(date -Is)] OOM Barrier Profiler on physical GPU ${GPU}"
echo "[$(date -Is)] output -> ${OUTPUT}"

mkdir -p logs

CUDA_VISIBLE_DEVICES="${GPU}" nohup "${PY_BIN}" scripts/profile_oom_barrier.py \
    --device cuda:0 \
    --output "${OUTPUT}" \
    --batch-size "${BATCH_SIZE}" \
    > "logs/oom_profiler.log" 2>&1 &

PID=$!
echo "$PID" > "logs/oom_profiler.pid"
echo "[$(date -Is)] pid=$PID  log=logs/oom_profiler.log"
echo "[$(date -Is)] Follow with:  tail -f logs/oom_profiler.log"
