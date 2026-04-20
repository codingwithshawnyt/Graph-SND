#!/usr/bin/env bash
# Graph-SND scaling experiment: frozen-init n=500 timing sweep.
#
# Runs experiments/exp2_timing_scaling.py on a single CUDA device with
# the paper's GPU frozen-init timing methodology (torch.cuda.synchronize on both
# sides of each time.perf_counter measurement). No training, no
# checkpoints; just the one-shot timing comparison of full SND vs
# Graph-SND at p in {0.1, 0.25, 0.5, 0.75, 1.0} on frozen-init policies.
#
# Logs go to logs/n${N_AGENTS}_timing_<timestamp>.log; the CSV lands at
# results/exp2/timing_n${N_AGENTS}.csv with a matching summary.json. The
# process detaches via nohup so you can close the terminal.
#
# Usage:
#   bash scripts/run_n500_timing.sh                             # defaults (n=500)
#   N_AGENTS="100 500" bash scripts/run_n500_timing.sh          # multi-n sweep
#   TRIALS=10 DEVICE=cuda:1 bash scripts/run_n500_timing.sh
#
# Environment variables (override defaults):
#   DEVICE          torch device string (default: cuda:0)
#   N_AGENTS        team size, space- or comma-separated (default: 500)
#   NUM_ENVS        vectorised env count for synthetic rollouts (default: 32)
#   ROLLOUT_STEPS   steps per synthetic rollout (default: 128)
#   OBS_DIM         synthetic observation dim (default: 18; matches VMAS nav)
#   ACT_DIM         synthetic action dim (default: 2)
#   P_VALUES        Bernoulli inclusion probabilities (default: 0.1,0.25,0.5,0.75,1.0)
#   TRIALS          timing trials per cell (default: 20)
#   WARMUP          warmup trials before the timed block (default: 3)
#   SEED            torch+numpy+python seed (default: 42)
#   DTYPE           float32 | float16 | bfloat16 | float64 (default: float32)
#   TAG             filename tag (default: none; appended to outputs if set)

set -euo pipefail

cd "$(dirname "$0")/.."

: "${DEVICE:=cuda:0}"
: "${N_AGENTS:=500}"
: "${NUM_ENVS:=32}"
: "${ROLLOUT_STEPS:=128}"
: "${OBS_DIM:=18}"
: "${ACT_DIM:=2}"
: "${P_VALUES:=0.1,0.25,0.5,0.75,1.0}"
: "${TRIALS:=20}"
: "${WARMUP:=3}"
: "${SEED:=42}"
: "${DTYPE:=float32}"
: "${TAG:=}"

mkdir -p logs results/exp2

stamp="$(date +%Y%m%d_%H%M%S)"

n_slug="$(echo "$N_AGENTS" | tr ' ,' '__' | sed 's/^_*//;s/_*$//')"
name_base="n${n_slug}"
if [ -n "$TAG" ]; then
    name_base="${name_base}_${TAG}"
fi
log_path="logs/${name_base}_timing_${stamp}.log"
csv_path="results/exp2/timing_${name_base}.csv"

PY_BIN=".venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
    PY_BIN="python"
fi

cmd=(
    "$PY_BIN"
    "experiments/exp2_timing_scaling.py"
    "--n-agents" "$N_AGENTS"
    "--num-envs" "$NUM_ENVS"
    "--rollout-steps" "$ROLLOUT_STEPS"
    "--obs-dim" "$OBS_DIM"
    "--act-dim" "$ACT_DIM"
    "--p-values" "$P_VALUES"
    "--timing-trials" "$TRIALS"
    "--warmup-trials" "$WARMUP"
    "--seed" "$SEED"
    "--device" "$DEVICE"
    "--dtype" "$DTYPE"
    "--out" "$csv_path"
)

# Append *all* launcher lines to the log so `tail -f $log_path` works
# even when the caller wraps this script in `nohup ... >/dev/null 2>&1`
# (stdout from lines below would otherwise be discarded).
{
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching: ${cmd[*]}"
    echo "  device  -> $DEVICE"
    echo "  n       -> $N_AGENTS"
    echo "  trials  -> $TRIALS (warmup=$WARMUP)"
    echo "  p       -> $P_VALUES"
    echo "  csv     -> $csv_path"
    echo "  summary -> ${csv_path%.csv}.summary.json"
    echo "  log     -> $log_path"
    echo "Detaching with nohup. Follow progress with:  tail -f $log_path"
} >>"$log_path"

nohup "${cmd[@]}" >>"$log_path" 2>&1 &
pid=$!
echo "$pid" > "logs/${name_base}_timing.pid"
echo "pid=$pid"
