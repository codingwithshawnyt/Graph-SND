#!/usr/bin/env bash
# Graph-SND scaling experiment: overnight n=100 run on a single GPU.
#
# Starts the batched trainer on CUDA device 0 with settings tuned for
# an 8-12 hour window on an RTX 4090: num_envs=64, rollout_steps=128,
# 5000 iterations, checkpoint every 250 iters, SND/Graph-SND logged
# every 50 iters.
#
# Logs go to logs/n100_overnight_<timestamp>.log; the process detaches
# via nohup so you can close the terminal and check on it later.
#
# Usage:
#   bash scripts/run_overnight_n100.sh                # defaults
#   ITERS=10000 SEED=1 bash scripts/run_overnight_n100.sh
#   RESUME=checkpoints/n100_overnight_latest.pt \
#       bash scripts/run_overnight_n100.sh
#
# Environment variables (override defaults):
#   DEVICE          torch device string (default: cuda:0)
#   N_AGENTS        team size (default: 100)
#   ITERS           total PPO iterations (default: 5000)
#   NUM_ENVS        vectorised env count (default: 64)
#   ROLLOUT_STEPS   steps per PPO rollout (default: 128)
#   MINIBATCH       PPO minibatch size (default: 2048)
#   EPOCHS          PPO epochs per update (default: 4)
#   SEED            torch+numpy+python seed (default: 0)
#   CKPT_EVERY      iters between checkpoints (default: 250)
#   SND_EVERY       iters between SND cost logs (default: 50)
#   SND_P           Bernoulli edge probability for the log (default: 0.1)
#   TAG             filename tag (default: overnight)
#   RESUME          path to a training *.pt to resume from (default: none)

set -euo pipefail

cd "$(dirname "$0")/.."

: "${DEVICE:=cuda:0}"
: "${N_AGENTS:=100}"
: "${ITERS:=5000}"
: "${NUM_ENVS:=64}"
: "${ROLLOUT_STEPS:=128}"
: "${MINIBATCH:=2048}"
: "${EPOCHS:=4}"
: "${SEED:=0}"
: "${CKPT_EVERY:=250}"
: "${SND_EVERY:=50}"
: "${SND_P:=0.1}"
: "${TAG:=overnight}"
: "${RESUME:=}"

mkdir -p logs checkpoints results/scaling

stamp="$(date +%Y%m%d_%H%M%S)"
log_path="logs/n${N_AGENTS}_${TAG}_${stamp}.log"

PY_BIN=".venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
    PY_BIN="python"
fi

resume_arg=""
if [ -n "$RESUME" ]; then
    resume_arg="--resume $RESUME"
fi

cmd=(
    "$PY_BIN"
    "training/train_navigation_batched.py"
    "--n-agents" "$N_AGENTS"
    "--iters" "$ITERS"
    "--seed" "$SEED"
    "--num-envs" "$NUM_ENVS"
    "--rollout-steps" "$ROLLOUT_STEPS"
    "--minibatch-size" "$MINIBATCH"
    "--epochs" "$EPOCHS"
    "--device" "$DEVICE"
    "--ckpt-every" "$CKPT_EVERY"
    "--snd-every" "$SND_EVERY"
    "--snd-p" "$SND_P"
    "--tag" "$TAG"
)
if [ -n "$RESUME" ]; then
    cmd+=("--resume" "$RESUME")
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching: ${cmd[*]}" | tee -a "$log_path"
echo "  log       -> $log_path"
echo "  snd csv   -> results/scaling/n${N_AGENTS}_${TAG}_snd_log.csv"
echo "  train csv -> results/scaling/n${N_AGENTS}_${TAG}_train_log.csv"
echo "  ckpts     -> checkpoints/n${N_AGENTS}_${TAG}_*.pt"
echo "Detaching with nohup. Follow progress with:  tail -f $log_path"

nohup "${cmd[@]}" >>"$log_path" 2>&1 &
pid=$!
echo "$pid" > "logs/n${N_AGENTS}_${TAG}.pid"
echo "pid=$pid"
