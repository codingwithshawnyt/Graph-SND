#!/usr/bin/env bash
# Run two DiCo navigation jobs on cuda:0 and cuda:1 in parallel; when the first
# one exits, start the third job on the GPU that became free. Then wait for
# all remaining work to finish.
#
# Usage (from repo fork root):
#   # The *orchestrator* must survive SSH disconnect, or the third job never
#   # starts. Either use tmux, or:
#   nohup bash scripts/run_graph_dico_two_gpus_then_third.sh > logs/orchestrator.log 2>&1 &
#
#   (The three Python runs are also nohup'd, but this shell decides when to
#   launch the third — keep it alive with nohup/tmux.)
#
# Override defaults with env vars, e.g.:
#   N_AGENTS=100 MAX_ITERS=500 DESIRED_SND=0.5 bash scripts/run_graph_dico_two_gpus_then_third.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

N_AGENTS="${N_AGENTS:-16}"
MAX_ITERS="${MAX_ITERS:-300}"
DESIRED_SND="${DESIRED_SND:-0.5}"
LOGGERS='experiment.loggers=[]'

# Hydra: put --config-name before overrides (matches a known-good manual CLI).
RUNNER=(python het_control/run_scripts/run_navigation_ippo.py)
HYDRA_OVERRIDES=(
  "${LOGGERS}"
  "task.n_agents=${N_AGENTS}"
  "experiment.max_n_iters=${MAX_ITERS}"
  "model.desired_snd=${DESIRED_SND}"
)

# Pin one physical GPU per process via CUDA_VISIBLE_DEVICES and always pass
# train/sampling_device=cuda:0 (each process only sees one logical GPU). This
# avoids Hydra/argparse edge cases with dynamic cuda:N overrides on the third
# job and matches common cluster practice.
echo "[$(date -Is)] Starting full SND on physical GPU 0 -> logs/dico_full_n${N_AGENTS}.log"
CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
  --config-name navigation_ippo_full_config \
  "${HYDRA_OVERRIDES[@]}" \
  experiment.train_device=cuda:0 \
  experiment.sampling_device=cuda:0 \
  > "logs/dico_full_n${N_AGENTS}.log" 2>&1 &
P0=$!

echo "[$(date -Is)] Starting graph p=0.25 on physical GPU 1 -> logs/dico_graph_p025_n${N_AGENTS}.log"
CUDA_VISIBLE_DEVICES=1 nohup "${RUNNER[@]}" \
  --config-name navigation_ippo_graph_p025_config \
  "${HYDRA_OVERRIDES[@]}" \
  experiment.train_device=cuda:0 \
  experiment.sampling_device=cuda:0 \
  > "logs/dico_graph_p025_n${N_AGENTS}.log" 2>&1 &
P1=$!

echo "[$(date -Is)] Spawned P0=$P0 P1=$P1; waiting a few seconds to confirm they stay up..."
sleep 5
if ! kill -0 "$P0" 2>/dev/null; then
  echo "[$(date -Is)] ERROR: full-SND job (P0) exited immediately. Tail logs/dico_full_n${N_AGENTS}.log:" >&2
  tail -n 80 "logs/dico_full_n${N_AGENTS}.log" >&2 || true
  exit 1
fi
if ! kill -0 "$P1" 2>/dev/null; then
  echo "[$(date -Is)] ERROR: graph-p025 job (P1) exited immediately. Tail logs/dico_graph_p025_n${N_AGENTS}.log:" >&2
  tail -n 80 "logs/dico_graph_p025_n${N_AGENTS}.log" >&2 || true
  exit 1
fi

echo "[$(date -Is)] Waiting until one of PIDs {$P0,$P1} finishes..."
while kill -0 "$P0" 2>/dev/null && kill -0 "$P1" 2>/dev/null; do
  sleep 30
done

# Whichever job exited first frees its GPU; start the third job there.
if kill -0 "$P0" 2>/dev/null; then
  # P0 still running => P1 (physical GPU 1) finished first.
  FREE_DEV=1
  echo "[$(date -Is)] First finish: GPU-1 job (PID $P1) exited -> starting third on physical GPU 1"
elif kill -0 "$P1" 2>/dev/null; then
  # P1 still running => P0 (physical GPU 0) finished first.
  FREE_DEV=0
  echo "[$(date -Is)] First finish: GPU-0 job (PID $P0) exited -> starting third on physical GPU 0"
else
  # Both ended in the same window (rare): pick GPU 0 and let the user inspect logs.
  FREE_DEV=0
  echo "[$(date -Is)] Both initial jobs already exited; starting third on physical GPU 0 (check logs)"
fi

echo "[$(date -Is)] Starting graph p=0.1 on physical GPU ${FREE_DEV} -> logs/dico_graph_p01_n${N_AGENTS}.log"
CUDA_VISIBLE_DEVICES="${FREE_DEV}" nohup "${RUNNER[@]}" \
  --config-name navigation_ippo_graph_p01_config \
  "${HYDRA_OVERRIDES[@]}" \
  experiment.train_device=cuda:0 \
  experiment.sampling_device=cuda:0 \
  > "logs/dico_graph_p01_n${N_AGENTS}.log" 2>&1 &
P2=$!

echo "[$(date -Is)] Waiting for remaining jobs P0=$P0 P1=$P1 P2=$P2 ..."
wait "$P0" 2>/dev/null || true
wait "$P1" 2>/dev/null || true
wait "$P2" 2>/dev/null || true

echo "[$(date -Is)] All three runs finished (or failed). Check logs/ and Hydra outputs/ for CSV paths."
