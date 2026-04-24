#!/usr/bin/env bash
# Master launcher: expander n=10 (3 seeds) → expander n=50 (1 seed).
#
# Runs everything sequentially so you can fire-and-forget in a single
# tmux session. Uses both GPUs where possible.
#
# Phase 1: n=10, seeds 0/1/2 on GPU 0 (~2h total)
# Phase 2: n=50, IPPO baseline on GPU 0 + expander on GPU 1 (~20h)
#
# Usage (from fork root, in tmux):
#   bash scripts/launch_all_expander.sh
#   FORCE=1 bash scripts/launch_all_expander.sh   # wipe existing runs
#
# Env var overrides (all optional):
#   SEEDS="0 1 2"  N10_GPU=0  FORCE=0  DESIRED_SND=0.1  MAX_ITERS=167

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-0 1 2}"
N10_GPU="${N10_GPU:-0}"
FORCE="${FORCE:-0}"

echo "============================================================"
echo "[$(date -Is)] PHASE 1: Expander n=10, seeds=${SEEDS}, GPU=${N10_GPU}"
echo "============================================================"

export FORCE
if GPU="${N10_GPU}" SEEDS="${SEEDS}" bash scripts/launch_expander_dico_seeds.sh; then
    echo
    echo "[$(date -Is)] PHASE 1 COMPLETE. All n=10 seeds finished."
else
    rc=$?
    echo
    echo "[$(date -Is)] PHASE 1 had failures (exit ${rc}). Continuing to n=50 anyway." >&2
fi

echo
echo "============================================================"
echo "[$(date -Is)] PHASE 2: Expander n=50, single seed (feasibility)"
echo "============================================================"

if bash scripts/launch_expander_n50.sh; then
    echo
    echo "[$(date -Is)] PHASE 2 COMPLETE. n=50 feasibility run finished."
else
    rc=$?
    echo "[$(date -Is)] PHASE 2 FAILED (exit ${rc})." >&2
fi

echo
echo "============================================================"
echo "[$(date -Is)] ALL DONE."
echo "============================================================"
echo
echo "Artefacts:"
RESULTS_N10="${ROOT}/results/neurips_final"
RESULTS_N50="${ROOT}/results/neurips_final_n50"
read -r -a SEEDS_ARR <<< "$SEEDS"
for s in "${SEEDS_ARR[@]}"; do
    echo "  n=10 seed${s}: ${RESULTS_N10}/seed${s}/expander/graph_snd_log.csv"
done
echo "  n=50 seed0 ippo:     ${RESULTS_N50}/seed0/ippo/graph_snd_log.csv"
echo "  n=50 seed0 expander: ${RESULTS_N50}/seed0/expander/graph_snd_log.csv"
