#!/usr/bin/env bash
# Multi-seed driver for launch_expander_dico.sh.
#
# Runs a sequence of seeds sequentially. Each seed takes ~30–40 min on a
# single RTX 4090 at n=10, so SEEDS="0 1 2" ≈ 2 h end-to-end.
#
# Usage (from repo fork root):
#   SEEDS="0 1 2" bash scripts/launch_expander_dico_seeds.sh
#   SEEDS="1 2" GPU=1 bash scripts/launch_expander_dico_seeds.sh
#
# Forwarded env vars (see launch_expander_dico.sh):
#   FORCE, GPU, N_AGENTS, MAX_ITERS, DESIRED_SND, RR_D, N_FOOD, RESULTS_BASE

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS_SPEC="${SEEDS:-0 1 2}"
read -r -a SEEDS_ARR <<< "$SEEDS_SPEC"

if [[ ${#SEEDS_ARR[@]} -eq 0 ]]; then
    echo "ERROR: SEEDS is empty. Provide e.g. SEEDS=\"0 1 2\"." >&2
    exit 1
fi

FAILED=()

for s in "${SEEDS_ARR[@]}"; do
    echo
    echo "============================================================"
    echo "[$(date -Is)] Launching expander seed ${s} (of: ${SEEDS_SPEC})"
    echo "============================================================"
    if SEED="$s" bash scripts/launch_expander_dico.sh; then
        echo "[$(date -Is)] seed ${s} completed."
    else
        rc=$?
        echo "[$(date -Is)] seed ${s} FAILED (exit ${rc}); continuing with remaining seeds." >&2
        FAILED+=("$s")
    fi
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[$(date -Is)] DONE with failures on seeds: ${FAILED[*]}" >&2
    exit 1
fi

echo "[$(date -Is)] All seeds completed: ${SEEDS_SPEC}"
echo
echo "Expected layout:"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final}"
for s in "${SEEDS_ARR[@]}"; do
    echo "  ${RESULTS_BASE}/seed${s}/expander/graph_snd_log.csv"
done
