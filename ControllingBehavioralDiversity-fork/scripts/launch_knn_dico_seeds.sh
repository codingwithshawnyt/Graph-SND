#!/usr/bin/env bash
# Multi-seed driver for launch_knn_dico.sh.
#
# Runs a sequence of seeds sequentially on riddle's two RTX 4090s; each seed
# triple (IPPO baseline + k-NN Graph-SND + Full SND DiCo on VMAS Dispersion,
# n=10, desired_snd=0.1, 167 iters) takes ~90 min wall clock
# (IPPO ~33 min on GPU 0 in parallel with k-NN ~33 min on GPU 1, then Full ~56 min
# on GPU 0 serially). SEEDS="0 1 2" => ~4.5 h end-to-end.
#
# Usage (from repo fork root):
#   SEEDS="0 1 2" bash scripts/launch_knn_dico_seeds.sh
#   # or skip seed 0 if you kept the existing single-seed run:
#   SEEDS="1 2" bash scripts/launch_knn_dico_seeds.sh
#
# Forwarded env vars (see launch_knn_dico.sh for details):
#   FORCE, N_AGENTS, MAX_ITERS, DESIRED_SND, N_FOOD, KNN_SUBSAMPLE_ENVS,
#   RESULTS_BASE
#
# Exits non-zero if any seed fails but still attempts the remaining seeds so
# a transient failure on one seed does not nuke the whole overnight run.

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
    echo "[$(date -Is)] Launching seed ${s} (of: ${SEEDS_SPEC})"
    echo "============================================================"
    if SEED="$s" bash scripts/launch_knn_dico.sh; then
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
    for v in ippo knn full; do
        echo "  ${RESULTS_BASE}/seed${s}/${v}/graph_snd_log.csv"
    done
done
