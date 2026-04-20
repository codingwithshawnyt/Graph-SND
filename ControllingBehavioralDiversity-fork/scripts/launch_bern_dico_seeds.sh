#!/usr/bin/env bash
# Multi-seed driver for launch_bern_dico.sh.
#
# Runs one Bernoulli-p Graph-SND + DiCo run per seed on VMAS Dispersion
# (n=10, desired_snd=0.1, 167 iters). Each single-seed run takes ~33 min
# on a single RTX 4090.
#
# Parallelizes across two GPUs: seeds are processed in waves of NUM_GPUS.
# With SEEDS="0 1 2" and NUM_GPUS=2, wall-clock is ~66 min (seeds 0+1 in
# parallel on GPU 0 + GPU 1, then seed 2 on GPU 0).
#
# Usage (from repo fork root):
#   SEEDS="0 1 2" bash scripts/launch_bern_dico_seeds.sh
#   SEEDS="0 1"   NUM_GPUS=2 bash scripts/launch_bern_dico_seeds.sh
#   SEEDS="0 1 2" NUM_GPUS=1 bash scripts/launch_bern_dico_seeds.sh   # all sequential on GPU 0
#   P=0.25 SEEDS="0 1 2" bash scripts/launch_bern_dico_seeds.sh        # Bernoulli p=0.25 instead
#
# Forwarded env vars (see launch_bern_dico.sh):
#   FORCE, P, N_AGENTS, MAX_ITERS, DESIRED_SND, N_FOOD, RESULTS_BASE
#
# Exits non-zero if any seed fails, but still attempts the remaining
# seeds so a transient failure on one does not nuke the overnight run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS_SPEC="${SEEDS:-0 1 2}"
NUM_GPUS="${NUM_GPUS:-2}"

read -r -a SEEDS_ARR <<< "$SEEDS_SPEC"

if [[ ${#SEEDS_ARR[@]} -eq 0 ]]; then
    echo "ERROR: SEEDS is empty. Provide e.g. SEEDS=\"0 1 2\"." >&2
    exit 1
fi
if (( NUM_GPUS < 1 )); then
    echo "ERROR: NUM_GPUS must be >= 1 (got ${NUM_GPUS})." >&2
    exit 1
fi

FAILED=()

echo "[$(date -Is)] SEEDS=${SEEDS_SPEC} NUM_GPUS=${NUM_GPUS} P=${P:-0.1}"

i=0
wave=0
total=${#SEEDS_ARR[@]}
while (( i < total )); do
    wave=$((wave + 1))
    pids=()
    labels=()
    # Launch up to NUM_GPUS seeds in parallel, one per GPU.
    for (( g=0; g<NUM_GPUS && i<total; g++, i++ )); do
        s="${SEEDS_ARR[i]}"
        echo
        echo "[$(date -Is)] wave ${wave}: seed=${s} -> GPU ${g}"
        SEED="$s" GPU="$g" bash scripts/launch_bern_dico.sh &
        pids+=("$!")
        labels+=("seed=${s}/gpu=${g}")
    done
    # Wait for all seeds in this wave and collect exit codes.
    for (( k=0; k<${#pids[@]}; k++ )); do
        pid="${pids[k]}"
        label="${labels[k]}"
        if wait "$pid"; then
            echo "[$(date -Is)] ${label} completed (PID=${pid})."
        else
            rc=$?
            echo "[$(date -Is)] ${label} FAILED (PID=${pid}, rc=${rc})." >&2
            FAILED+=("${label}")
        fi
    done
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[$(date -Is)] DONE with failures on: ${FAILED[*]}" >&2
    exit 1
fi

echo "[$(date -Is)] All seeds completed: ${SEEDS_SPEC}"
echo
echo "Expected layout:"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final}"
for s in "${SEEDS_ARR[@]}"; do
    echo "  ${RESULTS_BASE}/seed${s}/bern/graph_snd_log.csv"
done

echo
echo "To regenerate the 4-line Dispersion DiCo figure (IPPO / Full / k-NN / Bernoulli) on your Mac:"
echo "  cd <local repo>"
echo "  IPPO_CSVS=''; KNN_CSVS=''; FULL_CSVS=''; BERN_CSVS=''"
echo "  for s in 0 1 2; do"
echo "    IPPO_CSVS+=\"ControllingBehavioralDiversity-fork/results/neurips_final/seed\${s}/ippo/graph_snd_log.csv,\""
echo "    KNN_CSVS+=\"ControllingBehavioralDiversity-fork/results/neurips_final/seed\${s}/knn/graph_snd_log.csv,\""
echo "    FULL_CSVS+=\"ControllingBehavioralDiversity-fork/results/neurips_final/seed\${s}/full/graph_snd_log.csv,\""
echo "    BERN_CSVS+=\"ControllingBehavioralDiversity-fork/results/neurips_final/seed\${s}/bern/graph_snd_log.csv,\""
echo "  done"
echo "  IPPO_CSVS=\${IPPO_CSVS%,}; KNN_CSVS=\${KNN_CSVS%,}; FULL_CSVS=\${FULL_CSVS%,}; BERN_CSVS=\${BERN_CSVS%,}"
echo "  python scripts/plot_reward_curves.py \\"
echo "      --figure-type panels --smooth 5 \\"
echo "      \"ippo:\${IPPO_CSVS}\" \"knn:\${KNN_CSVS}\" \"full:\${FULL_CSVS}\" \"graph_p01:\${BERN_CSVS}\" \\"
echo "      --output Paper/figures/neurips_knn_plot.pdf"
