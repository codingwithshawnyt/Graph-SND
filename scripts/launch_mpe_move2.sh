#!/usr/bin/env bash
# Move 2: MPE simple-spread — IPPO training (3 seeds) + measurement panel.
#
# MPE training is CPU-bound (PettingZoo doesn't use GPU), so we run two
# seeds in parallel to use both CPU cores, then the third seed, then the
# measurement panel on the trained checkpoints.
#
# Phase 1: Train seeds 0+1 in parallel, then seed 2
# Phase 2: Run measurement panel on each trained checkpoint
# Phase 3: Run measurement panel on frozen random-init (baseline)
#
# Usage (from repo root, in tmux):
#   bash scripts/launch_mpe_move2.sh
#   FORCE=1 bash scripts/launch_mpe_move2.sh
#
# Env vars:
#   N_AGENTS      (default 10)
#   N_ITERS       (default 500)
#   N_ROLLOUTS    (default 20)
#   SEEDS         (default "0 1 2")
#   TIMEOUT_HOURS (default 24)  — per-seed timeout
#   FORCE         (default 0)
#   RESULTS_BASE  (default results/mpe_simple_spread)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Activate venv if present
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

N_AGENTS="${N_AGENTS:-10}"
N_ITERS="${N_ITERS:-500}"
N_ROLLOUTS="${N_ROLLOUTS:-20}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"
FORCE="${FORCE:-0}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/mpe_simple_spread}"

read -r -a SEEDS_ARR <<< "${SEEDS:-0 1 2}"

mkdir -p logs

echo "============================================================"
echo "[$(date -Is)] MOVE 2: MPE simple-spread, n=${N_AGENTS}"
echo "  Seeds: ${SEEDS_ARR[*]}"
echo "  Iters: ${N_ITERS}, Rollouts/iter: ${N_ROLLOUTS}"
echo "  Timeout: ${TIMEOUT_HOURS}h per seed"
echo "  Results: ${RESULTS_BASE}"
echo "============================================================"

# Check pettingzoo is installed
if ! python -c "from pettingzoo.mpe import simple_spread_v3" 2>/dev/null; then
    echo "ERROR: pettingzoo[mpe] not installed. Run: pip install 'pettingzoo[mpe]'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: run one training seed
# ---------------------------------------------------------------------------
run_seed() {
    local SEED="$1"
    local OUT_DIR="${RESULTS_BASE}/seed${SEED}"
    local LOG="${ROOT}/logs/mpe_seed${SEED}.log"
    local CSV="${OUT_DIR}/mpe_training_log.csv"

    if [[ -s "${OUT_DIR}/checkpoint_seed${SEED}.pt" && "$FORCE" != "1" ]]; then
        echo "[$(date -Is)] seed ${SEED}: checkpoint exists, skipping (use FORCE=1 to overwrite)"
        return 0
    fi
    if [[ "$FORCE" == "1" ]]; then
        rm -rf "$OUT_DIR"
    fi
    mkdir -p "$OUT_DIR"

    echo "[$(date -Is)] seed ${SEED}: starting training -> ${OUT_DIR}"
    python experiments/mpe_ippo_training.py \
        --n-agents "${N_AGENTS}" \
        --n-iters "${N_ITERS}" \
        --n-rollouts "${N_ROLLOUTS}" \
        --seed "${SEED}" \
        --output-dir "${OUT_DIR}" \
        --timeout-hours "${TIMEOUT_HOURS}" \
        > "${LOG}" 2>&1
    local rc=$?

    if [[ "$rc" -ne 0 ]]; then
        echo "[$(date -Is)] seed ${SEED}: FAILED (rc=${rc}). Tail:" >&2
        tail -n 40 "${LOG}" >&2 || true
        return "$rc"
    fi

    if [[ ! -s "${OUT_DIR}/checkpoint_seed${SEED}.pt" ]]; then
        echo "[$(date -Is)] seed ${SEED}: WARNING — no checkpoint saved." >&2
        return 1
    fi

    local rows
    rows=$(wc -l < "${CSV}" 2>/dev/null | tr -d ' ')
    echo "[$(date -Is)] seed ${SEED}: done. CSV=${rows} rows, checkpoint saved."
    return 0
}

# ---------------------------------------------------------------------------
# Phase 1: Train seeds — run first two in parallel, then third
# ---------------------------------------------------------------------------
echo
echo "[$(date -Is)] PHASE 1: IPPO training (no DiCo)"

FAILED=()

if [[ ${#SEEDS_ARR[@]} -ge 2 ]]; then
    echo "[$(date -Is)] Running seeds ${SEEDS_ARR[0]} and ${SEEDS_ARR[1]} in parallel..."
    run_seed "${SEEDS_ARR[0]}" &
    PID0=$!
    run_seed "${SEEDS_ARR[1]}" &
    PID1=$!

    wait "$PID0" || FAILED+=("${SEEDS_ARR[0]}")
    wait "$PID1" || FAILED+=("${SEEDS_ARR[1]}")

    # Run remaining seeds sequentially
    for ((i=2; i<${#SEEDS_ARR[@]}; i++)); do
        run_seed "${SEEDS_ARR[$i]}" || FAILED+=("${SEEDS_ARR[$i]}")
    done
else
    for s in "${SEEDS_ARR[@]}"; do
        run_seed "$s" || FAILED+=("$s")
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[$(date -Is)] PHASE 1: failures on seeds: ${FAILED[*]}" >&2
else
    echo "[$(date -Is)] PHASE 1: all seeds completed."
fi

# ---------------------------------------------------------------------------
# Phase 2: Measurement panel on trained checkpoints
# ---------------------------------------------------------------------------
echo
echo "[$(date -Is)] PHASE 2: Measurement panel (trained policies)"

for s in "${SEEDS_ARR[@]}"; do
    CKPT="${RESULTS_BASE}/seed${s}/checkpoint_seed${s}.pt"
    OUT_CSV="${RESULTS_BASE}/seed${s}/measurement_panel.csv"

    if [[ ! -f "$CKPT" ]]; then
        echo "[$(date -Is)] seed ${s}: no checkpoint, skipping measurement."
        continue
    fi

    echo "[$(date -Is)] seed ${s}: running measurement panel..."
    python experiments/mpe_measurement_panel.py \
        --n-agents "${N_AGENTS}" \
        --n-draws 2000 \
        --seed "${s}" \
        --policy-checkpoint "${CKPT}" \
        --output-csv "${OUT_CSV}" \
        2>&1 | tail -n 10

    echo "[$(date -Is)] seed ${s}: measurement -> ${OUT_CSV}"
done

# ---------------------------------------------------------------------------
# Phase 3: Measurement panel on frozen random-init (baseline)
# ---------------------------------------------------------------------------
echo
echo "[$(date -Is)] PHASE 3: Measurement panel (frozen random-init baseline)"

for s in "${SEEDS_ARR[@]}"; do
    OUT_CSV="${RESULTS_BASE}/seed${s}/measurement_panel_random.csv"

    python experiments/mpe_measurement_panel.py \
        --n-agents "${N_AGENTS}" \
        --n-draws 2000 \
        --seed "${s}" \
        --output-csv "${OUT_CSV}" \
        2>&1 | tail -n 10

    echo "[$(date -Is)] seed ${s}: random-init measurement -> ${OUT_CSV}"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "[$(date -Is)] MOVE 2 COMPLETE"
echo "============================================================"
echo
echo "Artefacts:"
for s in "${SEEDS_ARR[@]}"; do
    echo "  seed ${s}:"
    echo "    training log:       ${RESULTS_BASE}/seed${s}/mpe_training_log.csv"
    echo "    checkpoint:         ${RESULTS_BASE}/seed${s}/checkpoint_seed${s}.pt"
    echo "    measurement:        ${RESULTS_BASE}/seed${s}/measurement_panel.csv"
    echo "    measurement (rand): ${RESULTS_BASE}/seed${s}/measurement_panel_random.csv"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo
    echo "WARNING: training failed on seeds: ${FAILED[*]}" >&2
    exit 1
fi
exit 0
