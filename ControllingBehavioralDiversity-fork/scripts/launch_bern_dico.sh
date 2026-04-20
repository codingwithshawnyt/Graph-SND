#!/usr/bin/env bash
# NeurIPS Dispersion DiCo runs -- Bernoulli-p Graph-SND, single seed.
#
# Complements launch_knn_dico.sh: where that script validates a *fixed*
# structured $G$ (k-NN) inside DiCo, this script validates a *random*
# $G$ (Bernoulli-p sampling) inside DiCo on the same task/hyperparameters.
# Restoring the "both interpretations" promise of the paper without
# re-running the existing IPPO / Full SND / k-NN lines.
#
# Output layout (one CSV per seed):
#   results/neurips_final/seed${SEED}/bern/graph_snd_log.csv
#
# Usage (one seed, GPU 0, p=0.1):
#   SEED=0 bash scripts/launch_bern_dico.sh
#
# Usage (one seed on GPU 1 with p=0.25):
#   SEED=1 GPU=1 P=0.25 bash scripts/launch_bern_dico.sh
#
# Usage (driver loops seeds across both GPUs):
#   SEEDS="0 1 2" bash scripts/launch_bern_dico_seeds.sh
#
# Env vars:
#   SEED                    (default 0)       -- RL seed, also labels the output dir
#   GPU                     (default 0)       -- CUDA_VISIBLE_DEVICES for this run
#   P                       (default 0.1)     -- Bernoulli inclusion probability
#   N_AGENTS                (default 10)      -- must match launch_knn_dico.sh
#   MAX_ITERS               (default 167)     -- must match launch_knn_dico.sh
#   DESIRED_SND             (default 0.1)     -- must match launch_knn_dico.sh
#   N_FOOD                  (default N_AGENTS)
#   FORCE                   (default 0)       -- set 1 to wipe an existing run dir
#   RESULTS_BASE            (default $ROOT/results/neurips_final)
#
# On n=10 the Bernoulli(p=0.1) estimator sees ~4-5 sampled edges per
# agent-layout call (binom(10,2)*0.1 = 4.5). That is deliberately small:
# it demonstrates that DiCo's closed loop still tracks set point under a
# deliberately sparse sample, which is the regime the paper's unbiased
# sampling-estimator interpretation is about.
#
# Blocks until the run finishes; driver scripts rely on that.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

if [[ -f "$ROOT/../.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/../.venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
fi

SEED="${SEED:-0}"
GPU="${GPU:-0}"
P="${P:-0.1}"
N_AGENTS="${N_AGENTS:-10}"
MAX_ITERS="${MAX_ITERS:-167}"
DESIRED_SND="${DESIRED_SND:-0.1}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
FORCE="${FORCE:-0}"
LOGGERS='experiment.loggers=[]'

RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final}"
RESULTS_DIR="${RESULTS_BASE}/seed${SEED}"
BERN_DIR="${RESULTS_DIR}/bern"

# Pick the estimator key the fork's model registry already understands.
# graph_p01 / graph_p025 are the two Bernoulli paths exposed by
# het_control/graph_snd.py; anything else would require a new dispatch.
case "${P}" in
    0.1|0.10|1e-1)   ESTIMATOR="graph_p01"  ;;
    0.25|0.250)      ESTIMATOR="graph_p025" ;;
    *)
        echo "ERROR: P=${P} not supported by the fork's Graph-SND dispatch." >&2
        echo "       Supported: 0.1 -> graph_p01, 0.25 -> graph_p025." >&2
        exit 2
        ;;
esac

if [[ -s "${BERN_DIR}/graph_snd_log.csv" && "$FORCE" != "1" ]]; then
    echo "ERROR: ${BERN_DIR} already contains graph_snd_log.csv." >&2
    echo "       Re-run with FORCE=1 to wipe and replace." >&2
    exit 1
fi
if [[ "$FORCE" == "1" ]]; then
    rm -rf "$BERN_DIR"
fi
mkdir -p "$BERN_DIR"

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)

HYDRA_COMMON=(
    "${LOGGERS}"
    "seed=${SEED}"
    "task.n_agents=${N_AGENTS}"
    "task.n_food=${N_FOOD}"
    "task.share_reward=true"
    "experiment.max_n_iters=${MAX_ITERS}"
    "experiment.render=false"
    "experiment.train_device=cuda:0"
    "experiment.sampling_device=cuda:0"
    "experiment.buffer_device=cpu"
)

BERN_LOG="logs/neurips_final_seed${SEED}_bern_p${P}.log"

echo "[$(date -Is)] seed=${SEED} gpu=${GPU} estimator=${ESTIMATOR} p=${P} -> ${BERN_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=${ESTIMATOR}" \
    "model.diversity_p=${P}" \
    "hydra.run.dir=${BERN_DIR}" \
    > "${BERN_LOG}" 2>&1 &
PID=$!

echo "[$(date -Is)] Spawned PID=${PID} (GPU ${GPU}); first health check in 10 s..."
sleep 10

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: Bernoulli DiCo (PID ${PID}) exited immediately. Tail:" >&2
    tail -n 80 "${BERN_LOG}" >&2 || true
    exit 1
fi

echo "[$(date -Is)] Waiting for PID=${PID} ..."
wait "$PID" 2>/dev/null
rc=$?

if [[ "$rc" -ne 0 ]]; then
    echo "[$(date -Is)] ERROR: Bernoulli DiCo exited with rc=${rc}. Tail:" >&2
    tail -n 80 "${BERN_LOG}" >&2 || true
    exit "$rc"
fi

if [[ ! -s "${BERN_DIR}/graph_snd_log.csv" ]]; then
    echo "[$(date -Is)] ERROR: ${BERN_DIR}/graph_snd_log.csv missing or empty." >&2
    exit 3
fi

rows=$(wc -l < "${BERN_DIR}/graph_snd_log.csv" | tr -d ' ')
echo "[$(date -Is)] seed=${SEED} done. ${BERN_DIR}/graph_snd_log.csv = ${rows} lines."
