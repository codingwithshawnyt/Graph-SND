#!/usr/bin/env bash
# Feasibility run: DiCo + Bernoulli-0.1 Graph-SND on Dispersion at n=50.
#
# Rationale: the paper's main DiCo integration (Section 6.5 / Figure 3)
# is at n=10 with three seeds. This script runs one additional seed at
# n=50 -- large enough to be meaningful under the scaling argument, small
# enough to fit on a single 4090 without arena-geometry tuning -- to
# demonstrate that DiCo + Bernoulli-0.1 Graph-SND still tracks set point
# and beats the diversity-off baseline at ~5x the agent count.
#
# Two runs, one per GPU, in parallel:
#
#   GPU 0: IPPO baseline (desired_snd=-1) -> results/neurips_final_n50/seed0/ippo/
#   GPU 1: DiCo + Bernoulli-0.1            -> results/neurips_final_n50/seed0/bern/
#
# Single seed is by design: this is a feasibility demonstration, not a
# statistical claim. The correctness-of-substitution claim for DiCo +
# Graph-SND already has three seeds at n=10.
#
# Defaults below assume riddle's 2x RTX 4090. on_policy_n_envs_per_worker
# is dropped from 600 -> 120 relative to the default het_control_experiment
# to keep per-iter GPU memory ~linear in agent count at n=50 (5x more
# per-agent tensors than n=10 was already running fine at 600 envs). All
# other experiment hyperparameters match launch_bern_dico.sh.
#
# Usage (from repo fork root, riddle):
#   bash scripts/launch_dico_n50_feasibility.sh
#
# Override defaults:
#   N_AGENTS=100 MAX_ITERS=167 bash scripts/launch_dico_n50_feasibility.sh
#   ENV_N=60 MAX_ITERS=250 bash scripts/launch_dico_n50_feasibility.sh  # more conservative
#
# Force-clean prior runs under the same RESULTS_BASE:
#   FORCE=1 bash scripts/launch_dico_n50_feasibility.sh

set -uo pipefail

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
N_AGENTS="${N_AGENTS:-50}"
MAX_ITERS="${MAX_ITERS:-167}"
DESIRED_SND="${DESIRED_SND:-0.1}"
P="${P:-0.1}"
N_FOOD="${N_FOOD:-${N_AGENTS}}"
ENV_N="${ENV_N:-120}"                      # on_policy_n_envs_per_worker
FRAMES_PER_BATCH="${FRAMES_PER_BATCH:-$(( ENV_N * 100 ))}"  # 100-step horizon
FORCE="${FORCE:-0}"
LOGGERS='experiment.loggers=[]'

RESULTS_BASE="${RESULTS_BASE:-${ROOT}/results/neurips_final_n50}"
RESULTS_DIR="${RESULTS_BASE}/seed${SEED}"
IPPO_DIR="${RESULTS_DIR}/ippo"
BERN_DIR="${RESULTS_DIR}/bern"

case "${P}" in
    0.1|0.10|1e-1)   ESTIMATOR="graph_p01"  ;;
    0.25|0.250)      ESTIMATOR="graph_p025" ;;
    *)
        echo "ERROR: P=${P} not supported by the fork's Graph-SND dispatch." >&2
        echo "       Supported: 0.1 -> graph_p01, 0.25 -> graph_p025." >&2
        exit 2
        ;;
esac

for d in "$IPPO_DIR" "$BERN_DIR"; do
    if [[ -s "$d/graph_snd_log.csv" && "$FORCE" != "1" ]]; then
        echo "ERROR: $d already contains graph_snd_log.csv." >&2
        echo "       Re-run with FORCE=1 to wipe and replace." >&2
        exit 1
    fi
    if [[ "$FORCE" == "1" ]]; then
        rm -rf "$d"
    fi
    mkdir -p "$d"
done

RUNNER=(python het_control/run_scripts/run_dispersion_ippo.py)

HYDRA_COMMON=(
    "${LOGGERS}"
    "seed=${SEED}"
    "task.n_agents=${N_AGENTS}"
    "task.n_food=${N_FOOD}"
    "task.share_reward=true"
    "experiment.max_n_iters=${MAX_ITERS}"
    "experiment.on_policy_n_envs_per_worker=${ENV_N}"
    "experiment.on_policy_collected_frames_per_batch=${FRAMES_PER_BATCH}"
    "experiment.render=false"
    "experiment.train_device=cuda:0"
    "experiment.sampling_device=cuda:0"
    "experiment.buffer_device=cpu"
)

IPPO_LOG="logs/neurips_n50_seed${SEED}_ippo.log"
BERN_LOG="logs/neurips_n50_seed${SEED}_bern_p${P}.log"

echo "[$(date -Is)] n=${N_AGENTS} seed=${SEED} iters=${MAX_ITERS} envs/worker=${ENV_N} -> ${RESULTS_DIR}"

echo "[$(date -Is)] GPU 0: IPPO baseline (desired_snd=-1)      -> ${IPPO_DIR}"
CUDA_VISIBLE_DEVICES=0 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=-1" \
    "model.diversity_estimator=full" \
    "hydra.run.dir=${IPPO_DIR}" \
    > "${IPPO_LOG}" 2>&1 &
PID_IPPO=$!

echo "[$(date -Is)] GPU 1: DiCo + Bernoulli-${P} (estimator=${ESTIMATOR}) -> ${BERN_DIR}"
CUDA_VISIBLE_DEVICES=1 nohup "${RUNNER[@]}" \
    --config-name dispersion_ippo_knn_config \
    "${HYDRA_COMMON[@]}" \
    "model.desired_snd=${DESIRED_SND}" \
    "model.diversity_estimator=${ESTIMATOR}" \
    "model.diversity_p=${P}" \
    "hydra.run.dir=${BERN_DIR}" \
    > "${BERN_LOG}" 2>&1 &
PID_BERN=$!

echo "[$(date -Is)] Spawned PID_IPPO=${PID_IPPO} (GPU 0), PID_BERN=${PID_BERN} (GPU 1); first health check in 15 s ..."
sleep 15

if ! kill -0 "$PID_IPPO" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: IPPO baseline exited immediately. Tail:" >&2
    tail -n 80 "${IPPO_LOG}" >&2 || true
fi
if ! kill -0 "$PID_BERN" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: Bernoulli DiCo exited immediately. Tail:" >&2
    tail -n 80 "${BERN_LOG}" >&2 || true
fi

echo "[$(date -Is)] Waiting on PID_IPPO=${PID_IPPO} PID_BERN=${PID_BERN} ..."
wait "$PID_IPPO" 2>/dev/null
rc_ippo=$?
wait "$PID_BERN" 2>/dev/null
rc_bern=$?

echo
echo "[$(date -Is)] n=${N_AGENTS} seed=${SEED} exit codes: IPPO=${rc_ippo}, Bernoulli=${rc_bern}"
echo "Artefacts:"
for tag in ippo bern; do
    csv="${RESULTS_DIR}/${tag}/graph_snd_log.csv"
    if [[ -s "$csv" ]]; then
        rows=$(wc -l < "$csv" | tr -d ' ')
        echo "  ${tag}: ${csv} (${rows} rows)"
    else
        echo "  ${tag}: ${csv} MISSING OR EMPTY (check logs/)"
    fi
done

echo
echo "To rebuild the feasibility companion figure locally (after rsync-ing the CSVs):"
echo "  python3 scripts/plot_reward_curves.py \\"
echo "      --figure-type panels --smooth 5 \\"
echo "      \"ippo:ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed${SEED}/ippo/graph_snd_log.csv\" \\"
echo "      \"graph_p01:ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed${SEED}/bern/graph_snd_log.csv\" \\"
echo "      --desired-snd ${DESIRED_SND} \\"
echo "      --task-name \"VMAS Dispersion (n=${N_AGENTS}, one seed)\" \\"
echo "      --output Paper/figures/neurips_n50_feasibility.pdf"

if [[ "$rc_ippo" -ne 0 || "$rc_bern" -ne 0 ]]; then
    exit 1
fi
exit 0
