#!/usr/bin/env bash
# Resilient wrapper around scripts/launch_scale_training.sh:
# - optionally waits for GPU memory headroom
# - retries on failure with progressively smaller n=500 knobs
#
# Usage (from fork root):
#   bash scripts/launch_scale_training_resilient.sh
#
# Common:
#   WAIT_GPUS=0,1 MIN_FREE_MB=20000 bash scripts/launch_scale_training_resilient.sh
#   MAX_ATTEMPTS=5 bash scripts/launch_scale_training_resilient.sh
#
# All env vars are forwarded to launch_scale_training.sh except the ones this
# wrapper consumes: WAIT_GPUS, MIN_FREE_MB, POLL_SEC, MAX_ATTEMPTS

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WAIT_GPUS="${WAIT_GPUS:-}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
POLL_SEC="${POLL_SEC:-30}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

if [[ -n "${WAIT_GPUS}" ]]; then
  MIN_FREE_MB="${MIN_FREE_MB}" POLL_SEC="${POLL_SEC}" \
    bash "${ROOT}/scripts/wait_for_gpu_headroom.sh" "${WAIT_GPUS}"
fi

attempt=1
while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
  echo
  echo "============================================================"
  echo "[$(date -Is)] SCALE resilient attempt ${attempt}/${MAX_ATTEMPTS}"
  echo "============================================================"

  if bash "${ROOT}/scripts/launch_scale_training.sh"; then
    echo "[$(date -Is)] SUCCESS on attempt ${attempt}."
    exit 0
  fi

  rc=$?
  echo "[$(date -Is)] launch_scale_training.sh failed (exit=${rc})." >&2

  # Ratchet down the most failure-prone knobs for the next attempt.
  # (Safe for n=250-only runs too: smaller env/minibatch just slows training.)
  export FORCE="${FORCE:-1}"

  # Env parallelism
  if [[ "${N500_ENV_N+set}" == "set" ]]; then
    export N500_ENV_N=$(( N500_ENV_N > 1 ? N500_ENV_N - 1 : 1 ))
  fi
  if [[ "${N250_ENV_N+set}" == "set" ]]; then
    export N250_ENV_N=$(( N250_ENV_N > 1 ? N250_ENV_N - 1 : 1 ))
  fi

  # PPO minibatches (applies globally + n=500-specific defaults in launcher)
  if [[ "${ONP_MINIBATCH_SIZE+set}" == "set" ]]; then
    export ONP_MINIBATCH_SIZE=$(( ONP_MINIBATCH_SIZE > 256 ? ONP_MINIBATCH_SIZE / 2 : 256 ))
  else
    export ONP_MINIBATCH_SIZE=1024
  fi
  if [[ "${ONP_MINIBATCH_ITERS+set}" == "set" ]]; then
    export ONP_MINIBATCH_ITERS=$(( ONP_MINIBATCH_ITERS > 1 ? ONP_MINIBATCH_ITERS - 1 : 1 ))
  else
    export ONP_MINIBATCH_ITERS=3
  fi

  if [[ "${N500_ONP_MINIBATCH_SIZE+set}" == "set" ]]; then
    export N500_ONP_MINIBATCH_SIZE=$(( N500_ONP_MINIBATCH_SIZE > 256 ? N500_ONP_MINIBATCH_SIZE / 2 : 256 ))
  fi
  if [[ "${N500_ONP_MINIBATCH_ITERS+set}" == "set" ]]; then
    export N500_ONP_MINIBATCH_ITERS=$(( N500_ONP_MINIBATCH_ITERS > 1 ? N500_ONP_MINIBATCH_ITERS - 1 : 1 ))
  fi

  # Optional: shrink networks for feasibility after a couple failures
  if [[ "${attempt}" -ge 3 ]]; then
    export N500_MODEL_CELLS="${N500_MODEL_CELLS:-[128,128]}"
    export N500_CRITIC_CELLS="${N500_CRITIC_CELLS:-[128,128]}"
  fi

  attempt=$((attempt + 1))
  sleep 5
done

echo "[$(date -Is)] Gave up after ${MAX_ATTEMPTS} attempts." >&2
exit 1
