#!/usr/bin/env bash
# Wait until selected GPUs have at least MIN_FREE_MB MiB free memory each.
#
# Usage:
#   bash scripts/wait_for_gpu_headroom.sh 0,1
#
# Env:
#   MIN_FREE_MB   (default 18000)  — per-GPU free memory threshold (MiB)
#   POLL_SEC      (default 30)     — sleep between checks

set -uo pipefail

GPUS="${1:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
POLL_SEC="${POLL_SEC:-30}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

echo "[$(date -Is)] Waiting for GPU headroom on GPUs: ${GPU_ARR[*]} (need >= ${MIN_FREE_MB} MiB free each)"

while true; do
  ok=1
  for g in "${GPU_ARR[@]}"; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${g}" | head -n 1 | tr -d ' ')"
    if ! [[ "${free_mb}" =~ ^[0-9]+$ ]]; then
      echo "[$(date -Is)] WARN: could not parse free memory for GPU ${g} (got '${free_mb}')." >&2
      ok=0
      break
    fi
    if [[ "${free_mb}" -lt "${MIN_FREE_MB}" ]]; then
      ok=0
      echo "[$(date -Is)] GPU ${g}: free=${free_mb} MiB (< ${MIN_FREE_MB}). Sleeping ${POLL_SEC}s..."
      break
    fi
  done

  if [[ "${ok}" -eq 1 ]]; then
    echo "[$(date -Is)] GPU headroom OK."
    exit 0
  fi

  sleep "${POLL_SEC}"
done
