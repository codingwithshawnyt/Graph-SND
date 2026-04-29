#!/usr/bin/env bash
# Copy n=250 scale_training results (and optional n=250 logs) from riddle to this Mac.
# Prereq: VPN + SSH to riddle (e.g. `ssh shawnr@172.24.170.204` works).
# Usage (from ControllingBehavioralDiversity-fork):
#   bash scripts/copy_n250_from_riddle_to_mac.sh
# Override: RIDDLE_SSH=shawnr@HOST bash scripts/copy_n250_from_riddle_to_mac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${RIDDLE_SSH:-shawnr@172.24.170.204}"
REMOTE_FORK="/usr1/home/shawnr/shawnr/Graph-SND/ControllingBehavioralDiversity-fork"
REMOTE_N250="${REMOTE_FORK}/results/scale_training/n250"
REMOTE_LOGS_DIR="${REMOTE_FORK}/logs"

mkdir -p "${ROOT}/results/scale_training" "${ROOT}/logs"

echo "[copy_n250] REMOTE=${REMOTE}"
echo "[copy_n250] Pulling: ${REMOTE_N250} -> ${ROOT}/results/scale_training/"
scp -r "${REMOTE}:${REMOTE_N250}" "${ROOT}/results/scale_training/"

echo "[copy_n250] Pulling: scale_n250 seed0 logs -> ${ROOT}/logs/"
scp "${REMOTE}:${REMOTE_LOGS_DIR}/scale_n250_seed0_full.log" \
  "${REMOTE}:${REMOTE_LOGS_DIR}/scale_n250_seed0_bern_p01.log" \
  "${ROOT}/logs/"

echo "[copy_n250] graph_snd_log.csv line counts:"
find "${ROOT}/results/scale_training/n250" -name graph_snd_log.csv -print -exec wc -l {} \;
