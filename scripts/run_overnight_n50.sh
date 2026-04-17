#!/usr/bin/env bash
# Companion overnight run at n=50 on the second GPU.
#
# Run this alongside scripts/run_overnight_n100.sh to keep both 4090s
# busy overnight. The two processes write to disjoint checkpoint and
# log namespaces (n50_* vs n100_*), so they compose trivially:
#
#   bash scripts/run_overnight_n100.sh          # cuda:0, n=100
#   DEVICE=cuda:1 bash scripts/run_overnight_n50.sh  # cuda:1, n=50
#
# Defaults match run_overnight_n100.sh apart from N_AGENTS and TAG.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${DEVICE:=cuda:1}"
: "${N_AGENTS:=50}"
: "${ITERS:=5000}"
: "${NUM_ENVS:=64}"
: "${ROLLOUT_STEPS:=128}"
: "${MINIBATCH:=2048}"
: "${EPOCHS:=4}"
: "${SEED:=0}"
: "${CKPT_EVERY:=250}"
: "${SND_EVERY:=50}"
: "${SND_P:=0.1}"
: "${TAG:=overnight}"
: "${RESUME:=}"

export DEVICE N_AGENTS ITERS NUM_ENVS ROLLOUT_STEPS MINIBATCH EPOCHS SEED \
    CKPT_EVERY SND_EVERY SND_P TAG RESUME

exec bash "$(dirname "$0")/run_overnight_n100.sh"
