#!/bin/bash
# Train the LoRA adapter on II-Medical-8B.
# Data: RAC-filtered BC + GPT distilled trajectories with branching
# (see ../data/ and ../filter/ for how the jsonl was produced).
#
# Required env (or override defaults below):
#   GPU            — CUDA device index (default: 0)
#   PYTHON         — python interpreter (default: python on PATH)
#   SFT_SCRIPT     — path to sft.py        (default: ./sft.py next to this file)
#   DATA_DIR       — root with data/, models/, checkpoints/, logs/  (default: ./output)
#   TRAIN_JSONL    — training data path    (default: $DATA_DIR/data/train.jsonl)
#   ADAPTER_NAME   — adapter dir name      (default: adapter)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${GPU:-0}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
P=${PYTHON:-python}
S=${SFT_SCRIPT:-"$SCRIPT_DIR/sft.py"}
D=${DATA_DIR:-"$SCRIPT_DIR/../output"}
ADAPTER_NAME=${ADAPTER_NAME:-adapter}
TRAIN_JSONL=${TRAIN_JSONL:-"$D/data/train.jsonl"}

[ -f "$D/models/${ADAPTER_NAME}/adapter_config.json" ] || \
"$P" "$S" \
    --data        "$TRAIN_JSONL" \
    --output-dir  "$D/checkpoints/${ADAPTER_NAME}" \
    --adapter-dir "$D/models/${ADAPTER_NAME}" \
    --log-dir     "$D/logs/${ADAPTER_NAME}"