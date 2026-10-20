#!/bin/bash
# Eval the adapter over the evaluation case-id lists.
#
# Each .txt file in CASE_IDS_DIR is treated as one evaluation run; results are
# averaged across runs. Drop one or more case-id list files in CASE_IDS_DIR.
#
# Required env (or override defaults below):
#   GPU            — CUDA device index (default: 0)
#   PYTHON         — python interpreter (default: python)
#   EVAL_SCRIPT    — eval_model.py path (default: ./eval_model.py next to this file)
#   ADAPTER        — adapter dir
#   CASES_DIR      — directory holding {PMC...}.json case files
#   CASE_IDS_DIR   — directory holding the case-id list (.txt) files
#   RESULTS_DIR    — output directory for xlsx (default: ./results)
#   RUN_PREFIX     — prefix for the per-run name (default: run)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${GPU:-0}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
EVAL=${EVAL_SCRIPT:-"$SCRIPT_DIR/eval_model.py"}
ADAPTER=${ADAPTER:?"ADAPTER needs to be set"}
CASES_DIR=${CASES_DIR:?"CASES_DIR needs to be set"}
CASE_IDS_DIR=${CASE_IDS_DIR:?"CASE_IDS_DIR needs to be set"}
RESULTS_DIR=${RESULTS_DIR:-"./results"}
RUN_PREFIX=${RUN_PREFIX:-run}

mkdir -p "$RESULTS_DIR"

for CASE_IDS in "$CASE_IDS_DIR"/*.txt; do
    [ -e "$CASE_IDS" ] || { echo "No .txt case-id lists found in $CASE_IDS_DIR"; exit 1; }
    STEM=$(basename "$CASE_IDS" .txt)
    RUN_NAME="${RUN_PREFIX}_${STEM}"
    RESULT="$RESULTS_DIR/inference_results_${RUN_NAME}.xlsx"
    echo ""; echo "=== Eval ${RUN_PREFIX} on ${STEM} ==="
    [ -f "$RESULT" ] && echo "SKIP $STEM" && continue
    "$PYTHON" "$EVAL" \
        --adapter    "$ADAPTER" \
        --run-name   "$RUN_NAME" \
        --cases-dir  "$CASES_DIR" \
        --case-ids   "$CASE_IDS" \
        --output-dir "$RESULTS_DIR" \
        --resume
done

echo ""
echo "Done — $(date)"
