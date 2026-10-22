#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.baseline_matrix.qwen.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEGO_ROOT}/runs/qwen-technology-one-decision-baselines}"
GPU_ID="${GPU_ID:-6}"
METHODS="${METHODS:-fixed-harness-react react-reflexion}"
SEEDS="${SEEDS:-17 23 29 31 37 41 43 47 53 59}"

test -x "$TRAIN_PY"
test -f "$PROFILE"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LEGO_TASK_MODEL_DEVICE="${LEGO_TASK_MODEL_DEVICE:-cuda}"
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB="${LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB:-40}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
test -n "${LEGO_POLICY_API_URL:-}" || {
  echo "LEGO_POLICY_API_URL must point to the decision LLM endpoint" >&2
  exit 2
}
mkdir -p "$OUTPUT_DIR" "$LEGO_ROOT/logs"

for METHOD in $METHODS; do
  METHOD_ROOT="$OUTPUT_DIR/$METHOD"
  for SEED in $SEEDS; do
    DONE=$(find "$METHOD_ROOT" -mindepth 2 -maxdepth 2 \
      -path "*seed-${SEED}-*/summary.json" -print -quit)
    if [ -n "$DONE" ]; then
      echo "SKIP method=$METHOD seed=$SEED: $DONE"
      continue
    fi
    RUN_ID="qwen-technology-one-${METHOD}-seed-${SEED}-$(date +%Y%m%d-%H%M%S)"
    LOG="$LEGO_ROOT/logs/${RUN_ID}.log"
    echo "START method=$METHOD seed=$SEED run=$RUN_ID"
    METHOD="$METHOD" SEED="$SEED" PROFILE="$PROFILE" \
      OUTPUT_DIR="$METHOD_ROOT" RUN_ID="$RUN_ID" \
      bash "$LEGO_ROOT/scripts/run_decision_baseline.sh" \
      > "$LOG" 2>&1
    echo "DONE method=$METHOD seed=$SEED run=$RUN_ID"
  done
done
