#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.evolution.qwen.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEGO_ROOT}/runs/qwen-technology-one-evolution-formal}"
GPU_ID="${GPU_ID:-6}"
BATCHES="${BATCHES:-3}"
SEEDS="${SEEDS:-17 23 29 31 37 41 43 47 53 59}"

test -x "$TRAIN_PY"
test -f "$PROFILE"
test -n "${LEGO_POLICY_API_URL:-}" || {
  echo "LEGO_POLICY_API_URL must point to the decision LLM endpoint" >&2
  exit 2
}

export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LEGO_TASK_MODEL_DEVICE="${LEGO_TASK_MODEL_DEVICE:-cuda}"
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB="${LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB:-40}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUTPUT_DIR" "$LEGO_ROOT/logs"

for SEED in $SEEDS; do
  DONE=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 \
    -path "*seed-${SEED}-*/summary.json" -print -quit)
  if [ -n "$DONE" ]; then
    echo "SKIP seed=$SEED: $DONE"
    continue
  fi

  RUN_ID="qwen-technology-one-evolution-formal-seed-${SEED}-$(date +%Y%m%d-%H%M%S)"
  LOG="$LEGO_ROOT/logs/${RUN_ID}.log"
  echo "START seed=$SEED run=$RUN_ID log=$LOG"
  SEED="$SEED" \
  BATCHES="$BATCHES" \
  PROFILE="$PROFILE" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  RUN_ID="$RUN_ID" \
  bash "$LEGO_ROOT/scripts/run_technology_one_evolution.sh" \
    > "$LOG" 2>&1
  echo "DONE seed=$SEED run=$RUN_ID"
done
