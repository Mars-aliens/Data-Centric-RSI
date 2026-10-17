#!/usr/bin/env bash
set -euo pipefail

# Serial paired comparison for Technology One. Both arms use the same seeds,
# profile, batch count, train-group budget, and declared budgets; only the
# proposal constraint differs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.evolution.qwen.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LEGO_ROOT}/runs/qwen-technology-one-evolution-comparison}"
GPU_ID="${GPU_ID:-6}"
BATCHES="${BATCHES:-3}"
SEEDS="${SEEDS:-17 23 29 31 37 41 43 47 53 59}"
MODES="${MODES:-independent dependency-aware-joint}"

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
mkdir -p "$OUTPUT_ROOT" "$LEGO_ROOT/logs"

for MODE in $MODES; do
  MODE_ROOT="$OUTPUT_ROOT/$MODE"
  for SEED in $SEEDS; do
    RUN_ID="qwen-technology-one-evolution-${MODE}-seed-${SEED}-batches-${BATCHES}"
    SUMMARY="$MODE_ROOT/$RUN_ID/summary.json"
    if [ -f "$SUMMARY" ]; then
      echo "SKIP mode=$MODE seed=$SEED: $SUMMARY"
      continue
    fi
    LOG="$LEGO_ROOT/logs/${RUN_ID}.log"
    echo "START mode=$MODE seed=$SEED run=$RUN_ID"
    SEED="$SEED" BATCHES="$BATCHES" PROPOSAL_MODE="$MODE" \
      PROFILE="$PROFILE" OUTPUT_DIR="$MODE_ROOT" RUN_ID="$RUN_ID" \
      bash "$LEGO_ROOT/scripts/run_technology_one_evolution.sh" \
      > "$LOG" 2>&1
    echo "DONE mode=$MODE seed=$SEED run=$RUN_ID"
  done
done
