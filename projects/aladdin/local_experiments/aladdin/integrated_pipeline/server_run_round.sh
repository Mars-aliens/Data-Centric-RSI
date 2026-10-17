#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${TRAIN_PY:-python}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to normalized seed/development JSONL}"
REQUIREMENTS_PATH="${REQUIREMENTS_PATH:?set REQUIREMENTS_PATH to requirements.json}"
ENDPOINT="${ENDPOINT:?set ENDPOINT to vLLM /v1/chat/completions URL}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the served model name}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a new or resumable run directory}"
RUN_ID="${RUN_ID:?set RUN_ID}"
if [[ -n "${FEEDBACK_MODE:-}" ]]; then
  FEEDBACK_MODE_VALUE="$FEEDBACK_MODE"
elif [[ -n "${FEEDBACK_FILE:-}" ]]; then
  FEEDBACK_MODE_VALUE="file"
else
  FEEDBACK_MODE_VALUE="offline-replay"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  -m local_experiments.aladdin.integrated_pipeline.run_configured
  --data "$DATA_PATH"
  --requirements "$REQUIREMENTS_PATH"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --endpoint "$ENDPOINT"
  --model "$MODEL_NAME"
  --rounds "${ROUNDS:-1}"
  --quality-ensemble-size "${QUALITY_ENSEMBLE_SIZE:-1}"
  --feedback-mode "$FEEDBACK_MODE_VALUE"
  --candidate-count "${CANDIDATE_COUNT:-4}"
  --generation-tokens "${GENERATION_TOKENS:-1536}"
  --seed-example-limit "${SEED_EXAMPLE_LIMIT:-16}"
  --seed-example-chars "${SEED_EXAMPLE_CHARS:-700}"
)
if [[ -n "${CALIBRATION_PATH:-}" ]]; then
  ARGS+=(--calibration "$CALIBRATION_PATH")
fi
if [[ -n "${QUALITY_AUDIT_PATH:-}" ]]; then
  ARGS+=(--quality-audit-file "$QUALITY_AUDIT_PATH")
fi
if [[ -n "${RESUME_STATE:-}" ]]; then
  ARGS+=(--resume-state "$RESUME_STATE")
fi
if [[ "$FEEDBACK_MODE_VALUE" == "interactive" ]]; then
  ARGS+=(--interactive-feedback)
elif [[ "$FEEDBACK_MODE_VALUE" == "file" ]]; then
  : "${FEEDBACK_FILE:?set FEEDBACK_FILE when FEEDBACK_MODE=file}"
  ARGS+=(--feedback-file "$FEEDBACK_FILE")
fi
exec "$PYTHON_BIN" "${ARGS[@]}"
