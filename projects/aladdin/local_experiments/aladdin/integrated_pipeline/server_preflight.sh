#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${TRAIN_PY:-python}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to normalized seed/development/held_in JSONL}"
REQUIREMENTS_PATH="${REQUIREMENTS_PATH:?set REQUIREMENTS_PATH to requirements.json}"
CALIBRATION_PATH="${CALIBRATION_PATH:-}"
ENDPOINT="${ENDPOINT:-}"
if [[ -n "${FEEDBACK_MODE:-}" ]]; then
  FEEDBACK_MODE_VALUE="$FEEDBACK_MODE"
else
  FEEDBACK_MODE_VALUE="offline-replay"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  -m local_experiments.aladdin.integrated_pipeline.preflight
  --data "$DATA_PATH"
  --requirements "$REQUIREMENTS_PATH"
  --feedback-mode "$FEEDBACK_MODE_VALUE"
)
if [[ -n "$CALIBRATION_PATH" ]]; then
  ARGS+=(--calibration "$CALIBRATION_PATH")
fi
if [[ -n "$ENDPOINT" ]]; then
  ARGS+=(--endpoint "$ENDPOINT")
fi
exec "$PYTHON_BIN" "${ARGS[@]}"
