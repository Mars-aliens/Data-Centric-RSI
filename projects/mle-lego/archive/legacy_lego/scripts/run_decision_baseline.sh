#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.baseline_matrix.qwen.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEGO_ROOT}/runs/qwen-decision-baselines}"
METHOD="${METHOD:-react-reflexion}"
SEED="${SEED:-17}"

test -x "$TRAIN_PY"
test -f "$PROFILE"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$LEGO_ROOT/logs"
RUN_ID="${RUN_ID:-qwen-decision-${METHOD}-$(date +%Y%m%d-%H%M%S)}"

exec "$TRAIN_PY" -m local_experiments.lego.server_runner \
  --profile "$PROFILE" \
  --root "$LEGO_ROOT" \
  --method "$METHOD" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --seed "$SEED"
