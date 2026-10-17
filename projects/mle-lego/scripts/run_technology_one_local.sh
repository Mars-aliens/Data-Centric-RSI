#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
DATASET="${LEGO_ROOT}/data/helpsteer2-train.jsonl"
OUTPUT="${LEGO_ROOT}/runs/technology-one-server-smoke"

test -f "$DATASET"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$LEGO_ROOT/runs" "$LEGO_ROOT/logs"

exec "$TRAIN_PY" -m local_experiments.lego.technical_one_local \
  "$DATASET" \
  --output-dir "$OUTPUT" \
  --max-train-groups "${LEGO_TRAIN_GROUPS:-200}" \
  --max-eval-groups "${LEGO_EVAL_GROUPS:-300}"
