#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
DATASET="${LEGO_ROOT}/data/helpsteer2-train.jsonl"

test -x "$TRAIN_PY"
test -f "$DATASET"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"

"$TRAIN_PY" -c 'import sklearn, numpy; print("core dependencies: ok")'
"$TRAIN_PY" -m compileall -q "$LEGO_ROOT/code/core_system" "$LEGO_ROOT/code/technologies"
"$TRAIN_PY" -m local_experiments.lego.validate_data "$DATASET"

echo "LEGO install check: passed"
echo "code: ${LEGO_ROOT}/code"
echo "data: ${DATASET}"
