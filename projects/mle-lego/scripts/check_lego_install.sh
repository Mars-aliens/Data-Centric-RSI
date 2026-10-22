#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"

test -x "$TRAIN_PY"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"

"$TRAIN_PY" -c 'import numpy, sklearn, torch, transformers, peft; print("server dependencies: ok")'
"$TRAIN_PY" -m compileall -q "$LEGO_ROOT/code/core_system" "$LEGO_ROOT/code/technologies"

echo "LEGO install check: passed"
echo "code: ${LEGO_ROOT}/code"
echo "HelpSteer2 data is external; prepare and hash the raw split JSONL before running T1."
