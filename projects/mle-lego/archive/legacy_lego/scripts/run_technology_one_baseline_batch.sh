#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.baseline.qwen.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEGO_ROOT}/runs/qwen-technology-one-fixed-baseline}"
GPU_ID="${GPU_ID:-5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-65000}"
SEEDS="${SEEDS:-17 23 29 31 37 41 43 47 53 59}"

test -x "$TRAIN_PY"
test -f "$PROFILE"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$LEGO_ROOT/logs"

for SEED in $SEEDS; do
  if find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 \
      -path "*seed-${SEED}-*/summary.json" | grep -q .; then
    echo "SKIP seed=${SEED}, complete fixed baseline summary exists"
    continue
  fi

  while true; do
    if pgrep -f "[s]erver_technical_one(_baseline)?" >/dev/null; then
      echo "$(date) another LEGO technology-one run is active; waiting"
      sleep 60
      continue
    fi
    FREE_MIB=$(nvidia-smi -i "$GPU_ID" \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d ' ')
    echo "$(date) seed=${SEED} GPU${GPU_ID} free=${FREE_MIB} MiB"
    if [ "$FREE_MIB" -ge "$MIN_FREE_MIB" ]; then
      break
    fi
    sleep 60
  done

  RUN_ID="qwen-technology-one-fixed-baseline-seed-${SEED}-gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)"
  LOG="$LEGO_ROOT/logs/${RUN_ID}.log"
  echo "START ${RUN_ID}"
  if "$TRAIN_PY" -m local_experiments.lego.server_technical_one_baseline \
      --profile "$PROFILE" \
      --root "$LEGO_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$RUN_ID" \
      --seed "$SEED" \
      --max-train-groups "${MAX_TRAIN_GROUPS:-3000}" \
      --max-eval-groups "${MAX_EVAL_GROUPS:-1000}" \
      --train-group-budget "${TRAIN_GROUP_BUDGET:-1000}" \
      > "$LOG" 2>&1; then
    echo "DONE ${RUN_ID}"
  else
    STATUS=$?
    echo "FAILED seed=${SEED} exit=${STATUS} log=${LOG}; stopping batch"
    exit "$STATUS"
  fi
done
