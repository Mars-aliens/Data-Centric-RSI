#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.evolution.qwen.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEGO_ROOT}/runs/qwen-technology-one-evolution-pilot}"
SEED="${SEED:-17}"
BATCHES="${BATCHES:-3}"
COMMAND_RETRIES="${COMMAND_RETRIES:-0}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-5}"
PROPOSAL_MODE="${PROPOSAL_MODE:-dependency-aware}"
declare -a BUDGET_ARGS=()
for budget_name in \
  llm-calls llm-tokens training-steps training-seconds evaluator-calls \
  feedback-requests wall-clock-seconds; do
  budget_var="BUDGET_${budget_name//-/_}"
  budget_var="${budget_var^^}"
  budget_value="${!budget_var:-}"
  if [ -n "$budget_value" ]; then
    BUDGET_ARGS+=("--budget-${budget_name}" "$budget_value")
  fi
done

test -x "$TRAIN_PY"
test -f "$PROFILE"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$LEGO_ROOT/logs"
RUN_ID="${RUN_ID:-qwen-technology-one-evolution-seed-${SEED}-batches-${BATCHES}-$(date +%Y%m%d-%H%M%S)}"

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARGS+=(--resume)
fi

exec "$TRAIN_PY" -m local_experiments.lego.server_technical_one_evolution \
  --profile "$PROFILE" \
  --root "$LEGO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --seed "$SEED" \
  --batches "$BATCHES" \
  --max-train-groups "${MAX_TRAIN_GROUPS:-3000}" \
  --max-eval-groups "${MAX_EVAL_GROUPS:-1000}" \
  --train-group-budget "${TRAIN_GROUP_BUDGET:-1000}" \
  --command-retries "$COMMAND_RETRIES" \
  --retry-backoff-seconds "$RETRY_BACKOFF_SECONDS" \
  --proposal-mode "$PROPOSAL_MODE" \
  "${BUDGET_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
