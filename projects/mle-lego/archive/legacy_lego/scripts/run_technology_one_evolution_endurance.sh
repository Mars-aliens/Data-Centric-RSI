#!/usr/bin/env bash
set -euo pipefail

# Long-horizon runner. Re-run this command with the same RUN_ID and RESUME=1
# after an interruption; the Python runner resumes from its atomic state file.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
PROFILE="${PROFILE:-${LEGO_ROOT}/server_profile.evolution.qwen.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEGO_ROOT}/runs/qwen-technology-one-evolution-endurance}"
SEED="${SEED:-17}"
BATCHES="${BATCHES:-300}"
RUN_ID="${RUN_ID:-qwen-technology-one-evolution-endurance-seed-${SEED}}"
RESUME="${RESUME:-0}"
COMMAND_RETRIES="${COMMAND_RETRIES:-3}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-10}"
PROPOSAL_MODE="${PROPOSAL_MODE:-dependency-aware}"
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-3000}"
MAX_EVAL_GROUPS="${MAX_EVAL_GROUPS:-1000}"
TRAIN_GROUP_BUDGET="${TRAIN_GROUP_BUDGET:-1000}"

test -x "$TRAIN_PY"
test -f "$PROFILE"
test -n "${LEGO_POLICY_API_URL:-}" || {
  echo "LEGO_POLICY_API_URL must point to the decision LLM endpoint" >&2
  exit 2
}

export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$LEGO_ROOT/logs"

# Defaults cover 300 cycles with one active and one candidate evaluation per
# cycle. Set any BUDGET_* variable explicitly when the profile uses another
# training/evaluation schedule.
export BUDGET_LLM_CALLS="${BUDGET_LLM_CALLS:-1200}"
export BUDGET_LLM_TOKENS="${BUDGET_LLM_TOKENS:-1200000}"
export BUDGET_TRAINING_STEPS="${BUDGET_TRAINING_STEPS:-70000}"
export BUDGET_TRAINING_SECONDS="${BUDGET_TRAINING_SECONDS:--1}"
export BUDGET_EVALUATOR_CALLS="${BUDGET_EVALUATOR_CALLS:-1400}"
export BUDGET_WALL_CLOCK_SECONDS="${BUDGET_WALL_CLOCK_SECONDS:--1}"

exec env \
  PROFILE="$PROFILE" OUTPUT_DIR="$OUTPUT_DIR" SEED="$SEED" \
  BATCHES="$BATCHES" RUN_ID="$RUN_ID" RESUME="$RESUME" \
  COMMAND_RETRIES="$COMMAND_RETRIES" \
  RETRY_BACKOFF_SECONDS="$RETRY_BACKOFF_SECONDS" \
  PROPOSAL_MODE="$PROPOSAL_MODE" \
  MAX_TRAIN_GROUPS="$MAX_TRAIN_GROUPS" MAX_EVAL_GROUPS="$MAX_EVAL_GROUPS" \
  TRAIN_GROUP_BUDGET="$TRAIN_GROUP_BUDGET" \
  bash "$LEGO_ROOT/scripts/run_technology_one_evolution.sh"
