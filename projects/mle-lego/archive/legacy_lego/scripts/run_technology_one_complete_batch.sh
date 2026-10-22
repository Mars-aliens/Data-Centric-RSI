#!/usr/bin/env bash
set -euo pipefail

# Complete Technology One server package. This script is intentionally
# resumable: each child runner skips a seed when its summary already exists.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEGO_ROOT="${LEGO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export TRAIN_PY="${TRAIN_PY:-${LEGO_ROOT}/.venv/bin/python}"
export PYTHONPATH="${LEGO_ROOT}/code:${PYTHONPATH:-}"
export LEGO_POLICY_API_URL="${LEGO_POLICY_API_URL:-http://127.0.0.1:8000/v1/chat/completions}"
export PROFILE_EVOLUTION="${PROFILE_EVOLUTION:-${LEGO_ROOT}/server_profile.evolution.qwen.json}"
export PROFILE_BASELINE="${PROFILE_BASELINE:-${LEGO_ROOT}/server_profile.baseline_matrix.qwen.json}"
export GPU_ID="${GPU_ID:-6}"
export SEEDS="${SEEDS:-17 23 29 31 37 41 43 47 53 59}"
export BATCHES="${BATCHES:-3}"

COMPONENT_ROOT="${COMPONENT_ROOT:-${LEGO_ROOT}/runs/qwen-baseline-matrix}"
DECISION_ROOT="${DECISION_ROOT:-${LEGO_ROOT}/runs/qwen-technology-one-decision-baselines}"
EVOLUTION_ROOT="${EVOLUTION_ROOT:-${LEGO_ROOT}/runs/qwen-technology-one-evolution-comparison}"

mkdir -p "$LEGO_ROOT/logs"

if [ "${RUN_COMPONENT_MATRIX:-0}" = "1" ]; then
  PROFILE="$PROFILE_BASELINE" OUTPUT_DIR="$COMPONENT_ROOT" GPU_ID="$GPU_ID" \
    SEEDS="$SEEDS" bash "$LEGO_ROOT/scripts/run_baseline_matrix_batch.sh"
fi

METHODS="fixed-harness-react react-reflexion" \
  PROFILE="$PROFILE_BASELINE" OUTPUT_DIR="$DECISION_ROOT" GPU_ID="$GPU_ID" \
  SEEDS="$SEEDS" bash "$LEGO_ROOT/scripts/run_technology_one_decision_baseline_batch.sh"

PROFILE="$PROFILE_EVOLUTION" OUTPUT_ROOT="$EVOLUTION_ROOT" GPU_ID="$GPU_ID" \
  SEEDS="$SEEDS" BATCHES="$BATCHES" \
  bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"

echo "Technology One experiment package finished. Run the complete audit next."
