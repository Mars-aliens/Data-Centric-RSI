#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export ALADDIN_SKLEARN_SITE="${ALADDIN_SKLEARN_SITE:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ALADDIN_TASK_MODEL_DEVICE="${ALADDIN_TASK_MODEL_DEVICE:-cuda}"
export ALADDIN_TASK_MODEL_MAX_GPU_MEMORY_GIB="${ALADDIN_TASK_MODEL_MAX_GPU_MEMORY_GIB:-60}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TRAIN_PY="${TRAIN_PY:-python}"
MODEL="${ALADDIN_MODEL_PATH:?set ALADDIN_MODEL_PATH to a local model directory}"
DATA_PATH="${ALADDIN_DATA_PATH:?set ALADDIN_DATA_PATH to an HH-RLHF JSONL file}"
OUTPUT="${ALADDIN_OUTPUT_PATH:?set ALADDIN_OUTPUT_PATH to an external JSON output path}"

test -f "$MODEL/config.json"
curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models >/dev/null
test ! -e "$OUTPUT"

exec "$TRAIN_PY" -u -m local_experiments.aladdin.hh_rlhf_qwen_lora_label_diagnostic \
  --protocol-version custom-diagnostic-v1 \
  --data "$DATA_PATH" \
  --output "$OUTPUT" \
  --base-model "$MODEL" \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --seeds 17 18 19 20 21 22 23 24 25 26 \
  --strategies seed-only random aladdin-qwen-reliable \
  --candidate-budget 32 \
  --seed-group-limit 64 \
  --pool-group-limit 256 \
  --eval-group-limit 512 \
  --eval-batch-size 8 \
  --split-seed 29 \
  --split-unit prompt \
  --fail-on-split-overlap \
  --max-steps 100 \
  --training-budget-mode equal-epochs \
  --batch-size 2 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --max-gpu-memory-gib 60
