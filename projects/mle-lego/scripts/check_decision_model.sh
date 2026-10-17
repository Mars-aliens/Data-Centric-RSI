#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Decision LLM snapshot directory}"

test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$MODEL_PATH/tokenizer.json"

conda run -n vllm-serve python -c \
  "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_PATH', local_files_only=True); print('decision model tokenizer: ok')"

conda run -n vllm-serve python -c \
  "import torch, transformers, vllm; print('torch', torch.__version__); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('cuda', torch.cuda.is_available())"
