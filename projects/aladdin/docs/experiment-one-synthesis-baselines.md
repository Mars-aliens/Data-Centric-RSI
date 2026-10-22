# 实验一（1A）运行步骤

先完成 [`服务器配置步骤`](server-quickstart.md)第 1～3 步。以下命令在 Linux 项目目录中执行。

## 1. 设置模型、数据和输出路径

```bash
set -euo pipefail

export ALADDIN_ROOT="$PWD"
export PYTHON_BIN="$ALADDIN_ROOT/.venv/bin/python"
export PYTHONPATH="$ALADDIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=<训练用GPU编号>
export ALADDIN_ENDPOINT="<OpenAI兼容Chat Completions地址>"
export ALADDIN_MODEL="Qwen/Qwen3-8B"
export MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export SEED_DATA="$ALADDIN_ROOT/data/processed/prism/seed.jsonl"
export DEV_DATA="$ALADDIN_ROOT/data/processed/prism/development.jsonl"
export DEV_LABELS="$ALADDIN_ROOT/data/processed/prism/labels/development-labels.jsonl"

export BASE_MODEL="Qwen/Qwen3-8B"
export RUN="$ALADDIN_ROOT/runs/experiment-1a-prism-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
```

## 2. 检查生成 API

```bash
export API_CHECK="$ALADDIN_ROOT/runs/api-check-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$API_CHECK"

"$PYTHON_BIN" -m experiments.synthesis \
  --data "$SEED_DATA" \
  --output "$API_CHECK/fixed-prompt.jsonl" \
  --strategy fixed-prompt --count 1 --seed 17 \
  --max-tokens 512 --temperature 0.7 \
  --endpoint "$ALADDIN_ENDPOINT" --model "$ALADDIN_MODEL"

"$PYTHON_BIN" - "$API_CHECK/fixed-prompt.jsonl.manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
usage = manifest["generation_usage"]
assert manifest["status"] == "completed"
assert manifest["model"] == "Qwen/Qwen3-8B"
assert usage["requests"] == 1
assert usage["reported_requests"] == 1
assert usage["complete_usage_requests"] == 1
assert usage["prompt_tokens"] > 0
assert usage["completion_tokens"] >= 0
print("API check: passed")
PY
```

## 3. 运行完整实验矩阵

```bash
"$PYTHON_BIN" -u -m experiments.experiment_1a \
  --seed-data "$SEED_DATA" \
  --development-data "$DEV_DATA" \
  --development-labels "$DEV_LABELS" \
  --output "$RUN" \
  --base-model "$BASE_MODEL" \
  --model-revision "$MODEL_REVISION" \
  --endpoint "$ALADDIN_ENDPOINT" \
  --model "$ALADDIN_MODEL" \
  --strategies seed-only fixed-prompt self-instruct \
  --seeds 17 23 29 \
  --seed-group-limit 32 \
  --development-group-limit 64 \
  --development-selection-seed 101 \
  --generation-token-budget 24000 \
  --max-candidates-per-strategy 512 \
  --max-tokens 512 \
  --temperature 0.7 \
  --train-records 256 \
  --synthetic-train-records 128 \
  --train-token-budget 100000 \
  --max-steps 1000 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --eval-batch-size 8 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  2>&1 | tee "$RUN/run.log"
```

## 4. 汇总结果

```bash
"$PYTHON_BIN" -m scripts.analyze_experiment_1a "$RUN/result.json" \
  | tee "$RUN/summary.txt"
```

## 5. 检查运行状态

```bash
"$PYTHON_BIN" - "$RUN/result.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print("status:", result.get("status"))
print("results:", len(result.get("results", [])), "/ 9")
print("final_test_accessed:", result.get("data", {}).get("final_test_accessed"))
PY
```

确认输出为 `status: completed`、`results: 9 / 9`、`final_test_accessed: False`。汇总文件为 `$RUN/summary.txt`；逐条件结果、候选、训练数据和 LoRA adapter 保存在 `$RUN/` 下。
