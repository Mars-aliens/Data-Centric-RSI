# Aladdin Linux 运行步骤

## 1. 安装

```bash
set -euo pipefail

git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/aladdin

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install torch==<PyTorch版本> --index-url <PyTorch CUDA wheel源>
python -m pip install -r requirements-qwen.txt

export ALADDIN_ROOT="$PWD"
export PYTHON_BIN="$ALADDIN_ROOT/.venv/bin/python"
export PYTHONPATH="$ALADDIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m pip check
python -m pytest -q
python -m compileall -q experiments scripts tests
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
nvidia-smi
```

## 2. 准备数据

```bash
mkdir -p data runs

"$PYTHON_BIN" -m scripts.download_datasets \
  --dataset prism --confirm-research-terms

"$PYTHON_BIN" -m scripts.prepare_prism --split-seed 17

"$PYTHON_BIN" -m scripts.audit_datasets --dataset prism \
  | tee runs/prism-audit.json
```

## 3. 启动模型服务

服务环境未安装 vLLM 时执行：

```bash
python3.11 -m venv .venv-service
source .venv-service/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==<PyTorch版本> --index-url <PyTorch CUDA wheel源>
python -m pip install -r requirements-vllm.txt
python -m pip check
```

终端一：Qwen3-8B。

```bash
export CUDA_VISIBLE_DEVICES=<8B服务GPU>

vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --served-model-name Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000
```

终端二：Qwen3-32B。

```bash
export CUDA_VISIBLE_DEVICES=<32B服务GPU列表>

vllm serve Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --served-model-name Qwen/Qwen3-32B \
  --tensor-parallel-size <GPU数量> \
  --host 0.0.0.0 \
  --port 8001
```

## 4. 配置

```bash
cp configs/feedback-rsi-pilot.example.json \
   configs/feedback-rsi.local.json

${EDITOR:-vi} configs/feedback-rsi.local.json
```

修改：

```text
endpoint          Qwen3-8B 服务地址
feedback_endpoint Qwen3-32B 服务地址
base_model       Qwen3-8B 本地模型路径或模型 ID
output            本次运行的输出目录
```

保持：

```text
comparison       overall
feedback_mode    model
feedback_model   Qwen/Qwen3-32B
save_adapters    true
feedback_temperature 0
```

## 5. 运行小规模 pilot

使用训练环境和独立训练 GPU。

```bash
export CUDA_VISIBLE_DEVICES=<训练GPU>
"$PYTHON_BIN" -m scripts.check_model_services \
  --config configs/feedback-rsi.local.json

"$PYTHON_BIN" -m experiments.rsi \
  --config configs/feedback-rsi.local.json

"$PYTHON_BIN" -m scripts.analyze_feedback_rsi \
  runs/feedback-rsi-pilot/result.json \
  --output runs/feedback-rsi-pilot/analysis.json
```

检查结果：

```bash
"$PYTHON_BIN" - "runs/feedback-rsi-pilot/result.json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("feedback_mode:", data.get("configuration", {}).get("feedback_mode"))
print("feedback_model:", data.get("configuration", {}).get("feedback_model"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    print(name, "rounds:", len(arm.get("rounds", [])), "summary:", data.get("summary", {}).get(name))
PY
```

pilot 必须满足：`status: completed`、`comparison: overall`、两个 arm 各有 2 轮、`feedback_mode: model`、`feedback_model: Qwen/Qwen3-32B`、`final_test_accessed: False`。

pilot 失败时查看：

```bash
tail -n 100 runs/feedback-rsi-pilot/seed-17/feedback/round-0.model-attempts.jsonl
find runs/feedback-rsi-pilot -type f | sort
```

## 6. 中断恢复

```bash
"$PYTHON_BIN" -m experiments.rsi \
  --config configs/feedback-rsi.local.json \
  --resume
```

## 7. pilot 结束

确认 `result.json` 的 `status` 为 `completed` 后，查看 `analysis.json` 和日志，再决定是否扩大实验。
