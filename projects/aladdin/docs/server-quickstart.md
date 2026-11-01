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

## 4. 配置实验一单 seed rehearsal

```bash
cp configs/experiment-one-single-seed.example.json \
   configs/experiment-one-single-seed.local.json

${EDITOR:-vi} configs/experiment-one-single-seed.local.json
```

修改：

```text
endpoint          Qwen3-8B 服务地址
feedback_endpoint Qwen3-32B 服务地址
base_model        Qwen3-8B 本地模型路径或模型 ID
output            使用独立的 experiment-one rehearsal 输出目录
```

保持：

```text
comparison       overall
run_purpose      pilot
seeds            [17]
rounds           3
feedback_mode    model
feedback_model   Qwen/Qwen3-32B
save_adapters    true
feedback_temperature 0
train_token_budget 100000
max_steps        1000
```

`max_steps` 是训练安全上限，`train_token_budget` 才是每个 adapter 的训练预算。全预算使用 `100000` token 时不能把 `max_steps` 设为 32，否则训练会在达到 token 预算前因安全上限退出。上次 pilot 的 `max_steps=32` 只与其 `2048` token 预算配套。

`comparison: overall` 只运行实验一的两个实验组：`fixed_policy` 和 `recursive_policy`。每组 3 轮，共 6 次任务模型训练。每轮每组生成 32 个候选并收集 16 条 Qwen3-32B 模拟反馈。这是实验一计划预算的单 seed rehearsal，不是多 seed 确认性实验，也不包含 final-test 评价。更新消融和反馈选择对照属于后续实验，不在本配置中。

## 5. 运行实验一单 seed rehearsal

使用训练环境和独立训练 GPU。

```bash
export CUDA_VISIBLE_DEVICES=<训练GPU>
"$PYTHON_BIN" -m scripts.check_model_services \
  --config configs/experiment-one-single-seed.local.json

"$PYTHON_BIN" -m experiments.rsi \
  --config configs/experiment-one-single-seed.local.json

"$PYTHON_BIN" -m scripts.analyze_feedback_rsi \
  runs/experiment-one-single-seed/result.json \
  --output runs/experiment-one-single-seed/analysis.json
```

检查结果：

```bash
"$PYTHON_BIN" - "runs/experiment-one-single-seed/result.json" <<'PY'
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

rehearsal 必须满足：`status: completed`、`comparison: overall`、恰有 `fixed_policy` 和 `recursive_policy` 两个 arm 且各有 3 轮、每轮训练状态为 `completed`、每轮每组完成 16 条模型反馈、`feedback_mode: model`、`feedback_model: Qwen/Qwen3-32B`、`final_test_accessed: False`。同时确认递归组的策略更新记录及下一轮输入，固定组的策略版本保持不变。

pilot 失败时查看：

```bash
tail -n 100 runs/experiment-one-single-seed/seed-17/feedback/round-0.model-attempts.jsonl
find runs/experiment-one-single-seed -type f | sort
```

## 6. 中断恢复

```bash
"$PYTHON_BIN" -m experiments.rsi \
  --config configs/experiment-one-single-seed.local.json \
  --resume
```

## 7. 实验一 rehearsal 结束

确认 `result.json` 的 `status` 为 `completed` 后，检查两个组、6 个组轮次、反馈请求、策略更新和 GPU 训练日志。该单 seed 结果用于验证实验一能否按计划执行，不作效果显著性或最终结论。通过后再开展实验二消融和实验三反馈选择对照。确认性实验使用冻结配置和多个预先指定的 seed；final-test 只在确认性运行完成并冻结分析方案后单独执行。
