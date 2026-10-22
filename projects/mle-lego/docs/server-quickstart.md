# 技术一服务器执行步骤

按顺序执行。所有 `<...>` 都要替换成当前服务器的实际值。模型、数据、密钥和运行结果不要上传 GitHub。

## 1. 获取代码和配置环境

```bash
git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/mle-lego
git checkout <要运行的commit>

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

按服务器驱动从 PyTorch 官网选择 CUDA wheel，然后安装：

```bash
python -m pip install torch==2.7.1 --index-url <PyTorch选择器给出的CUDA源>
python -m pip install -r requirements-research.txt
python -m pip check
python -m pytest -q
```

设置项目路径：

```bash
export LEGO_ROOT="$PWD"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export LEGO_TASK_MODEL_DEVICE=cuda
mkdir -p "$LEGO_ROOT/server-private" "$LEGO_ROOT/data" "$LEGO_ROOT/runs"
bash scripts/check_lego_install.sh
nvidia-smi
```

## 2. 准备数据

下载固定 revision，并导出 runner 要求的字段：

```bash
"$TRAIN_PY" - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

dataset_id = "nvidia/HelpSteer2"
revision = "990b2711a36180dd19d9c94b8627844866f8982a"
fields = (
    "prompt", "response", "helpfulness", "correctness", "coherence",
    "complexity", "verbosity",
)
outputs = {
    "train": Path("data/helpsteer2-train.jsonl"),
    "validation": Path("data/helpsteer2-validation.jsonl"),
}

for split, path in outputs.items():
    dataset = load_dataset(dataset_id, revision=revision, split=split)
    missing = set(fields) - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split} is missing columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in dataset:
            item = {field: row[field] for field in fields}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"{split} is empty")
    print(f"{split}: {count} rows -> {path}")
PY
```

`validation` 是锁定的 final-test split；只供可信 runner 做离线汇总，不能提供给 Decision LLM 或候选。生成完整流 manifest：

```bash
"$TRAIN_PY" scripts/prepare_long_horizon_task.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --output-dir "$LEGO_ROOT/runs/data-audit" \
  --stream-seed 17 --groups-per-batch 50 --feedback-delay 2
```

确认 development 数据可用且没有重复 prompt group，保存 manifest SHA256。数据审计通过后继续配置 Task Model 并运行固定流水线 smoke；T1 候选隔离 smoke 在后续执行手册中运行。

## 3. 配置 Task Model

```bash
cp code/technologies/technology_one/task_model.qwen.example.json \
  "$LEGO_ROOT/server-private/task-model-smoke.json"
```

编辑 JSON：填写本机 Qwen3-8B 权重绝对路径和不可变 revision；smoke 的 `max_steps` 设为 `1`。GPU 编号按 `nvidia-smi` 实际空闲情况填写：

```bash
export CUDA_VISIBLE_DEVICES=<stable使用的GPU编号>
```

## 4. 跑固定流水线 smoke

这一步只验证真实数据、Task Model 预测、延迟反馈释放和 parent-checkpoint 续训，不包含 T1 候选或 Docker 隔离。T1 的 stable/candidate 配对 smoke 在下一份执行手册中另行运行，目的不同。

```bash
"$TRAIN_PY" scripts/run_long_horizon_qwen.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --config-json "$LEGO_ROOT/server-private/task-model-smoke.json" \
  --output-dir "$LEGO_ROOT/runs/baseline-smoke/b0-seed-17" \
  --run-id b0-seed-17 --seed 17 --stream-seed 17 \
  --groups-per-batch 50 --feedback-delay 2 --max-batches 4
```

通过条件：`status=complete`、`completed_batches=4`，第 4 批使用到期反馈训练，`final_test_used=false`。失败就停止排查。此步骤只验证固定流水线，不是 T1 候选隔离测试。

## 5. 配置 Decision LLM 并做 API smoke

复制 `code/technologies/technology_one/decision_model.qwen.example.json` 到 `server-private/decision-model.json`，填写服务商确认的 API model ID。然后设置真实 endpoint 和密钥：

```bash
export LEGO_POLICY_API_URL="<OpenAI兼容Chat Completions地址>"
export LEGO_POLICY_API_KEY="<服务器密钥管理器注入的密钥>"
export LEGO_POLICY_TIMEOUT_SECONDS=600

"$TRAIN_PY" scripts/smoke_decision_endpoint.py \
  --profile "$LEGO_ROOT/server-private/decision-model.json"
```

通过条件：`status=ok`、`schema_valid=true`。密钥不要写进 JSON 或提交 Git。

## 6. 转入技术一执行手册

完成本文件第 1～5 步后，环境、数据、Task Model、固定流水线 smoke 和 Decision LLM API smoke 已准备好。第 4 步的结果位于 `$LEGO_ROOT/runs/baseline-smoke/b0-seed-17/`。接下来直接阅读并执行[技术一执行手册](technology-one-safe-evolution-runbook.md)，完成 T1 clean smoke 的 `4/4`。

执行手册执行到 T1 clean smoke 的 `4/4` 后停止。检查 stable/candidate 状态、反馈释放、checkpoint 和 `final_test_used=false`，确认无误后再等待后续安排。

运行日志、manifest、summary、状态文件和配置哈希保存在 `runs/`；不要把数据、模型、checkpoint、密钥或私有配置提交到 Git。
