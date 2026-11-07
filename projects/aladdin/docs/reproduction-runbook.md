# Aladdin 实验运行命令

所有命令在 Linux 服务器的 `Data-Centric-RSI/projects/aladdin/` 目录执行。下面从环境和数据已按 `server-quickstart.md` 准备完成开始，先运行实验一至实验三的 smoke。

## 当前实验顺序

| 顺序 | 实验 | 当前配置状态 |
|---|---|---|
| 1 | 整体效果：固定策略 vs 完整递归策略 | 同一实验的两种运行规模：Smoke 验证 `experiment-one-smoke.example.json`；完整 Pilot `experiment-one-single-seed.example.json` |
| 2 | 更新机制：合成/清洗/选择的 8 组因子实验 | smoke：`experiment-two-factorial-smoke.example.json`；pilot：`experiment-two-factorial.example.json` |
| 3 | 反馈效率：固定候选池下的三种选择策略 | smoke：`experiment-three-selection-smoke.example.json`；固定池 pilot：`experiment-three-selection-fixed-pool.example.json` |
| 4 | 少样本适用性：低/中/高初始数据量 | 实验设计和 runner 已准备；当前不运行 smoke |
| 5 | 需求逐步明确：分轮释放新要求 | 实验设计和 runner 已准备；当前不运行 smoke |

研究实验一只有一个，比较固定策略与完整递归策略。`experiment-one-smoke.example.json` 和 `experiment-one-single-seed.example.json` 是同一实验的两种运行规模：前者是小预算 Smoke，用于检查执行链路；后者是完整 Pilot，用于检查计划预算下的端到端运行。当前先运行实验一至实验三各自的 Smoke；实验一至实验三的 Smoke 只验证执行链路、数据产物和指标生成，不支持方法效果结论。实验四和实验五暂不纳入本轮 Smoke 执行。

**执行顺序：**先运行本文件中的“实验一 Smoke 验证”“实验二 Smoke”“实验三 Smoke”三节。三项 Smoke 全部通过后，再运行实验一完整 Pilot、实验二完整因子实验和实验三完整反馈选择。实验四、实验五暂不运行，待其正式实验方案和验收标准冻结后再补充执行命令。

## 环境、数据和模型服务

先完整执行 [server-quickstart.md](server-quickstart.md) 第 1 至第 3 步：安装训练环境和依赖、下载并准备 PRISM、启动 Qwen3-8B 和 Qwen3-32B 服务。`server-quickstart.md` 中的 `<PyTorch版本>`、`<PyTorch CUDA wheel源>`、GPU 编号及模型服务 GPU 数量必须替换成服务器实际值。

在训练环境终端进入项目目录并设置变量：

```bash
cd Data-Centric-RSI/projects/aladdin
source .venv/bin/activate
export ALADDIN_ROOT="$PWD"
export PYTHON_BIN="$ALADDIN_ROOT/.venv/bin/python"
export PYTHONPATH="$ALADDIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="<模型服务要求的API密钥；本机服务不校验时填任意非空值>"
export CUDA_VISIBLE_DEVICES="<空闲训练GPU编号>"
```

先确认数据文件和 GPU 可用：

```bash
test -s data/processed/prism/seed.jsonl
test -s data/processed/prism/development.jsonl
test -s data/processed/prism/labels/development-labels.jsonl
nvidia-smi
```

两个模型服务必须从训练服务器可访问。若服务与训练进程在同一台机器，示例配置中的 `127.0.0.1:8000` 和 `127.0.0.1:8001` 可直接使用；若服务在别的机器，把每个 smoke 的本地配置中 `endpoint`、`feedback_endpoint` 改成对应服务地址。`base_model` 必须是训练服务器能读取的 Qwen3-8B 模型 ID 或本地模型目录。配置中的 `output` 指向本次运行专用目录；重跑时改成一个新的目录名，不能复用已有的非空输出目录。

以下每个实验都按“复制配置、检查/修改配置、检查服务、运行、分析、验收”顺序执行。复制命令只在首次准备该 smoke 配置时执行；已有 `.local.json` 时保留并继续使用，不要覆盖本地配置。

## 实验一 Smoke 验证：单 seed、1 轮、小预算

复制独立 smoke 配置。首次运行时将三个服务/路径字段改成服务器实际值：`endpoint`、`feedback_endpoint`、`base_model`。若输出目录已经存在，在配置中把 `output` 改成新的目录：

```bash
test -e configs/experiment-one-smoke.local.json || cp configs/experiment-one-smoke.example.json configs/experiment-one-smoke.local.json
${EDITOR:-vi} configs/experiment-one-smoke.local.json
```

运行并生成分析文件：

```bash
export CUDA_VISIBLE_DEVICES=<训练GPU>

"$PYTHON_BIN" -m scripts.check_model_services \
  --config configs/experiment-one-smoke.local.json

"$PYTHON_BIN" -m experiments.rsi \
  --config configs/experiment-one-smoke.local.json

"$PYTHON_BIN" -m scripts.analyze_feedback_rsi \
  runs/experiment-one-smoke/result.json \
  --output runs/experiment-one-smoke/analysis.json
```

Smoke 完成条件：`status=completed`、`comparison=overall`、两个 arm 各完成 1 轮训练、每 arm 收到 2 条模型反馈、训练状态为 `completed`、`final_test_accessed=false`。检查结果：

```bash
"$PYTHON_BIN" - "runs/experiment-one-smoke/result.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    print(name, "rounds:", len(arm.get("rounds", [])),
          "feedback:", [row.get("completed_feedback_events") for row in arm.get("rounds", [])],
          "training:", [row.get("training", {}).get("status") for row in arm.get("rounds", [])])
PY
```

Smoke 未通过时，先检查 `runs/experiment-one-smoke/` 下的 `result.json`、各 arm 的 `round-0/` 产物和 `seed-17/feedback/round-0.model-attempts.jsonl`；修复后在本地配置中设置新的 `output` 目录再重跑，不复用不完整目录。

## 实验二 smoke：8 组、1 seed、1 轮、小预算

```bash
test -e configs/experiment-two-factorial-smoke.local.json || cp configs/experiment-two-factorial-smoke.example.json configs/experiment-two-factorial-smoke.local.json
${EDITOR:-vi} configs/experiment-two-factorial-smoke.local.json

"$PYTHON_BIN" -m scripts.check_model_services --config configs/experiment-two-factorial-smoke.local.json
"$PYTHON_BIN" -m experiments.rsi --config configs/experiment-two-factorial-smoke.local.json
"$PYTHON_BIN" -m scripts.analyze_feedback_rsi \
  runs/experiment-two-factorial-smoke/result.json \
  --output runs/experiment-two-factorial-smoke/analysis.json
```

完成条件：`status=completed`、`comparison=factorial`、8 个 arm 各完成 1 轮、每 arm 收到 2 条反馈、训练完成、`final_test_accessed=false`，且 analysis 含 `factorial_update_effects`。这只验证八组执行和分析管线，不检验统计效应。

打印关键结果：

```bash
"$PYTHON_BIN" - "runs/experiment-two-factorial-smoke/result.json" "runs/experiment-two-factorial-smoke/analysis.json" <<'PY'
import json, sys
run = json.load(open(sys.argv[1], encoding="utf-8"))
analysis = json.load(open(sys.argv[2], encoding="utf-8"))
print("status:", run.get("status"), "comparison:", run.get("comparison"))
print("arms:", len(run.get("arms", {})), "analysis keys:", list(analysis))
print("final_test_accessed:", run.get("data", {}).get("final_test_accessed"))
PY
```

## 实验三 smoke：固定候选池、3 组、1 seed、1 轮、小预算

```bash
test -e configs/experiment-three-selection-smoke.local.json || cp configs/experiment-three-selection-smoke.example.json configs/experiment-three-selection-smoke.local.json
${EDITOR:-vi} configs/experiment-three-selection-smoke.local.json

"$PYTHON_BIN" -m scripts.check_model_services --config configs/experiment-three-selection-smoke.local.json
"$PYTHON_BIN" -m experiments.rsi --config configs/experiment-three-selection-smoke.local.json
"$PYTHON_BIN" -m scripts.analyze_feedback_rsi \
  runs/experiment-three-selection-smoke/result.json \
  --output runs/experiment-three-selection-smoke/analysis.json
```

完成条件：`status=completed`、`comparison=selection`、三个选择 arm 各完成 1 轮、每 arm 收到 2 条反馈、训练完成、`final_test_accessed=false`。三组首轮共享候选池；这只验证选择实验管线，不用于判断哪种策略更好。

打印关键结果：

```bash
"$PYTHON_BIN" - "runs/experiment-three-selection-smoke/result.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("status:", data.get("status"), "comparison:", data.get("comparison"))
print("arms:", sorted(data.get("arms", {})))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    print(name, "rounds:", len(arm.get("rounds", [])), "feedback:", [r.get("completed_feedback_events") for r in arm.get("rounds", [])])
PY
```

## 实验一完整 Pilot：单 seed、完整计划预算

实验一 Smoke 通过后，按 [server-quickstart.md](server-quickstart.md) 第 4 步创建 `configs/experiment-one-single-seed.local.json`，确认使用完整计划预算：3 轮、每组每轮 32 个候选和 16 条反馈、每 adapter 100000 训练 token。它仍是实验一，不是新增的实验。

```bash
.venv/bin/python -m scripts.check_model_services \
  --config configs/experiment-one-single-seed.local.json

.venv/bin/python -m experiments.rsi \
  --config configs/experiment-one-single-seed.local.json

.venv/bin/python -m scripts.analyze_feedback_rsi \
  runs/experiment-one-single-seed/result.json \
  --output runs/experiment-one-single-seed/analysis.json
```

配置文件 `configs/experiment-one-single-seed.example.json` 使用实验一的 `overall` 对照，仅设置一个 seed（17）。它包含 `fixed_policy` 和 `recursive_policy` 两个实验组、3 轮、每轮每组 32 个候选和 16 条 Qwen3-32B 模拟反馈；每个 adapter 的训练 token 预算为 100000，`max_steps=1000` 是安全上限。此完整 Pilot 验证实验一的计划预算能否执行，不是多 seed 确认性结果，也不运行 final-test。

检查：

```bash
.venv/bin/python - "runs/experiment-one-single-seed/result.json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("feedback_model:", data.get("configuration", {}).get("feedback_model"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    rounds = arm.get("rounds", [])
    print(name, "rounds:", len(rounds), "training:", [row["training"].get("status") for row in rounds])
PY
```

完成条件：`status` 为 `completed`、`comparison` 为 `overall`、结果含 2 个 arm 且每个 arm 完成 3 轮、所有训练状态为 `completed`、每轮每组完成 16 条模型反馈，并且 `final_test_accessed` 为 `false`。

## 中断恢复

```bash
.venv/bin/python -m experiments.rsi \
  --config configs/experiment-one-single-seed.local.json \
  --resume
```

## 实验一完整 Pilot 结束

确认两个组各 3 轮全部完成后，检查训练预算、反馈尝试日志和递归组策略版本变化。实验一完整 Pilot 通过后，再运行实验二消融和实验三反馈选择；正式实验沿用冻结配置，只把 seeds 替换成预先确定的多 seed 列表。final-test 另行执行。

## 实验二：三项更新的完整因子实验

```bash
.venv/bin/python -m scripts.check_model_services \
  --config configs/experiment-two-factorial.example.json

.venv/bin/python -m experiments.rsi \
  --config configs/experiment-two-factorial.example.json

.venv/bin/python -m scripts.analyze_feedback_rsi \
  runs/experiment-two-factorial/result.json \
  --output runs/experiment-two-factorial/analysis.json
```

该配置包含合成、清洗、选择三种策略更新的全部 8 种开关组合，可估计三个主效应和交互作用。示例配置仍为单 seed pilot；正式运行前依据 pilot 的 seed-level 方差和预先设定的最小有意义差异冻结 seed 数与分析方案。

检查：

```bash
.venv/bin/python - "runs/experiment-two-factorial/result.json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
print("arms:", sorted(data.get("arms", {})))
for name, arm in data.get("arms", {}).items():
    rounds = arm.get("rounds", [])
    print(name, "rounds:", len(rounds), "training:", [row["training"].get("status") for row in rounds])
PY
```

完成条件：`status` 为 `completed`、`comparison` 为 `factorial`、结果含 8 个 arm 且每个 arm 完成 3 轮、所有训练状态为 `completed`，并且 `final_test_accessed` 为 `false`。分析文件应包含 `factorial_update_effects`。

## 实验三：固定候选池下的反馈选择

```bash
.venv/bin/python -m scripts.check_model_services \
  --config configs/experiment-three-selection-fixed-pool.example.json

.venv/bin/python -m experiments.rsi \
  --config configs/experiment-three-selection-fixed-pool.example.json

.venv/bin/python -m scripts.analyze_feedback_rsi \
  runs/experiment-three-selection-fixed-pool/result.json \
  --output runs/experiment-three-selection-fixed-pool/analysis.json
```

检查：

```bash
.venv/bin/python - "runs/experiment-three-selection-fixed-pool/result.json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    rounds = arm.get("rounds", [])
    print(name, "rounds:", len(rounds), "training:", [row["training"].get("status") for row in rounds])
PY
```

完成条件：`status` 为 `completed`、`comparison` 为 `selection`、结果含 `query_random`、`query_uncertainty`、`query_feedback_value` 三个 arm，且每个 arm 完成 1 轮、每 arm 反馈数相同、训练状态为 `completed`，并且 `final_test_accessed` 为 `false`。三组 round-0 的生成候选语义多重集应相同；不相同则本次不能用于选择策略的干净对照。

## 实验四、实验五

实验四和实验五的 runner、配置草案和测试代码已保留在仓库中，但当前不作为 smoke 执行。待两项实验的正式问题、预算、对照组和验收指标冻结后，再补充正式运行命令。

## 支撑验证和最终盲评

清洗审计、Qwen3-32B 与真人反馈一致性、故障恢复、预算/噪声敏感性属于支撑验证，不替代五个主实验。`final_test` 只能在五个主实验的代码、模型、策略、阈值、seed 和分析方法冻结后，通过独立 blind pairwise 流程运行。
