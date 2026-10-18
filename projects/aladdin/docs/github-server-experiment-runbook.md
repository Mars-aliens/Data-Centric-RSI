# GitHub 服务器实验执行手册

本文档只适用于 GitHub 总仓库中的 `projects/aladdin/`。服务器实验应从该
子项目目录运行，不应从总仓库根目录运行。

## 实验顺序

按下面顺序执行：

1. 固定 Git commit，完成代码和环境检查。
2. 运行无 GPU 的 HelpSteer2 技术一 Debug。
3. Debug 通过后运行技术一正式的 180 条件历史 replay。
4. 分析 180 条件结果，确认泄漏审计和 `final_test_accessed=false`。
5. 使用冻结的反馈矩阵运行需要 GPU 的 Qwen LoRA 下游验证，共 48 条件。
6. 分别报告技术一 replay 和 Qwen 下游结果，不合并成一个指标。

集成管线是独立的工程验证，不能替代上述技术一实验，也不能把它的 smoke
结果作为生成模型效果结果。

## 1. 检出 GitHub 项目

从总仓库根目录进入 Aladdin 子项目：

```bash
git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/aladdin

# 使用 GitHub main 的最新代码，并记录实际运行 commit。
git checkout main
git pull --ff-only origin main
git status --short
git rev-parse HEAD
```

如果仓库已经存在，执行 `git pull --ff-only origin main` 后再记录
`git rev-parse HEAD`。实验报告必须包含这个 commit。

数据集、模型、日志和结果必须放在 `projects/aladdin/` 之外。

## 2. 安装和代码检查

技术一历史 replay 只需要 CPU Python 环境，不需要 GPU、Transformers、
vLLM 或模型 endpoint。

```bash
cd /path/to/Data-Centric-RSI/projects/aladdin

export PYTHON_BIN=/path/to/aladdin-python
export PYTHONPATH="$PWD"

"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pytest local_experiments/tests core_system/tests -q
"$PYTHON_BIN" -m compileall -q core_system local_experiments scripts
"$PYTHON_BIN" scripts/check_aladdin_install.py
```

预期结果是测试全部通过、安装检查显示 `Aladdin install OK`。如果这一步失败，
不要开始正式实验。

## 3. 准备技术一数据

需要提供规范化的 HelpSteer2 train JSONL，且必须与以下 revision 和 SHA-256
一致：

```text
dataset: nvidia/HelpSteer2
revision: 990b2711a36180dd19d9c94b8627844866f8982a
sha256: 5bd88e03cb93527131d2a75b30940ea8a969a23096f8f529eb0e9bb114f4d6e9
```

在服务器上设置路径并核对哈希：

```bash
export DATA_PATH=/external/data/helpsteer2-train.jsonl
export RUN_ROOT=/external/runs/aladdin-technique-one
mkdir -p "$RUN_ROOT"

sha256sum "$DATA_PATH"
```

只有 SHA-256 完全一致时才继续。不要把数据集复制进 Git 仓库，也不要使用
`final-test` 数据文件。

## 4. 技术一 Debug：无 GPU

先运行小规模真实数据 Debug：

```bash
cd /path/to/Data-Centric-RSI/projects/aladdin
export PYTHONPATH="$PWD"

"$PYTHON_BIN" -u \
  -m local_experiments.aladdin.active_feedback_helpsteer2 \
  --data "$DATA_PATH" \
  --output "$RUN_ROOT/technique-one-debug.json" \
  --split-seeds 17 \
  --policy-seeds 17 18 \
  --seed-group-limit 32 \
  --query-group-limit 32 \
  --selection-group-limit 64 \
  --heldout-group-limit 64 \
  --feedback-time-budget-seconds 300 \
  --max-queries 8 \
  --selection-budget 16 \
  2>&1 | tee "$RUN_ROOT/technique-one-debug.log"
```

检查 Debug 输出：

```bash
"$PYTHON_BIN" - "$RUN_ROOT/technique-one-debug.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print("status:", report.get("status"))
print("results:", len(report.get("results", [])))
print("final_test_accessed:", report.get("final_test_accessed"))
PY
```

必须确认 `status` 为 `completed`，结果非空，并且
`final_test_accessed` 为 `False`。

## 5. 技术一正式实验：180 条件

Debug 通过后运行冻结矩阵：

```bash
cd /path/to/Data-Centric-RSI/projects/aladdin
export PYTHONPATH="$PWD"
export TECH1_RUN="$RUN_ROOT/technique-one-180.json"

"$PYTHON_BIN" -u \
  -m local_experiments.aladdin.active_feedback_helpsteer2 \
  --data "$DATA_PATH" \
  --output "$TECH1_RUN" \
  --split-seeds 17 23 29 37 \
  --policy-seeds 17 18 19 \
  --target-profiles helpfulness-only correctness-only coherence-only \
  --seed-group-limit 64 \
  --query-group-limit 256 \
  --selection-group-limit 512 \
  --heldout-group-limit 512 \
  --feedback-time-budget-seconds 1800 \
  --max-queries 48 \
  --selection-budget 64 \
  --random-reserve-fraction 0.25 \
  --tie-margin 0.0 \
  2>&1 | tee "$RUN_ROOT/technique-one-180.log"
```

条件数为：

```text
4 split seeds x 3 policy seeds x 3 target profiles x 5 strategies = 180
```

正式结果分析：

```bash
export TECH1_ANALYSIS="$RUN_ROOT/technique-one-180-analysis.json"

"$PYTHON_BIN" -u \
  -m local_experiments.aladdin.analyze_active_feedback_helpsteer2 \
  "$TECH1_RUN" \
  --output "$TECH1_ANALYSIS" \
  | tee "$RUN_ROOT/technique-one-180-summary.txt"
```

正式结果必须记录：完成条件数、数据 SHA-256、各策略指标、配对差异、置信区间、
泄漏审计和 `final_test_accessed`。查看 held-out 结果后不得修改参数并复用同一
运行作为确认性结果。

## 6. Qwen LoRA 下游验证：48 条件

这一阶段需要 GPU 和 `requirements-qwen.txt` 对应的 Python 环境。它使用技术一
生成的冻结反馈矩阵，不重新生成矩阵，不修改 split seed、policy seed、预算或
选择结果。

```bash
cd /path/to/Data-Centric-RSI/projects/aladdin

export QWEN_PY=/path/to/qwen-python
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export DATA_PATH=/external/data/helpsteer2-seed-development-clean.jsonl
export MATRIX_PATH="$TECH1_RUN"
export BASE_MODEL=/external/models/Qwen2.5-7B-Instruct
export QWEN_RUN="$RUN_ROOT/helpsteer2-qwen-lora-48.json"

"$QWEN_PY" -m pip install -r requirements-qwen.txt

"$QWEN_PY" -u \
  -m local_experiments.aladdin.helpsteer2_qwen_lora_matrix \
  --data "$DATA_PATH" \
  --matrix "$MATRIX_PATH" \
  --output "$QWEN_RUN" \
  --base-model "$BASE_MODEL" \
  --split-seeds 17 23 29 37 \
  --policy-seeds 17 18 19 \
  --target-profile helpfulness-only \
  --max-steps 150 \
  --batch-size 2 \
  --eval-batch-size 16 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --max-gpu-memory-gib 60 \
  2>&1 | tee "$RUN_ROOT/helpsteer2-qwen-lora-48.log"
```

条件数为：

```text
4 split seeds x 3 policy seeds x 4 strategies = 48
```

Qwen 结果分析：

```bash
"$QWEN_PY" -u \
  -m local_experiments.aladdin.analyze_helpsteer2_qwen_lora_matrix \
  "$QWEN_RUN" \
  --output "$RUN_ROOT/helpsteer2-qwen-lora-48-analysis.json" \
  | tee "$RUN_ROOT/helpsteer2-qwen-lora-48-summary.txt"
```

## 7. 返回实验结果

不要提交数据集、模型、checkpoint、日志或完整 JSON 运行结果。将下面的信息
直接发给实验负责人：

```text
repository: Mars-aliens/Data-Centric-RSI
commit: <git rev-parse HEAD>
python: <python --version>
dataset sha256: <sha256sum output>
model/revision: <Qwen model revision, if Qwen stage ran>
technique-one status: completed/failed
technique-one conditions: <completed>/180
qwen status: completed/failed/not-run
qwen conditions: <completed>/48
final_test_accessed: false
output paths: external paths only
summary: link or paste the analyzer summary
```

执行者只报告运行事实。研究结论、参数调整和新实验由负责人统一决定。

## 8. 集成管线（独立工程验证）

只有在技术一两阶段完成并单独归档后，才运行集成管线。它从同一项目目录的
`local_experiments/aladdin/integrated_pipeline/` 启动，使用
`server_preflight.sh` 和 `server_run_round.sh`。该运行可以使用 vLLM endpoint，
但其结果必须单独报告，不能替换技术一 180/48 条件结果。
