# 贡献者实验执行手册

本手册说明贡献者如何从 GitHub 仓库执行技术一实验并提交可审计证据。Issue 中的任务参数必须来自[技术一实验参数冻结卡](technical-one-experiment-parameter-card.md)。除非 Issue 明确授权修订协议，否则不得更改协议、随机种子、模型版本、接受规则或数据切分。

## 角色

- **维护者：** 负责研究问题、协议、GitHub Issue 和结果解释。
- **贡献者：** 准备服务器、运行指定命令、保存日志与输出，并准确报告结果。
- **审查者：** 核对 commit、配置哈希、数据清单和审计结果，并确认运行是否符合协议、可进行比较。

## 1. 从 GitHub Issue 开始

每次运行都必须对应一个 Issue，包含：

- experiment name and purpose;
- Git commit or tag to execute;
- profile template and private profile revision;
- mode, seeds, batch count, GPU assignment, and output root;
- required smoke test and formal command;
- expected result files and acceptance criteria.

贡献者须在运行报告中记录 Issue 编号。不得从未提交的工作树开展正式实验。

## 2. 准备服务器

克隆或更新仓库，然后记录代码版本：

```bash
git clone <repository-url>
cd <repository-directory>
git checkout <commit-or-tag>
git rev-parse HEAD
```

在 Git 仓库之外准备私有运行环境：

```bash
export LEGO_ROOT="$PWD"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"
export LEGO_POLICY_API_URL="http://127.0.0.1:8000/v1/chat/completions"
export LEGO_POLICY_TIMEOUT_SECONDS=600
export CUDA_VISIBLE_DEVICES=<gpu-id>
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=40
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

将示例配置复制到私有路径，填写本地模型和命令路径，并确保私有配置不进入 Git。启动运行前，校验归一化数据、配置、Decision LLM 端点和 Task Model。

归一化文件属于外部输入；当前安装检查脚本要求它们位于检出目录中以下已忽略路径：

```text
$LEGO_ROOT/data/helpsteer2-train.jsonl
$LEGO_ROOT/data/helpsteer2-validation.jsonl
```

不要提交这些文件。执行前在 Issue 中核验其来源版本和 SHA256。

运行预检：

```bash
mkdir -p "$LEGO_ROOT/server"
"$TRAIN_PY" -m local_experiments.lego.server_preflight \
  --profile "$LEGO_ROOT/server_profile.evolution.qwen.json" \
  --root "$LEGO_ROOT" \
  --report "$LEGO_ROOT/server/preflight.json"

bash "$LEGO_ROOT/scripts/check_lego_install.sh"
```

Decision LLM 服务必须单独测试：

```bash
POLICY_API_ROOT="${LEGO_POLICY_API_URL%/chat/completions}"
curl -fsS --max-time 30 "$POLICY_API_ROOT/models"
"$TRAIN_PY" -m local_experiments.lego.policy_command <<'JSON'
{
  "protocol": "lego-harness-edit-v1",
  "model": {"id": "<pinned-decision-model>", "revision": "<revision>"},
  "prompt": {
    "system": "Return JSON only.",
    "user": "Return a minimal valid proposal with base_harness_version v1, empty edits, and rationale smoke test."
  }
}
JSON
```

仅当服务器上的 vLLM 环境为已配置的 `vllm-serve` 环境时，才使用仓库的 `scripts/check_decision_model.sh`；否则在 Issue 中记录等效的 tokenizer、Transformers、vLLM 和 CUDA 检查结果。

## 3. 冒烟测试

先运行指定的冒烟命令。冒烟运行只检查链路，不构成论文实验依据。确认：

- `/v1/models` exposes the pinned Decision LLM model;
- one policy command returns valid JSON;
- the Task Model and evaluator can load;
- `final_test_used` is `false`;
- the run writes `progress.json`, `summary.json`, and an audit trail.

如果冒烟测试失败，应停止并报告命令、退出状态、相关日志末尾、GPU 分配及配置/模型版本。不得悄悄修改协议以使运行通过。

冒烟命令：

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/<experiment-id>-smoke" \
GPU_ID=<gpu-id> BATCHES=2 SEEDS="17" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

## 4. 正式实验

使用 Issue 指定的命令。技术一正式比较的模式为 `independent` 和 `dependency-aware-joint`：

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/<experiment-id>" \
GPU_ID=<gpu-id> \
BATCHES=<batches> \
SEEDS="17 23 29 31 37 41 43 47 53 59" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

只有在 GPU 和显存相互隔离时才并行运行不同模式，否则串行运行。配置、代码版本、模型版本、随机种子集合或预算发生变化时，必须使用新的输出目录。

运行期间监控 `progress.json` 和日志。拒绝率高本身不代表实验失败；应按 `held_in_regression`、`duplicate_configuration`、提案无效和执行失败分类报告。

## 5. 必需的证据包

实验完成后，在源码仓库之外提交压缩证据包，包含：

- `summary.json` for every completed run;
- `progress.json` and `endurance_state.json`;
- run logs and the final audit output;
- code commit, profile hash, protocol hash, data manifest/hash;
- model IDs and revisions for Task Model and Decision LLM;
- GPU, seed, batch, training, evaluator, and retry settings;
- a short `run-report.md` using the template below.

不要将原始数据集、模型权重、检查点、凭据、日志或 `runs/` 提交到公开仓库。大型证据应上传至约定的产物存储，并只在 Issue 中记录链接、校验和及摘要。

## 6. 运行报告模板

完整报告字段请使用[运行报告模板](run-report-template.md)。

贡献者只报告事实，由维护者判断结果是否支持研究结论。

## 7. 完成与审查

只有在所有指定摘要均已生成、审计通过、`final_test_used` 符合协议且 Issue 已链接证据包后，运行才算完成。随后由维护者通过独立文档提交更新 `docs/current-status.md` 和实验摘要。

如果运行崩溃，保留失败输出和日志。若支持恢复，只能在代码、配置、数据、模型、随机种子和输出目录完全相同的条件下恢复；否则使用新输出目录，并在 Issue 中说明原因。

## 8. 可直接执行的完整流程

本节把上面的原则整理成贡献者可以逐项执行的清单。正式运行必须使用 Issue 指定的 commit、profile、数据和模型；以下尖括号内容必须先替换。

### 8.1 Issue 必填参数

维护者在分配 Issue 前应明确记录：

| 字段 | 内容 |
| --- | --- |
| 目的 | smoke、组件矩阵、固定基线、两种自进化模式或长程验证 |
| 代码 | 精确 commit 或 tag |
| profile | 私有 profile 文件名、版本和 SHA256 |
| 数据 | HelpSteer2 revision、两个 JSONL 的 SHA256、freeze manifest 的 SHA256 |
| 模型 | Task Model 与 Decision LLM 的 ID、revision 和本地路径标识 |
| 参数 | 模式、seed、batch 数、训练组预算、训练/评价预算和 GPU |
| 输出 | 唯一的 `OUTPUT_ROOT`、日志文件和证据归档位置 |
| 完成条件 | 所需 `summary.json` 数量、审计命令和 `final_test` 锁定状态 |

贡献者不得从未提交工作树开展正式运行，也不得在同一个输出目录混合不同 commit、profile、模型或数据版本。

### 8.2 克隆、环境和数据

在 Linux 服务器上执行：

```bash
git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/mle-lego
git checkout <issue-指定的-commit-or-tag>
export LEGO_ROOT="$PWD"
git rev-parse HEAD

python3 -m venv "$LEGO_ROOT/.venv"
source "$LEGO_ROOT/.venv/bin/activate"
python -m pip install -r "$LEGO_ROOT/requirements-server.txt"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"
```

准备以下外部文件，不要提交到 GitHub：

```text
$LEGO_ROOT/data/helpsteer2-train.jsonl
$LEGO_ROOT/data/helpsteer2-validation.jsonl
```

如果需要从锁定 revision 的原始 HelpSteer2 生成归一化文件，先安装研究依赖，再执行：

```bash
python -m pip install -r "$LEGO_ROOT/requirements-research.txt"
mkdir -p "$LEGO_ROOT/data"
"$TRAIN_PY" -m local_experiments.lego.prepare_helpsteer2 \
  "$LEGO_ROOT/data/helpsteer2-train.jsonl" --split train
"$TRAIN_PY" -m local_experiments.lego.prepare_helpsteer2 \
  "$LEGO_ROOT/data/helpsteer2-validation.jsonl" --split validation
```

若维护者已经提供归一化 JSONL，应直接使用提供的版本。执行数据校验并记录 SHA256：

```bash
"$TRAIN_PY" -m local_experiments.lego.validate_data \
  "$LEGO_ROOT/data/helpsteer2-train.jsonl"
"$TRAIN_PY" -m local_experiments.lego.validate_data \
  "$LEGO_ROOT/data/helpsteer2-validation.jsonl"
sha256sum "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
          "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
          "$LEGO_ROOT/code/technologies/technology_one/deployment_freeze_manifest.json"
```

`deployment_freeze_manifest.json` 中的 train 文件 SHA256 必须与实际文件一致。若不一致，停止并报告，不要自行修改 freeze manifest。

### 8.3 profile 和模型检查

示例 profile 在源码中，私有 profile 放在 `LEGO_ROOT` 根目录，且不得提交：

```bash
cp "$LEGO_ROOT/code/technologies/technology_one/server_profile.evolution.qwen.example.json" \
   "$LEGO_ROOT/server_profile.evolution.qwen.json"
cp "$LEGO_ROOT/code/technologies/technology_one/server_profile.baseline_matrix.qwen.example.json" \
   "$LEGO_ROOT/server_profile.baseline_matrix.qwen.json"
```

修改并核对 `dataset`、`decision_model`、`task_model.training_config.base_model_path`、三个 `commands`、`stream` 和 `action_system`。尤其确认：

- `action_system.final_test_unlocked` 为 `false`；
- `candidate_selection_split` 为 `held_in`；
- `regression_split` 为 `held_out`；
- `final_report_split` 为 `final_test`；
- 两个正式演化模式使用相同的模型、数据、seed、batch 和候选预算；
- `task_model` 的本地权重路径存在且是 Issue 指定版本。

```bash
export LEGO_POLICY_API_URL="http://127.0.0.1:8000/v1/chat/completions"
export LEGO_POLICY_TIMEOUT_SECONDS=600
export CUDA_VISIBLE_DEVICES=<GPU编号>
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=40
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$LEGO_ROOT/server" "$LEGO_ROOT/logs"

"$TRAIN_PY" - "$LEGO_ROOT/server_profile.evolution.qwen.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print("decision_model:", p["decision_model"]["model_id"], p["decision_model"]["revision"])
print("task_model:", p["task_model"]["model_id"], p["task_model"]["revision"])
print("base_model_path:", p["task_model"]["training_config"]["base_model_path"])
print("final_test_unlocked:", p["action_system"]["final_test_unlocked"])
print("candidate_selection_split:", p["stream"]["candidate_selection_split"])
print("regression_split:", p["stream"]["regression_split"])
PY
```

### 8.4 预检和 Decision LLM 连通性

```bash
bash "$LEGO_ROOT/scripts/check_lego_install.sh"
"$TRAIN_PY" -m local_experiments.lego.server_preflight \
  --profile "$LEGO_ROOT/server_profile.evolution.qwen.json" \
  --root "$LEGO_ROOT" \
  --report "$LEGO_ROOT/server/preflight-evolution.json"
```

预检报告必须是 `ready_for_server_model_configuration`，且 `errors` 为空、`final_test_unlocked` 为 `false`。然后测试 Decision LLM：

```bash
curl -fsS --max-time 30 \
  "${LEGO_POLICY_API_URL%/chat/completions}/models"

"$TRAIN_PY" -m local_experiments.lego.policy_command <<'JSON'
{
  "protocol": "lego-harness-edit-v1",
  "model": {"id": "<profile中的model_id>", "revision": "<profile中的revision>"},
  "prompt": {
    "system": "Return JSON only.",
    "user": "Return a minimal valid proposal with base_harness_version v1, empty edits, and rationale smoke test."
  }
}
JSON
```

只有服务器确实使用名为 `vllm-serve` 的 Conda 环境时，才执行 `scripts/check_decision_model.sh`。否则记录等效的 tokenizer、Transformers、vLLM 和 CUDA 检查结果即可。

### 8.5 两批次 smoke

演化程序至少需要两个 batch 才会产生 transition。smoke 只检查链路，不进入论文统计：

```bash
export SMOKE_ROOT="$LEGO_ROOT/runs/technology-one-smoke-<issue-id>"
GPU_ID=<GPU编号> BATCHES=2 SEEDS="17" \
MODES="independent dependency-aware-joint" \
OUTPUT_ROOT="$SMOKE_ROOT" \
PROFILE="$LEGO_ROOT/server_profile.evolution.qwen.json" \
COMMAND_RETRIES=3 RETRY_BACKOFF_SECONDS=15 \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh" \
  2>&1 | tee "$LEGO_ROOT/logs/technology-one-smoke-<issue-id>.log"
```

通过条件是两个模式各生成一个完成的 `summary.json`，`decision_llm_used=true`，`final_test_used=false`，并且日志中有 proposal、候选评估和接受或回滚记录。候选被拒绝不等于 smoke 失败；进程崩溃、接口错误、数据错误和 final-test 被使用才是失败。

### 8.6 正式五类实验

完整入口脚本默认跳过组件矩阵，必须显式设置 `RUN_COMPONENT_MATRIX=1`。推荐使用一个新的输出根：

```bash
export FORMAL_ROOT="$LEGO_ROOT/runs/technology-one-formal-<issue-id>"
export SEEDS="17 23 29 31 37 41 43 47 53 59"
export BATCHES=10

RUN_COMPONENT_MATRIX=1 \
COMPONENT_ROOT="$FORMAL_ROOT/component-matrix" \
DECISION_ROOT="$FORMAL_ROOT/decision-baselines" \
EVOLUTION_ROOT="$FORMAL_ROOT/evolution-comparison" \
PROFILE_BASELINE="$LEGO_ROOT/server_profile.baseline_matrix.qwen.json" \
PROFILE_EVOLUTION="$LEGO_ROOT/server_profile.evolution.qwen.json" \
GPU_ID=<GPU编号> SEEDS="$SEEDS" BATCHES="$BATCHES" \
bash "$LEGO_ROOT/scripts/run_technology_one_complete_batch.sh" \
  2>&1 | tee "$LEGO_ROOT/logs/technology-one-formal-<issue-id>.log"
```

该脚本串行运行，实际目录为：

```text
$FORMAL_ROOT/component-matrix/<run>/summary.json
$FORMAL_ROOT/decision-baselines/fixed-harness-react/<run>/summary.json
$FORMAL_ROOT/decision-baselines/react-reflexion/<run>/summary.json
$FORMAL_ROOT/evolution-comparison/independent/<run>/summary.json
$FORMAL_ROOT/evolution-comparison/dependency-aware-joint/<run>/summary.json
```

只运行已完成前置证据的两种演化模式时：

```bash
OUTPUT_ROOT="$FORMAL_ROOT/evolution-comparison" \
PROFILE="$LEGO_ROOT/server_profile.evolution.qwen.json" \
GPU_ID=<GPU编号> BATCHES=10 COMMAND_RETRIES=3 RETRY_BACKOFF_SECONDS=15 \
SEEDS="17 23 29 31 37 41 43 47 53 59" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

正式模式名称必须是 `independent` 和 `dependency-aware-joint`。旧的 `dependency-aware` 允许 generator-only proposal，只能作为历史 pilot，不能替代正式联合模式。

### 8.7 GPU 排队、进度和日志

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
pgrep -af 'server_technical_one|server_baseline_matrix|policy_command|qwen_task_model_command|qwen_evaluator_command' || true
```

需要等待 GPU 时，在启动实验前执行：

```bash
GPU_ID=<GPU编号>
MIN_FREE_MIB=65000
while true; do
  FREE_MIB=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date) GPU${GPU_ID} free=${FREE_MIB} MiB"
  if [ "$FREE_MIB" -ge "$MIN_FREE_MIB" ] && \
     ! pgrep -f '[s]erver_technical_one|[s]erver_baseline_matrix' >/dev/null; then
    break
  fi
  sleep 60
done
```

将 `RUN_ROOT` 设置为具体 run 目录后查询进度：

```bash
export RUN_ROOT="$FORMAL_ROOT/evolution-comparison/independent/<具体run-id>"
"$TRAIN_PY" - "$RUN_ROOT/progress.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print("status:", p.get("status"))
print("progress:", p.get("completed_batches"), "/", p.get("target_batches"))
print("phase:", p.get("phase"))
print("accepted/rejected:", p.get("accepted"), "/", p.get("rejected"))
print("retries:", p.get("reliability", {}).get("command_retry_count"))
PY
tail -n 100 "$LEGO_ROOT/logs/<run-id>.log"
```

`rejected` 不是自动故障。必须根据 transition 的 `reason` 分类统计 `held_in_regression`、`duplicate_configuration`、无效 proposal 和执行失败。

### 8.8 失败、恢复和完成判定

API 超时、OOM、命令非零退出或数据/profile 校验失败时，保留 `progress.json`、summary（若有）和完整日志。只有代码、profile、数据、模型、seed、batch、预算和输出目录完全相同，才使用支持 resume 的 runner 恢复；否则新建 `RUN_ID` 和输出根。

```bash
RESUME=1 PROFILE="$LEGO_ROOT/server_profile.evolution.qwen.json" \
OUTPUT_DIR="$FORMAL_ROOT/evolution-comparison/independent" \
RUN_ID="<原run-id>" SEED=<原seed> BATCHES=<原batch数> \
PROPOSAL_MODE=independent \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution.sh" \
  > "$LEGO_ROOT/logs/<原run-id>-resume.log" 2>&1
```

不得手工修改 `progress.json`、`summary.json`、`endurance_state.json` 或 transition。某个 seed 已有 complete summary 时，批处理器会跳过它；先确认 summary 真的完成，不能以旧 incomplete 结果冒充完成。

## 9. 汇总与完整审计

先检查五类输出数量：

```bash
for ROOT in \
  "$FORMAL_ROOT/component-matrix" \
  "$FORMAL_ROOT/decision-baselines/fixed-harness-react" \
  "$FORMAL_ROOT/decision-baselines/react-reflexion" \
  "$FORMAL_ROOT/evolution-comparison/independent" \
  "$FORMAL_ROOT/evolution-comparison/dependency-aware-joint"; do
  printf '%s: ' "$ROOT"
  find "$ROOT" -name summary.json -print 2>/dev/null | wc -l
done
```

运行组件矩阵和两种演化模式的专项审计：

```bash
"$TRAIN_PY" "$LEGO_ROOT/scripts/audit_technical_one_matrix.py" \
  "$FORMAL_ROOT/component-matrix" --json \
  > "$FORMAL_ROOT/audit-component-matrix.json"

for MODE in independent dependency-aware-joint; do
  "$TRAIN_PY" "$LEGO_ROOT/scripts/audit_technology_one_evolution.py" \
    "$FORMAL_ROOT/evolution-comparison/$MODE" \
    --seeds 17 23 29 31 37 41 43 47 53 59 \
    --batches "$BATCHES" --json \
    > "$FORMAL_ROOT/audit-$MODE.json"
done
```

最后运行完整五类审计：

```bash
"$TRAIN_PY" "$LEGO_ROOT/scripts/audit_technology_one_complete.py" \
  --component-root "$FORMAL_ROOT/component-matrix" \
  --independent-root "$FORMAL_ROOT/evolution-comparison/independent" \
  --dependency-root "$FORMAL_ROOT/evolution-comparison/dependency-aware-joint" \
  --fixed-root "$FORMAL_ROOT/decision-baselines/fixed-harness-react" \
  --reflexion-root "$FORMAL_ROOT/decision-baselines/react-reflexion" \
  --output "$FORMAL_ROOT/audit-technology-one-complete.json"
```

审计必须为 `PASS`。它会检查十个 seed、五类 evidence、两种演化模式的预算一致性、每个联合模式 seed 至少一次联合 proposal，以及所有正式 summary 的 `final_test_used=false`。审计通过只说明证据满足协议，不等于某方法已经优于其他方法。

## 10. 证据归档和 final-test

证据包放在仓库之外，至少包括每个 run 的 `summary.json`、`progress.json`、`endurance_state.json`（若生成）、日志、预检报告、专项审计、完整审计、代码/profile/数据校验和、模型版本、GPU 和预算信息，并按 `docs/run-report-template.md` 回填 Issue。

```bash
tar -czf "<证据存储目录>/technology-one-<issue-id>.tar.gz" \
  -C "$FORMAL_ROOT" . \
  -C "$LEGO_ROOT/server" preflight-evolution.json \
  -C "$LEGO_ROOT/logs" <相关日志文件>
sha256sum "<证据存储目录>/technology-one-<issue-id>.tar.gz"
```

不得将数据集、模型权重、checkpoint、凭据、私有 profile、`runs/`、`logs/` 或证据包上传到公开仓库。正式候选选择阶段永远保持 `final_test` 锁定：`held_in` 用于接受/拒绝，`held_out` 只作报告；只有协议、方法选择和统计分析冻结后，维护者才能单独执行 final-test，并将其从主比较统计中隔离。
