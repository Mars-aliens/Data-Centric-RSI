# 技术一实验执行指南

本文件用于在新服务器上完整执行 LEGO 技术一。执行者只需要配置服务器本地路径、GPU、模型和 Decision LLM 服务；研究参数按本文固定，不能临时修改。

## 1. 实验目标

技术一验证 Data Harness 是否能够根据运行反馈修改自身，并通过候选训练和评测决定接受或回滚修改。

正式比较的两个方法是：

- `independent`：每次只修改一个组件；
- `dependency-aware-joint`：按照依赖闭包同时修改相关组件，正式提案必须包含 `generator + selector`，也可以包含 `filter`。

两种方法的 Decision LLM、Task Model、数据、随机种子、批次数、预算和评测规则必须完全相同。唯一的实验变量是修改约束。

正式实验只使用本文定义的 `independent` 和 `dependency-aware-joint` 两种模式。

## 2. 固定实验参数

正式十随机种子：

```text
17 23 29 31 37 41 43 47 53 59
```

正式演化实验使用：

```text
evolution batches: 10
train group budget: 1000
max train groups: 3000
max evaluation groups: 1000
command retries: 3
retry backoff seconds: 15
task model max GPU memory: 40 GiB
candidate selection split: held_in
regression/report split: held_out
final test: locked
```

正式实验期间必须满足：

- `final_test_used=false`；
- held-out 只用于报告，不参与接受或拒绝；
- Decision LLM 的模型和 revision 固定；
- 每个 seed 使用新的、唯一的输出目录；
- 不因为拒绝率高而修改接受规则、数据切分、seed 或模型。

## 3. 在服务器上准备项目

服务器上使用 GitHub 仓库的 LEGO 项目目录：

```bash
git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/mle-lego
git rev-parse HEAD
```

设置环境变量。把 `<GPU编号>` 替换为实际空闲 GPU：

```bash
export LEGO_ROOT="$PWD"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=<GPU编号>
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=40
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 填写服务器上实际运行的 Decision LLM 地址。
export POLICY_API_ROOT="http://<DECISION-LLM-HOST>:<PORT>/v1"
export LEGO_POLICY_API_URL="$POLICY_API_ROOT/chat/completions"
export LEGO_POLICY_TIMEOUT_SECONDS=600

mkdir -p "$LEGO_ROOT/logs" "$LEGO_ROOT/server" "$LEGO_ROOT/runs"
```

如果服务器没有环境：

```bash
python3 -m venv "$LEGO_ROOT/.venv"
source "$LEGO_ROOT/.venv/bin/activate"
python -m pip install -r "$LEGO_ROOT/requirements-server.txt"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
```

## 4. 准备数据、profile 和模型

以下文件属于服务器私有输入，不要提交到 GitHub：

```text
$LEGO_ROOT/data/helpsteer2-train.jsonl
$LEGO_ROOT/data/helpsteer2-validation.jsonl
$LEGO_ROOT/server_profile.baseline_matrix.qwen.json
$LEGO_ROOT/server_profile.evolution.qwen.json
```

复制 profile 示例：

```bash
cp "$LEGO_ROOT/code/technologies/technology_one/server_profile.baseline_matrix.qwen.example.json" \
   "$LEGO_ROOT/server_profile.baseline_matrix.qwen.json"

cp "$LEGO_ROOT/code/technologies/technology_one/server_profile.evolution.qwen.example.json" \
   "$LEGO_ROOT/server_profile.evolution.qwen.json"
```

在两个私有 profile 中填写本服务器的 JSONL 路径、Task Model 权重路径、Decision LLM 的 model ID/revision，以及 trainer、evaluator 和 policy command 路径。

开始实验前必须确认以下内容：

- Decision LLM 的 host、port、model ID 和 revision；
- Task Model 权重绝对路径；
- Python、trainer 和 evaluator 路径；
- GPU 编号、显存上限和输出目录；
- HelpSteer2 文件版本和 SHA256。

核对演化 profile：

```bash
"$TRAIN_PY" - "$LEGO_ROOT/server_profile.evolution.qwen.json" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
print("decision_model:", p["decision_model"]["model_id"], p["decision_model"]["revision"])
print("task_model:", p["task_model"]["model_id"], p["task_model"]["revision"])
print("base_model_path:", p["task_model"]["training_config"]["base_model_path"])
print("final_test_unlocked:", p["action_system"]["final_test_unlocked"])
print("candidate_selection_split:", p["stream"]["candidate_selection_split"])
print("regression_split:", p["stream"]["regression_split"])
PY
```

`final_test_unlocked` 必须是 `false`，candidate split 必须是 `held_in`，regression split 必须是 `held_out`。

校验数据：

```bash
"$TRAIN_PY" -m local_experiments.lego.validate_data \
  "$LEGO_ROOT/data/helpsteer2-train.jsonl"

"$TRAIN_PY" -m local_experiments.lego.validate_data \
  "$LEGO_ROOT/data/helpsteer2-validation.jsonl"
```

## 5. 运行安装和服务预检

```bash
bash "$LEGO_ROOT/scripts/check_lego_install.sh"

"$TRAIN_PY" -m local_experiments.lego.server_preflight \
  --profile "$LEGO_ROOT/server_profile.evolution.qwen.json" \
  --root "$LEGO_ROOT" \
  --report "$LEGO_ROOT/server/preflight-evolution.json"

curl -fsS --max-time 30 \
  "$POLICY_API_ROOT/models"
```

运行前必须把 `<DECISION-LLM-HOST>` 和 `<PORT>` 替换成新服务器实际可访问的地址；只修改这里，不要修改代码。

Decision LLM 的 `/v1/models`、policy command、Task Model 和 evaluator 都正常后，才能开始 smoke。不要用修改协议的方式绕过预检错误。

## 6. 第一步：两批次 smoke

smoke 只检查完整执行链路，不作为论文统计结果：

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/technology-one-smoke" \
GPU_ID=<GPU编号> \
BATCHES=2 \
SEEDS="17" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh" \
  > "$LEGO_ROOT/logs/technology-one-smoke.log" 2>&1
```

检查结果：

```bash
find "$LEGO_ROOT/runs/technology-one-smoke" -name summary.json -print

grep -RniE 'Traceback|out of memory|error|failed|exception' \
  "$LEGO_ROOT/logs/technology-one-smoke.log" || true
```

smoke 通过条件：

- 两个模式都生成 `summary.json`；
- 状态为 `complete`；
- `decision_llm_used=true`；
- `final_test_used=false`；
- 日志中有 proposal、候选评估、接受或回滚记录；
- 没有未处理的 traceback 或 CUDA OOM。

## 7. 第二步：正式十 seed 实验

推荐使用完整入口。它会依次运行组件矩阵、固定 ReAct/Reflexion 对照，以及两种技术一演化方法；已经完成的 seed 会自动跳过。

```bash
export PROFILE_BASELINE="$LEGO_ROOT/server_profile.baseline_matrix.qwen.json"
export PROFILE_EVOLUTION="$LEGO_ROOT/server_profile.evolution.qwen.json"
export GPU_ID=<GPU编号>
export SEEDS="17 23 29 31 37 41 43 47 53 59"
export BATCHES=10

RUN_COMPONENT_MATRIX=1 \
COMPONENT_ROOT="$LEGO_ROOT/runs/technology-one-formal/component-matrix" \
DECISION_ROOT="$LEGO_ROOT/runs/technology-one-formal/decision-baselines" \
EVOLUTION_ROOT="$LEGO_ROOT/runs/technology-one-formal/evolution-comparison" \
bash "$LEGO_ROOT/scripts/run_technology_one_complete_batch.sh" \
  > "$LEGO_ROOT/logs/technology-one-formal.log" 2>&1
```

正式核心比较目录：

```text
runs/technology-one-formal/evolution-comparison/independent/
runs/technology-one-formal/evolution-comparison/dependency-aware-joint/
```

正式汇总只使用 `independent` 和 `dependency-aware-joint` 两个目录。

如果组件矩阵已经完成，可以设置 `RUN_COMPONENT_MATRIX=0`，但必须先确认十个 seed 的摘要全部存在。

## 8. 运行期间查看进度

```bash
ROOT="$LEGO_ROOT/runs/technology-one-formal/evolution-comparison"

for MODE in independent dependency-aware-joint; do
  echo "===== $MODE ====="
  find "$ROOT/$MODE" -name progress.json -print0 2>/dev/null |
  while IFS= read -r -d '' FILE; do
    "$TRAIN_PY" - "$FILE" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
print("file:", sys.argv[1])
print("status:", p.get("status"))
print("progress:", p.get("completed_batches"), "/", p.get("target_batches"))
print("phase:", p.get("phase"))
print("accepted/rejected:", p.get("accepted"), "/", p.get("rejected"))
print("retries:", p.get("reliability", {}).get("command_retry_count"))
PY
  done
done
```

拒绝必须按原因分类：

- `held_in_regression`：候选真实运行后性能退步；
- `duplicate_configuration`：候选配置已出现；
- `proposal is a no-op`：提案没有产生有效改变；
- `decision_proposal_invalid`：提案不符合协议；
- 命令失败或 CUDA OOM：执行可靠性问题。

拒绝不等于程序失败，也不能删除拒绝记录或临时修改接受规则。

## 9. 正式结果审计

五类证据全部完成后执行：

```bash
"$TRAIN_PY" "$LEGO_ROOT/scripts/audit_technology_one_complete.py" \
  --component-root "$LEGO_ROOT/runs/technology-one-formal/component-matrix" \
  --fixed-root "$LEGO_ROOT/runs/technology-one-formal/decision-baselines/fixed-harness-react" \
  --reflexion-root "$LEGO_ROOT/runs/technology-one-formal/decision-baselines/react-reflexion" \
  --independent-root "$LEGO_ROOT/runs/technology-one-formal/evolution-comparison/independent" \
  --dependency-root "$LEGO_ROOT/runs/technology-one-formal/evolution-comparison/dependency-aware-joint" \
  --seeds 17 23 29 31 37 41 43 47 53 59 \
  --output "$LEGO_ROOT/runs/technology-one-formal/audit.json"
```

只有审计通过后，才能把结果用于技术一正式结论。

## 10. 第三步：长程验证

长程验证不能替代十 seed 正式比较。应从正式比较中选择代表性 seed，并使用新的输出目录：

```bash
export SEED=17
export BATCHES=100
export RUN_ID="technology-one-long-horizon-seed-$SEED"
export OUTPUT_DIR="$LEGO_ROOT/runs/technology-one-long-horizon"
export RESUME=0

nohup bash "$LEGO_ROOT/scripts/run_technology_one_evolution_endurance.sh" \
  > "$LEGO_ROOT/logs/$RUN_ID.log" 2>&1 &
```

中断后只能在代码、profile、数据、模型、seed、batch、预算和 `RUN_ID` 都相同的情况下恢复：

```bash
export RESUME=1
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_endurance.sh" \
  > "$LEGO_ROOT/logs/$RUN_ID-resume.log" 2>&1
```

长程报告必须包含完成批次数、接受/拒绝、回滚、重复配置、命令重试、resume 次数、checkpoint 写入次数和最终 active Harness 版本。

## 11. 结果报告和文件边界

实验完成后，在仓库之外保存证据包，至少包含每个 seed 的 `summary.json`、`progress.json`、`endurance_state.json`、日志、最终审计 JSON、代码/profile/数据/模型版本和运行报告。

不要提交到 GitHub：

- HelpSteer2 数据；
- 模型权重、checkpoint；
- 私有 profile、凭据和 API key；
- `runs/`、`logs/` 和完整证据包。

正式结论至少同时报告性能指标、接受率、拒绝原因、回滚次数、Decision LLM 成本和长程可靠性。一次 smoke、一次成功运行或较高接受率都不能单独证明技术一有效。
