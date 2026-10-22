# 技术一实验参数冻结卡

> **历史旧路线参数，仅用于复现已归档的 Harness 演化实验。** 不适用于当前长周期 T1/T2/T3 新路线；当前模型、任务和后续步骤以 `docs/model-configuration.md`、`docs/long-horizon-task-definition.md` 和 `docs/technology-one-safe-evolution-runbook.md` 为准。

本文件是给服务器贡献者和实验 Issue 使用的参数模板。贡献者不得自行选择 batch、seed、模式或预算；维护者应从下列任务类型中选择一种，并把完整参数复制到 Issue。服务器本地路径、GPU、endpoint 和 profile 哈希由贡献者填写后返回。

## 1. 不可修改的协议参数

| 参数 | 固定值 |
| --- | --- |
| 数据集 | `nvidia/HelpSteer2`，revision `990b2711a36180dd19d9c94b8627844866f8982a` |
| 任务 | `helpsteer2-preference` |
| Decision LLM | `Qwen/Qwen2.5-7B-Instruct`，使用 Issue 指定 revision |
| Task Model | `Qwen3-4B-preference-lora`，使用 Issue 指定 revision |
| 正式演化模式 | `independent`、`dependency-aware-joint` |
| 正式 seed | `17 23 29 31 37 41 43 47 53 59` |
| 候选接受集 | `held_in` |
| 回归报告集 | `held_out`，不参与接受/拒绝 |
| 最终测试集 | `final_test`，正式候选实验期间锁定 |
| Task Model 最大显存 | `40 GiB`，除非 Issue 明确记录新的统一值 |
| 随机性 | profile 中的 deterministic 设置必须保持不变 |

## 2. 任务类型

每个 Issue 只能选择下面一种任务类型。不要把不同任务的结果放在同一个输出目录。

### A. Smoke 链路检查

用于检查新服务器、数据、profile、Decision LLM、Task Model 和 evaluator 是否连通；不用于论文统计。

```text
task_type: smoke
profile: server_profile.evolution.qwen.json
seeds: 17
batches: 2
modes: independent dependency-aware-joint
train_group_budget: 1000
max_train_groups: 3000
max_eval_groups: 1000
command_retries: 3
retry_backoff_seconds: 15
final_test_unlocked: false
```

通过条件：两个模式都生成完成的 `summary.json`，`decision_llm_used=true`，`final_test_used=false`，并且日志能看到 proposal、候选评估和接受或回滚记录。

### B. 正式十 seed 比较

用于技术一的主比较，必须同时完成五类证据：组件矩阵、`fixed-harness-react`、`react-reflexion`、`independent` 和 `dependency-aware-joint`。

```text
task_type: formal-comparison
baseline_profile: server_profile.baseline_matrix.qwen.json
evolution_profile: server_profile.evolution.qwen.json
seeds: 17 23 29 31 37 41 43 47 53 59
evolution_batches: 10
component_matrix_batches: 1
fixed_baseline_runs: 10
reflexion_baseline_runs: 10
evolution_runs_per_mode: 10
modes: independent dependency-aware-joint
train_group_budget: 1000
max_train_groups: 3000
max_eval_groups: 1000
command_retries: 3
retry_backoff_seconds: 15
final_test_unlocked: false
```

完成条件：组件矩阵 10 个 summary、固定 ReAct 10 个 summary、ReAct+Reflexion 10 个 summary、两种演化模式各 10 个 summary；随后运行完整审计并得到 `PASS`。

### C. 长程验证

用于验证状态连续性、检查点、回滚和后续批次效果；它不能替代十 seed 正式比较。长程实验必须先完成 smoke，并使用新的输出根。

```text
task_type: long-horizon
profile: server_profile.evolution.qwen.json
seed: <从正式比较中预注册的代表性seed>
batches: 100 或 300
mode: <independent 或 dependency-aware-joint>
train_group_budget: 1000
max_train_groups: 3000
max_eval_groups: 1000
command_retries: 3
retry_backoff_seconds: 15
final_test_unlocked: false
```

长程 Issue 必须在启动前明确写出 `batches` 是 `100` 还是 `300`，不能由贡献者临时决定。报告中必须包含完成批次数、接受/拒绝、回滚、重复配置、命令重试、resume 次数、checkpoint 写入次数和最终 active Harness 版本。

## 3. 必须由 Issue 冻结的服务器参数

以下参数不能由仓库替贡献者猜测，必须逐项写入 Issue：

```text
repository: https://github.com/Mars-aliens/Data-Centric-RSI.git
commit: <精确commit或tag>
lego_root: <服务器上的projects/mle-lego绝对路径>
decision_endpoint: <例如http://127.0.0.1:8000/v1/chat/completions>
decision_model_id: <model_id>
decision_model_revision: <revision>
task_model_path: <服务器上的模型绝对路径>
task_model_id: <model_id>
task_model_revision: <revision>
train_jsonl_sha256: <sha256>
validation_jsonl_sha256: <sha256>
profile_sha256: <sha256>
protocol_sha256: <sha256>
gpu_id: <编号>
cuda_visible_devices: <编号>
task_model_max_gpu_memory_gib: 40
output_root: <本次任务唯一输出目录>
log_root: <日志目录>
```

`lego_root`、模型路径、endpoint、GPU 和输出目录是服务器本地值，不要提交到 GitHub。贡献者只需在 Issue 中记录脱敏后的路径、版本和哈希。

## 4. 参数一致性规则

- `independent` 和 `dependency-aware-joint` 必须使用相同的代码 commit、数据、模型、seed、batch、训练组预算、训练预算、评价预算和 GPU 资源条件；唯一实验变量是 proposal 约束。
- Smoke、正式比较和长程验证必须使用不同的 `OUTPUT_ROOT`。
- profile、数据、模型或代码发生变化时，必须新建 Issue 和输出根，不能 resume 旧 run。
- `RESUME=1` 只允许用于同一 commit、profile、数据、模型、seed、batch、预算和 run ID 的中断恢复。
- `final_test_unlocked` 必须为 `false`，且所有正式 summary 的 `final_test_used` 必须为 `false`。
- 贡献者不得因为拒绝率高而修改接受规则、seed、数据切分、模型或 proposal mode。

## 5. Issue 复制模板

```text
task_type: formal-comparison
commit: <填写>
profile_sha256: <填写>
train_jsonl_sha256: <填写>
validation_jsonl_sha256: <填写>
decision_model: Qwen/Qwen2.5-7B-Instruct @ <填写revision>
task_model: Qwen3-4B-preference-lora @ <填写revision>
decision_endpoint: <填写>
gpu_id: <填写>
cuda_visible_devices: <填写>
output_root: <填写唯一目录>
log_root: <填写>
seeds: 17 23 29 31 37 41 43 47 53 59
evolution_batches: 10
component_matrix_batches: 1
train_group_budget: 1000
max_train_groups: 3000
max_eval_groups: 1000
command_retries: 3
retry_backoff_seconds: 15
final_test_unlocked: false
smoke_required: true
audit_required: true
```
