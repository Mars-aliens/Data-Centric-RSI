# 技术一代码入口

当前技术一是“安全自进化与故障隔离”。服务器执行从 [`docs/server-quickstart.md`](../../../docs/server-quickstart.md) 开始，正式实验规则见 [`experiments/technology_one/formal-protocol-v1.md`](../../../experiments/technology_one/formal-protocol-v1.md)。

主要入口：

- `long_horizon_task.py`：HelpSteer2 分组、批次和延迟反馈；
- `long_horizon_runner.py`：固定流水线和状态恢复；
- `candidate_proposal.py`：Decision LLM 候选 bundle 校验；
- `candidate_sandbox.py` / `candidate_shadow.py`：候选隔离、输出校验和故障注入；
- `qwen_preference.py`：Task Model 预测、LoRA 训练和 checkpoint 续训。

旧 Harness 演化入口已经移到 `archive/legacy_lego/`。当前新 T1 尚未完成周期性多候选调度；不要把单候选 smoke 当作正式递归结果。

## 研究定位

LEGO 研究 Data System 如何在长周期运行中持续改进并保持稳定。Data System 的整体状态包括数据构建与处理、Task Model 更新、评价规则、反馈记录、Memory、Skill、工具调用和 Harness 代码；Harness 是组织这些环节的主要改进载体。

技术一的整体方向是**安全自进化与故障隔离**：系统可以开放地提出 Data System 修改，但候选先隔离并做安全检查；失败时当前稳定流程继续运行。依赖关系用于影响分析，不是固定修改白名单。其稳定性评价覆盖故障传播、运行连续性和恢复。

已有正式技术一实验研究的是这个方向中的一个具体子问题：依赖关系引导的 Harness 组件联合修改。该实验已实现候选评估、接受/拒绝、检查点和恢复，但尚未验证开放式修改空间下完整 Data System 的故障隔离。因此，不能将已有联合修改实验等同于新 T1 的全部研究目标。

新集成路径另提供 `CandidateSafetyGate`：安全检查通过只让候选进入后续观察，不直接提升为稳定版本。既有 `HarnessEvolutionController` 保留原实验协议中的即时 held-in 接受语义；新集成流程不得将旧协议结果解释成 T3 的多批次版本选择结果。

LEGO 不使用 Aladdin 的策略或实验结果。初始 Data System 必须固定、可复现，并明确记录数据清单、模型版本、评价版本、Memory/Skill 版本和 Harness 版本。

## 三个技术名称

- 技术一：安全自进化与故障隔离；
- 技术二：可执行经验生命周期管理；
- 技术三：基于后续运行表现的更新保留与回退。

三个技术共同使用固定的 Data System 起点、统一的数据和模型预算，并分别报告即时质量、连续运行、错误恢复和资源稳定性。固定基线流水线是共同实验起点，不属于其中任何一个技术。

## 对照方法登记

所有需要报告的对照及其来源、当前实现状态和服务器适配状态记录在
`baseline_matrix.json`；原论文核验、官方实现/许可证状态和复现边界记录在
`baseline_research.md`。其中 `general-self-harness` 和
`curateevo-adaptation` 是相关工作适配，不应写成原论文的原始结果；
`independent-edit`、`random-joint-edit`、`one-batch-selection` 和
`same-memory-no-process-update` 是本项目的消融或控制。当前主实验数据集锁定为
HelpSteer2；PRISM 仅作为未来用户/session 扩展，不进入当前主比较。真实 HelpSteer2
目前只完成了 fixed scaffold，toy strategy 结果不能替代服务器上的统一模型实验。

## 当前研究执行顺序

旧组件演化比较（independent、dependency-aware-joint、generator/selector/filter
消融等）仅作为技术一早期研究记录归档；不再是当前方向的前置实验，也不作为新框架的
稳定性结论。

当前新框架按以下顺序推进：

1. 在服务器用固定 revision 的真实 HelpSteer2 数据生成 manifest，审计 group、标签和流批次统计。
2. 先跑固定流式 Qwen baseline，验证“先预测当前批，再用已释放反馈更新后续 checkpoint”的因果时序、恢复与日志。
3. 加入 T1 候选隔离、安全检查和故障注入，确认失败修改不污染稳定运行。
4. 加入 T2 经验/Skill 的来源追踪、验证、过期和冲突治理，确认失效经验不会长期污染后续批次。
5. 加入 T3 延迟反馈归因、多批次观察、版本保留与恢复，确认更新选择不会因短期噪声或错配反馈而不稳定。
6. 在同一数据流、初始状态和总预算下开展组合、消融、多随机种子和长周期故障压力测试。

每一阶段都同时报告任务质量、连续完成率、失败隔离/恢复、数据完整性和资源消耗；不能以
接受率或某一静态指标替代长周期稳定性证据。`held_out` 仅供可信评测报告，`final_test`
在全部协议冻结前保持锁定。

## 研究数据协议

真实数据接入使用 `protocol.py` 中的 JSONL adapter。每行至少包含
`record_id`、`split`、`input`、`target`；可选 `task_id`、`batch_id`、`feedback`、
`metadata` 和 `parent_record_ids`。真实数据应先按原始任务或会话划分，再把派生记录保留在同一 split；`held_out` 用于候选回归，`final_test` 只用于最终报告。
反馈可以携带 `source_batch` 与 `release_batch`，本地队列只在到达释放批次时返回一次。当前首选数据集和备选数据集、切分规则、批次规则及接入前核验项记录在 `real_data_plan.json`；原始数据获取、许可、规模和任务映射核验仍是接入真实数据前的单独门槛。

拿到归一化 JSONL 后，先运行：

```powershell
python -m technologies.technology_one.validate_data path\to\records.jsonl
```

只有校验通过的数据才进入 fixed baseline；校验器会检查字段、split、source group、parent provenance、batch_id 和延迟反馈规则。

HelpSteer2 的归一化命令为：

```powershell
python -m technologies.technology_one.prepare_helpsteer2 data\helpsteer2-train.jsonl
python -m technologies.technology_one.validate_data data\helpsteer2-train.jsonl
python -m technologies.technology_one.prepare_helpsteer2 data\helpsteer2-validation.jsonl --split validation
python -m technologies.technology_one.validate_data data\helpsteer2-validation.jsonl
```

train 只产生 seed/held-in/held-out；官方 validation 独占 final_test。该 adapter 将同一 prompt 的回答保持在同一个 source group，并根据组内五维评分均值生成 `preferred`、`rejected` 或 `tie`；最高分为 preferred，最低分为 rejected，中间分为 tie。它不声称拥有用户长期会话或人工 rationale。

真实固定基线流水线默认不读取 final_test。最终报告时显式运行：

```powershell
python -m technologies.technology_one.real_data_runner data\helpsteer2-train.jsonl `
  --final-test data\helpsteer2-validation.jsonl `
  --report-final-test
```

服务器接入前，先对真实 train JSONL 做两次相同运行的重复性检查：

```powershell
python -m technologies.technology_one.reproducibility data\helpsteer2-train.jsonl `
  --output-dir experiments\technology_one\runs\real-data\reproducibility `
  --run-id-prefix helpsteer2-fixed-repro `
  --manifest experiments\technology_one\runs\real-data\lego-fixed-baseline-freeze.json
```

该命令只读取事件日志比较配置、数据和 checkpoint hash、指标、状态与预算，
不会加载大型 checkpoint 的词表。只有输出 `differences: []` 才能冻结当前
deterministic scaffold；它仍不等价于服务器上的真实 ReAct LLM baseline。

服务器交付清单和服务器端执行顺序记录在 `server_handoff.md`。

## Decision 基线

本节记录旧 Harness 策略实验的 Decision LLM 设置；当前固定流式 Task Model baseline 不调用 Decision LLM。

真实服务器适配器为两个策略对照使用同一个固定版本的决策模型。建议起始模型为 `Qwen/Qwen2.5-7B-Instruct`，temperature 设为 `0.0`，固定模型版本，并在两个实验组使用相同的上下文长度和输出限制。该模型用于决策策略，不是接受训练的 Task Model。

`fixed-harness-react` 是固定基线流水线上的 ReAct 对照，流水线包含数据处理、Task Model 训练、评估和运行记录。模型根据一条观察选择一项许可操作，但不能持久修改流水线。`react-reflexion` 使用相同 prompt、操作、工具、模型和预算，并在符合条件的结果后额外生成受限长度的文字反思。反思追加到 JSONL 文件，从下一次策略决策起可见，且不修改代码、Prompt、Skill 或 Memory 规则。反思调用和 Token 计入共享决策预算。

策略命令接收 `lego-policy-v1` payload 和完整渲染后的 system/user prompts。它必须输出包含 `proposal` 的 JSON 对象；Reflexion 实验组也可以输出 `reflection`。适配器记录模型版本、prompt 哈希、符合条件的反思 ID 及反思来源信息。这是对已发表 ReAct 和 Reflexion 思路的任务适配，并不表示其原始基准代码直接实现了 HelpSteer2。

## 接入点

LEGO 实现 `core_system.mle_core.interfaces.PolicyAdapter` 和独立的 Harness
manifest/version 管理，不修改 Aladdin 目录。

## 新长周期任务流

新的 Task Definition v1 由 `long_horizon_task.py` 实现，与历史 `helpsteer2.py`
适配器和旧技术一实验语义分离。它在可信 runner 侧按完整 prompt group 生成
seed/development/held-in/held-out/final-test 划分和哈希 manifest；agent 视图中的
development 当前批次不含 `target` 或评分。`FeedbackOracle` 在来源批次结束后的
规定边界释放标签，并只在释放后计算绑定模型版本的预序列指标。

可在已有原始数据 JSONL 的服务器上生成真实规模的预检 bundle：

```powershell
python scripts/prepare_long_horizon_task.py `
  --train-jsonl <path-to-train.jsonl> `
  --validation-jsonl <path-to-validation.jsonl> `
  --output-dir <new-preflight-output-dir> `
  --stream-seed 17 --groups-per-batch 50 --feedback-delay 2
```

bundle 中 `agent-visible/` 只含 seed labels 与 manifest；development 输入和反馈释放时间表
放在 `runner-only/`，必须用文件权限或独立挂载隔离，不能只依赖目录命名。不含 development
gold targets/scores，也不导出 held-out/final-test。该预检包不作为 agent 的运行输入。

新固定流式 Qwen baseline 已有独立 runner。训练配置 JSON 至少指定本地底座目录和不可变
revision，例如：

```json
{
  "base_model_path": "/models/qwen-local-snapshot",
  "base_model_revision": "<immutable-model-snapshot-id>",
  "max_steps": 100,
  "learning_rate": 0.0002,
  "per_device_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_length": 1024,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05
}
```

以上是 runner 输入示例，不代表正式实验参数已冻结。真实运行须显式登记实际模型 revision、
训练配置、GPU/内存设置和随机种子；每批先对当前无标签数据预测，到期反馈到来后才将
seed 加已释放反馈用于 parent-checkpoint 续训。恢复目录必须是同一个 run 的输出目录。

```powershell
$env:LEGO_TASK_MODEL_DEVICE = "cuda"
python scripts/run_long_horizon_qwen.py `
  --train-jsonl <path-to-train.jsonl> `
  --validation-jsonl <path-to-validation.jsonl> `
  --config-json <path-to-frozen-task-model-config.json> `
  --output-dir <new-or-resumable-run-dir> `
  --run-id <unique-run-id> `
  --seed 17 --stream-seed 17 --groups-per-batch 50 --feedback-delay 2
```

这只验证固定流水线的数据时序、Qwen 训练/预测和恢复，不代表 T1/T2/T3 实验已完成。
启动正式长周期比较前仍需以真实数据生成 manifest、完成真实模型 smoke，并冻结各技术的
故障注入、总预算和统计方案。run 目录保存原子 `run_state.json`、`command_attempts.jsonl`、
每批的可信预测 JSONL/checksum 和已释放标签上的合并预序列 confusion matrix 指标；
尚未释放的标签不进入 agent 可见结果。摘要区分已提交成本与命令尝试账本中模型报告的成本；
命令失败时实际 GPU 成本可能无法从子进程获得，须同时保留服务器进程/GPU 监控记录。

## 新技术一：候选安全自进化

新 T1 runner 从一份带 `candidate_proposal.json` 的源码 bundle 启动。使用
`scripts/propose_t1_candidate.py` 调用配置的 Decision LLM 生成 bundle，使用
`scripts/run_long_horizon_t1_qwen.py` 对同一 HelpSteer2 流并行执行可信稳定分支和隔离候选分支。
候选获得当前批次无标签输入及已释放反馈，不接触 held-out/final-test；安全门只允许后续观察，
不会在 T1 阶段晋升稳定版本。失败只终止候选分支，稳定运行继续。

所有模型均由运行配置选择。Qwen 系列可作为本地 Task Model 或兼容 API 的 Decision LLM；
Kimi K3 可作为支持对应 OpenAI-compatible Chat Completions 请求的托管 Decision LLM。
模型市场条目并非全部自动兼容；Task Model 还必须有本地权重、PEFT/Transformers 兼容性和足够显存。
