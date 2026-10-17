# MLE-LEGO 本地实验

当前已落地的独立 baseline 是 `fixed-harness-react` 的确定性骨架，基础流程名称为“固定基线流水线”。它包含固定的数据处理、Task Model 训练、统一评测和实验记录，行为保持不变，作为所有 LEGO 方法的控制组。新增的本地机制层在 toy stream 上实现 Harness 版本、失败诊断、文字 Reflexion、独立/随机成组/依赖感知成组修改、held-in/held-out 验证、拒绝与回滚，以及可执行 Experience/Skill 记录。

运行：

```powershell
python -m local_experiments.lego.runner

# 运行本地 Harness 策略比较。
python -m local_experiments.lego.experiment_runner
```

结果写入 `experiments/technology_one/runs/<run_id>/` 和
`experiments/technology_one/runs/strategies/<strategy>/`。

## 独立研究边界

LEGO 只研究 Data Harness 的自我修改：

- 根据失败诊断候选 Harness 修改；
- 根据组件依赖决定单组件或成组修改；
- 将经验保存为可执行的修改方法；
- 使用连续批次结果接受或回滚候选版本。

它不使用 Aladdin 的策略或实验结果。LEGO 的初始 Data Harness 必须独立、固定、可复现。

## 三个技术名称

- 技术一：依赖关系引导的成组修改；
- 技术二：可执行经验复用；
- 技术三：连续运行反馈选择。

“固定基线流水线”是三个技术共同使用的实验起点，不属于其中任何一个技术。

## 对照方法登记

所有需要报告的对照及其来源、当前实现状态和服务器适配状态记录在
`baseline_matrix.json`；原论文核验、官方实现/许可证状态和复现边界记录在
`baseline_research.md`。其中 `general-self-harness` 和
`curateevo-adaptation` 是相关工作适配，不应写成原论文的原始结果；
`independent-edit`、`random-joint-edit`、`one-batch-selection` 和
`same-memory-no-process-update` 是本项目的消融或控制。当前主实验数据集锁定为
HelpSteer2；PRISM 仅作为未来用户/session 扩展，不进入当前主比较。真实 HelpSteer2
目前只完成了 fixed scaffold，toy strategy 结果不能替代服务器上的统一模型实验。

## 建议的本地实验阶段

1. 先运行固定基线流水线 baseline。
2. 加入文字 Reflexion baseline。
3. 加入 generator-only（当前命名为 independent-edit）、selector-only 和 random-joint-edit baseline。
4. 最后实现 dependency-aware joint edit。
5. 使用 held-in 发现问题，使用 held-out 做回归验证。
6. final_test 只用于最终报告，不参与 Harness 选择。

## 研究数据协议

真实数据接入使用 `protocol.py` 中的 JSONL adapter。每行至少包含
`record_id`、`split`、`input`、`target`；可选 `task_id`、`batch_id`、`feedback`、
`metadata` 和 `parent_record_ids`。真实数据应先按原始任务或会话划分，再把派生记录保留在同一 split；`held_out` 用于候选回归，`final_test` 只用于最终报告。
反馈可以携带 `source_batch` 与 `release_batch`，本地队列只在到达释放批次时返回一次。当前首选数据集和备选数据集、切分规则、批次规则及接入前核验项记录在 `real_data_plan.json`；原始数据获取、许可、规模和任务映射核验仍是接入真实数据前的单独门槛。

拿到归一化 JSONL 后，先运行：

```powershell
python -m local_experiments.lego.validate_data path\to\records.jsonl
```

只有校验通过的数据才进入 fixed baseline；校验器会检查字段、split、source group、parent provenance、batch_id 和延迟反馈规则。

HelpSteer2 的归一化命令为：

```powershell
python -m local_experiments.lego.prepare_helpsteer2 data\helpsteer2-train.jsonl
python -m local_experiments.lego.validate_data data\helpsteer2-train.jsonl
python -m local_experiments.lego.prepare_helpsteer2 data\helpsteer2-validation.jsonl --split validation
python -m local_experiments.lego.validate_data data\helpsteer2-validation.jsonl
```

train 只产生 seed/held-in/held-out；官方 validation 独占 final_test。该 adapter 将同一 prompt 的回答保持在同一个 source group，并根据组内五维评分均值生成 `preferred`、`rejected` 或 `tie`；最高分为 preferred，最低分为 rejected，中间分为 tie。它不声称拥有用户长期会话或人工 rationale。

真实固定基线流水线默认不读取 final_test。最终报告时显式运行：

```powershell
python -m local_experiments.lego.real_data_runner data\helpsteer2-train.jsonl `
  --final-test data\helpsteer2-validation.jsonl `
  --report-final-test
```

服务器接入前，先对真实 train JSONL 做两次相同运行的重复性检查：

```powershell
python -m local_experiments.lego.reproducibility data\helpsteer2-train.jsonl `
  --output-dir experiments\technology_one\runs\real-data\reproducibility `
  --run-id-prefix helpsteer2-fixed-repro `
  --manifest experiments\technology_one\runs\real-data\lego-fixed-baseline-freeze.json
```

该命令只读取事件日志比较配置、数据和 checkpoint hash、指标、状态与预算，
不会加载大型 checkpoint 的词表。只有输出 `differences: []` 才能冻结当前
deterministic scaffold；它仍不等价于服务器上的真实 ReAct LLM baseline。

服务器交付清单和服务器端执行顺序记录在 `server_handoff.md`。

## Decision 基线

真实服务器适配器为两个策略对照使用同一个固定版本的决策模型。建议起始模型为 `Qwen/Qwen2.5-7B-Instruct`，temperature 设为 `0.0`，固定模型版本，并在两个实验组使用相同的上下文长度和输出限制。该模型用于决策策略，不是接受训练的 Task Model。

`fixed-harness-react` 是固定基线流水线上的 ReAct 对照，流水线包含数据处理、Task Model 训练、评估和运行记录。模型根据一条观察选择一项许可操作，但不能持久修改流水线。`react-reflexion` 使用相同 prompt、操作、工具、模型和预算，并在符合条件的结果后额外生成受限长度的文字反思。反思追加到 JSONL 文件，从下一次策略决策起可见，且不修改代码、Prompt、Skill 或 Memory 规则。反思调用和 Token 计入共享决策预算。

策略命令接收 `lego-policy-v1` payload 和完整渲染后的 system/user prompts。它必须输出包含 `proposal` 的 JSON 对象；Reflexion 实验组也可以输出 `reflection`。适配器记录模型版本、prompt 哈希、符合条件的反思 ID 及反思来源信息。这是对已发表 ReAct 和 Reflexion 思路的任务适配，并不表示其原始基准代码直接实现了 HelpSteer2。

## 接入点

LEGO 实现 `core_system.mle_core.interfaces.PolicyAdapter` 和独立的 Harness
manifest/version 管理，不修改 Aladdin 目录。
