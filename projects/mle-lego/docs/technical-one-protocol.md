# MLE-LEGO 技术一：完整实验协议与执行清单

## 1. 研究对象

技术一验证的是 Data Harness 能否在持续运行中根据执行反馈修改自身，并在候选修改经过实际训练和评价后决定保留或回滚。Harness 的改进对象是数据生成、选择、过滤、执行顺序、工具调用，以及 Memory/Skill 的使用规则；数据样本、模型检查点和反馈记录本身不是 Harness 改进。

核心机制为：执行 -> 反思/反馈 -> Decision LLM 提议 -> 候选 Harness 实验 -> held-in 接受或拒绝 -> 保留/回滚。held-out 只作报告，final test 在正式冻结前始终锁定。

## 2. 必须回答的问题

1. 相互依赖的组件是否需要组合修改，而不是只改一个组件。
2. 依赖关系约束的组合修改是否比独立修改更容易形成有效更新。
3. 更新后的 Harness 是否能在后续批次继续运行，并在长程运行中保持状态、预算、版本和回滚链完整。
4. 结果是否来自 Harness 运行方法的改变，而不是 Decision LLM 参数变化。

## 3. 实验臂

### 3.1 组件交互矩阵

固定十个种子：`17, 23, 29, 31, 37, 41, 43, 47, 53, 59`。矩阵包含：

`fixed`、`independent-edit`、`selector-only-edit`、`filter-only-edit`、`dependency-aware-joint-edit`、`generator-filter-joint-edit`、`selector-filter-joint-edit`、`generator-selector-filter-joint-edit`。

该矩阵用于回答“哪些组件组合具有交互效应”。它是受控组件实验，不等同于完整的 Decision-LLM 自进化实验。

### 3.2 固定策略基线

真实执行两个控制方法：

- `fixed-harness-react`：固定 Harness，使用 ReAct 式决策和工具执行。
- `react-reflexion`：固定 Harness，额外持久化并使用反思经验。

这两个方法用于区分“仅有交互/经验记录”和“实际修改 Harness”的贡献。

### 3.3 Decision LLM 自进化对照

两个实验臂使用相同的模型、数据、种子、批次数、训练组预算、训练预算和评价预算：

- `independent`：每次只允许修改一个组件。
- `dependency-aware-joint`：严格要求合法的依赖闭包联合修改，只允许 `generator + selector` 或 `generator + selector + filter`；单独修改 `generator` 不属于该正式实验臂。

此前完成的 `dependency-aware` 十种子、十批结果保留为 natural proposal pilot：它允许 generator-only，因此不能作为联合修改效果证据。

两臂都通过真实 Decision LLM 生成 JSON proposal，由严格解析器校验；候选修改必须进入候选分支实际运行，held-in 改善才接受，否则回滚。

## 4. 公平性与锁定规则

- Decision LLM 的模型和 revision 固定。
- 所有正式臂使用同一组十个种子。
- independent 与 dependency-aware-joint 使用相同的批次数和候选训练/评价预算。
- 每个候选记录 parent version、candidate version、edits、指标、接受/拒绝原因和 rollback。
- held-out 不参与接受判定，只作报告。
- `final_test_used` 在正式冻结前必须为 `false`。
- 每个种子必须有完整 summary；重跑时审计器选择该种子最新的 complete summary。

## 5. 指标

主要指标：held-in accuracy、macro-F1，以及候选接受率、拒绝率、回滚次数、重复配置次数、提议无操作次数和 Decision LLM 调用成本。

长程可靠性指标：完成批次数/目标批次数、命令重试次数、可恢复候选失败次数、checkpoint 写入次数、resume 次数、状态文件连续性和最终 Harness 版本。

长程结果不能只报告“没有崩溃”。应同时报告修改是否继续被接受、性能是否退化、回滚是否生效、预算是否越界，以及错误是否被隔离。

## 6. 当前实现状态

已在本地完成：

- Decision LLM proposal 协议和依赖闭包校验。
- `dependency-aware-joint` 严格联合修改协议，禁止 generator-only proposal。
- independent 与 dependency-aware-joint 两种正式模式的真实执行路径；旧 dependency-aware natural pilot 保留为历史对照。
- 候选评估、接受、拒绝、回滚、checkpoint 和 resume。
- 固定 ReAct 与 ReAct+Reflexion 基线的真实 runner。
- 十种子组件矩阵和两种自进化臂的批量脚本。
- 完整审计器：检查证据臂、种子、方法标签、模式标签、声明预算和 final-test 锁。
- 本地协议、集成和审计测试。

已有服务器证据：

- 60 批 endurance：`60/60`，`1 accepted / 58 rejected`，无命令重试，约 11.57 小时。该结果证明循环稳定性，不证明长期性能改进。
- 组件矩阵已有十种子结果，selector-filter 组合相对 fixed 的 held-out 平均差值为 `+0.0404`，`8/10` 获胜。该结果属于组件交互证据。
- Decision LLM 10 批 pilot 已显示依赖闭包非法提议和 held-in regression，说明 proposal 约束与候选质量仍需作为结果报告的一部分。

尚未声称完成的服务器证据：

- 统一协议下 fixed-harness-react 十种子完整结果。
- 统一协议下 react-reflexion 十种子完整结果。
- independent 十种子完整结果。
- dependency-aware-joint 十种子完整结果，且每个 seed 至少产生一个联合 proposal。
- 两种 Decision LLM 模式的统计比较和长程 100/300 批稳定性结果。

## 7. 服务器执行顺序

上传代码后先运行小规模 smoke test，确认 profile、Decision LLM endpoint、任务模型和 evaluator 均可用；再按以下顺序运行：

1. 固定 ReAct 与 ReAct+Reflexion 十种子。
2. independent 与 dependency-aware-joint 十种子配对实验。
3. 从已通过 smoke test 的模式中选择代表性种子运行 100 或 300 批长程实验。
4. 所有结果完成后运行 `audit_technology_one_complete.py`。
5. final test 只在实验协议冻结、审计通过后单独执行，并从主比较统计中隔离。

批量入口：`scripts/run_technology_one_complete_batch.sh`。该入口默认跳过组件矩阵；完整五类证据必须显式设置 `RUN_COMPONENT_MATRIX=1`。审计入口：`scripts/audit_technology_one_complete.py`。

完整运行的目录约定如下，审计时必须把这五个 root 分别传入：

```text
<FORMAL_ROOT>/component-matrix
<FORMAL_ROOT>/decision-baselines/fixed-harness-react
<FORMAL_ROOT>/decision-baselines/react-reflexion
<FORMAL_ROOT>/evolution-comparison/independent
<FORMAL_ROOT>/evolution-comparison/dependency-aware-joint
```

贡献者的逐步部署、数据校验、profile 预检、Decision LLM 连通性、两批次 smoke、GPU 排队、进度查询、失败恢复、证据归档和完整审计命令见[技术一贡献者实验执行手册](contributor-execution-runbook.md)。正式 Issue 必须给出精确 commit、私有 profile 哈希、数据哈希、模型版本、seed、batch、预算、GPU 和输出根；不能只写“运行技术一”。

具体的 smoke、正式十 seed 比较和 100/300 批长程任务参数见[技术一实验参数冻结卡](technical-one-experiment-parameter-card.md)。贡献者只能执行 Issue 选择的任务类型，不能把默认值自行改成另一种实验。

## 8. 结论边界

技术一要证明的是“可审计的 Harness 自我改进过程能够在相互依赖组件、候选验证和长程运行约束下工作”。它不要求每次 Decision LLM 提议都被接受，也不把高拒绝率自动解释为系统故障；拒绝必须按 regression、duplicate、invalid proposal、执行失败等原因分解。

但在正式结论中，若 invalid proposal 占比过高，必须作为 proposal 质量问题报告，并不能用“系统稳定”掩盖。只有完成统一十种子实验、对照实验和长程实验后，才能报告技术一的经验结论。
