# 候选修改隔离与运行连续性实验协议 v1

版本：`t1-safe-evolution-pilot-v1`
状态：本地工程 pilot 协议；正式多种子统计协议尚未冻结。
适用代码：本文件所在仓库当前 commit。
主执行手册：[技术一安全自进化执行手册](../../docs/technology-one-safe-evolution-runbook.md)

统一 QA、指标与图表呈现见[实验总蓝图 v1](../experiment-blueprint-v1.md)。

## 1. 研究问题

技术一研究 Data System 在持续运行时能否执行由 Decision LLM 提出的候选修改，同时将候选错误限制在隔离分支，使稳定分支继续处理后续批次。主要结论对象是候选隔离、稳定运行连续性、状态完整性和失败恢复，不是单批任务分数是否提高。

当前代码的实验边界是：一份候选源码 bundle 在整条受控 HelpSteer2 流中运行；候选模型可从自己的 parent checkpoint 续训；稳定分支独立运行并保持 active。该 runner 尚未实现每批自动重新提案，也不负责将候选晋升为稳定版本。候选通过安全门只代表“允许进入后续观察”；长期保留/回退由延迟反馈归因与版本选择实验处理。

本技术点当前覆盖 Harness/Task Model 执行路径的候选源码和有限训练配置修改。完整的数据构建策略、Memory/Skill 与所有工具编排尚未全部接入候选运行接口，不能宣称已经验证任意 Data System 组件均可安全演化。

## 2. 对照边界

当前可执行的比较条件如下：

| 条件 | 运行方式 | 当前用途 |
| --- | --- | --- |
| `B0-fixed` | 只运行固定 HelpSteer2 流式 Data System，不生成或运行候选 | 建立稳定流水线的完成率、质量、成本和恢复基线 |
| `T1-shadow-clean` | 使用相同任务流运行固定稳定分支，并由固定可信 runner 在独立本地 candidate 分支运行受限数据策略；不注入故障 | 检查候选正常运行、双分支状态和额外成本；候选与稳定指标分别报告 |
| `T1-shadow-fault` | 与 `T1-shadow-clean` 相同，但对候选注入一种预先指定故障；每种故障使用独立 run | 检查错误是否被检测并限制在候选支路，稳定分支是否继续完成 |
| `T1-resume` | 在批次边界中断后，用同一 run ID、配置和输出目录恢复 | 检查稳定 runner、反馈队列和候选支路的恢复及幂等性 |

`B0-fixed` 与本实验的 stable branch 是配对运行控制。它们用于检查加入候选机制后稳定分支是否仍能完成相同任务流，不单独证明候选修改有效。故障注入是安全属性测试，不是另一种演化算法。

当前对照用于评估固定流水线、候选隔离、故障检测和恢复。不得在真实稳定目录或共享服务器上运行不受限的 LLM 生成代码。外部方法比较需在完成可复核的 HelpSteer2 适配后另行设计。

## 3. 共同任务与配对条件

- 数据集：`nvidia/HelpSteer2` 固定 revision `990b2711a36180dd19d9c94b8627844866f8982a`。
- 任务：按完整 prompt group 划分的 `preferred/rejected/tie` 质量代理分类；标签定义、信息边界和延迟规则以 [`long-horizon-task-definition.md`](../../docs/long-horizon-task-definition.md) 为准。
- Task Model：pilot 建议固定本地 `Qwen/Qwen3-8B` causal LM；运行者填写不可变 snapshot revision。正式结果不得混合不同模型、Tokenizer、LoRA recipe 或权重 revision。
- Decision LLM：只在生成真实研究候选时使用；pilot 建议先筛选配置好的 Qwen3.8-Flash API。模型筛选不读取 held-out/final-test，也不按下游任务分数挑模型。确定性 smoke fixture 不调用 Decision LLM。
- 配对键：Task Model seed、`stream_seed`、数据 manifest、`groups_per_batch`、反馈延迟、训练配置、代码 commit 和资源预算均相同。
- `held_out` 只用于可信报告；`final_test` 在研究协议冻结前不访问。summary 中 `held_out_used` 和 `final_test_used` 必须为 `false`。

## 4. 本地准备与服务器 smoke

pilot 固定使用 `seed=17`、`stream_seed=17`、`groups_per_batch=50`、`feedback_delay=2`、`max_batches=4`。此流前缀足以在第 4 批开始时释放第 0 批反馈，因而可检查一次真实的预测、延迟释放和 parent-checkpoint 续训链路。Task Model smoke 配置先设 `max_steps=1`；该值只用于缩短工程验证，不是正式训练配方。

按下列顺序运行，每项使用全新的 output directory：

1. 生成真实数据 manifest，审计 revision、源数据哈希、分组泄漏、类别统计、可用/排程批次数与最后一批大小。
2. 运行 `B0-fixed` 的 4 周期部署 smoke，确认完成 `4/4`，且至少发生一次到期反馈后的训练与 checkpoint 更新。
3. 用固定合规输入完成 Decision LLM endpoint/schema smoke 和提案质量筛选；不把此筛选计入任务效果实验。
4. 用 `scripts/create_t1_smoke_candidate.py` 生成行为不变的候选 fixture，运行 `T1-shadow-clean`。确认稳定与候选在第 0 至 3 批均完成、候选第 3 批收到已释放反馈并能保存自己的 checkpoint。fixture 只测部署链路，不计作 RSI 候选或方法结果。
5. 为独立的候选 fixture 设置 `fault_injection: {"3": "invalid_protocol"}`，运行 `T1-shadow-fault`。确认候选在故障批被拒绝并停止，稳定分支仍完成 `4/4`，稳定 checkpoint 前后哈希相同，故障和候选成本可审计。
6. 用通过本地 proposal/bundle 检查的真实 Decision LLM 候选执行独立 pilot。候选失败可作为被隔离的失败记录，但不能把 fixture 结果计作 LLM 自进化结果。
7. 用相同 run 配置在批次边界中断并恢复一次，确认已提交批次不重复、反馈不重复释放、run signature 不变。恢复测试与干净 smoke 分开报告。

具体环境、API、GPU、配置和命令见执行手册。默认 `local_shadow` 只运行固定可信 adapter，并把 candidate 输出和 parent checkpoint 放入独立分支；它不是容器或任意代码安全沙箱。Kubernetes backend 是可选增强，不是本 smoke 的前置条件。新 runner 的 `--max-batches 4` 会将确定性流限制为前四批；manifest 同时记录完整可用批数、实际排程批数和限制值。Smoke 通过后按执行手册运行 10～20 周期 pilot，再确定正式运行规模。

## 5. 故障覆盖

候选沙箱当前支持以下故障，正式故障覆盖测试每个故障使用一个独立 run，在 `batch_index=3` 注入，不得把多个故障塞进同一 run：

| 故障类型 | 检查性质 |
| --- | --- |
| `timeout`、`nonzero_exit` | 执行失败检测、候选终止、稳定分支继续 |
| `invalid_json`、`invalid_protocol` | 响应解析与协议门控 |
| `wrong_parent` | parent-version 绑定 |
| `missing_prediction`、`invalid_label` | 输出覆盖率和标签域校验 |
| `budget_overrun` | 报告训练步数不能越过每批上限 |
| `write_protected_parent` | 父 checkpoint 只读与写保护 |

工程 pilot 只要求先通过一种端到端注入（`invalid_protocol`）；其余故障先由本地契约测试覆盖，再逐类进行服务器验证。当前 `local_shadow` 仅执行受信任的固定 adapter，故障注入验证的是协议门控、candidate 分支和 stable 状态保护；它不能证明任意候选代码的安全性或操作系统级资源限制。候选模型报告的成本须标明为“模型报告值”，并与宿主 GPU/进程监控分开保存。

## 6. 主要指标与 pilot 通过门槛

主要稳定性指标：

- 稳定流完成率：完成批次 / 计划批次；记录非预期中断和恢复结果。
- 故障隔离率：稳定流仍完整完成且稳定父 checkpoint 未变化的注入故障数 / 已注入故障数。
- 故障检测率：被安全门拒绝并给出可审计 failure code 的注入故障数 / 已注入故障数。
- 状态完整性：批次提交、反馈释放、预测 artifact/checksum、checkpoint lineage 是否连续且无重复。
- 恢复能力：中断前后状态是否一致、恢复成功率与恢复耗时。
- 任务与成本：stable/candidate 分别报告预序列 Macro-F1、accuracy、训练步骤/耗时、预测耗时、候选失败成本和额外运行时间。

部署 smoke 通过条件：

1. manifest 与泄漏审计通过，真实数据的 group 与标签统计可复核；
2. `B0-fixed` 与 `T1-shadow-clean` 的稳定分支各自完成 `4/4`，final-test 未访问；
3. clean fixture 候选完成 `4/4`，候选和 stable checkpoint 不互相覆盖；
4. `invalid_protocol` 被识别，候选停止，stable 仍完成 `4/4`，stable checkpoint 前后哈希一致；
5. 至少一次批次边界恢复通过，且同一批次/反馈没有重复提交。

部署 smoke 通过后运行 10～20 周期 pilot，评估稳定运行、隔离、恢复和资源成本。根据 pilot 结果再冻结正式实验协议与规模。任何 smoke 门槛失败都先修复并重跑 smoke。

## 7. 正式实验仍需冻结

工程 pilot 通过后，维护者需发布新版本的正式协议，并在首次正式运行前冻结：

- 固定全流长度（不超过数据可用 development groups）、正式 Task Model recipe、Decision LLM API ID 与调用日期；
- 随机种子列表、重复数、故障类型/位置/频率和中断恢复计划；
- 各臂的 GPU/CPU/内存/时间/Token/训练步数总预算，以及失败尝试成本的计入方式；
- 主要统计终点、配对统计方法、稳定性非劣界值和停止条件；
- 是否加入合法且实现完成的外部方法适配，以及对应验收测试。

这些值不能用 smoke 配置代替，也不能在看到 held-out 结果后再调整。HelpSteer2 的 development 流约可支持百批量级，但服务器 manifest 和预估训练成本必须先实测；不得重复记录来凑运行长度。冻结前不访问 `final_test`。

## 8. 结论边界

本协议能验证的是受控 HelpSteer2 回放中的候选隔离、稳定流水线连续性、版本/反馈完整性和受控故障恢复。它不能单独证明：开放环境中的任意代码均安全、真人反馈/自然时间漂移有效、候选修改长期提升数据质量、该实验优于所有 Self-Harness 方法，或完整 Data-Centric RSI 已实现。上述更强主张必须由对应的外部方法比较、多种子长周期结果和三类机制组合实验支持。

## 9. 按步骤执行与结果验收

### 9.1 运行前准备

1. 在服务器完成 [`docs/server-quickstart.md`](../../docs/server-quickstart.md)，确认真实数据审计、Task Model 固定流水线 smoke 和 Decision LLM API smoke 已通过。
2. 从同一代码 commit 生成数据 manifest；保存 manifest 文件和 SHA256。检查 `development_batch_count`、三类 label counts、最后一批大小、无跨 split prompt group、final-test 未进入 agent bundle。
3. 固定本轮 Task Model、Decision LLM、Tokenizer、LoRA、候选策略、预算、GPU 分配和命令环境；保存配置副本及 hash。正式结果不混合不同配置。
4. 所有 run 使用新目录，命名为 `runs/candidate-isolation/<condition>/seed-<seed>/`。不得覆盖 smoke、pilot 或失败 run。

### 9.2 工程 smoke

依照[服务器执行手册](../../docs/technology-one-safe-evolution-runbook.md)顺序运行 `B0-fixed`、clean fixture、`invalid_protocol` 故障 fixture 和一次批次边界恢复。Smoke 只回答部署链路与状态不变量是否成立，不进入质量收益统计。

每个 smoke 必须核对：4/4 批完成；稳定和候选 checkpoint 路径互不重叠；故障 fixture 在注入批有 failure code 并停止；stable checkpoint/hash 不变；反馈 ID 无重复/丢失；恢复组与不中断组的 batch ID、feedback ID、prediction checksum、checkpoint lineage 和 confusion matrix 完全一致；held-out/final-test 均未用于决策。任一项失败则停止 pilot，保留目录和错误记录。

### 9.3 候选效果 pilot

1. 预先冻结候选修改模板及 proposal rubric；每个 seed 记录全部 Decision LLM 提案，包括无效、no-op、duplicate、拒绝和失败提案。候选 bundle 需针对该 seed 的 parent version 重新生成并验证，不能复用已绑定其他 parent 的 bundle。
2. 用 `seed=17,23,29` 配对运行固定 stable 与候选分支，每个 run 10～20 批，流、初始模型、训练预算、反馈时序完全一致。候选故障注入另开 run，不混入 clean 效果臂。
3. 每个 seed 结束后保存 `summary.json`、`run_state.json`、stable/candidate 预测 artifact、candidate trial audit、配置和 manifest hash、Decision LLM token/cost、宿主 GPU 监控。保留全部失败样本。
4. 按来源批次合并 stable/candidate Macro-F1，计算每批配对差；按 seed 报平均差、最差批差、正收益批比例。报告数据选择前后类别计数及成本。仅当数据选择记录可由 ID/hash 重算时，才对“改变了哪些训练数据”作结论。

Pilot 图表：来源批次质量折线与 seed 配对差图；每批输入/训练/丢弃数据计数图。表格：逐候选的提案 hash、parent、门控结果、来源批次质量差、训练与 LLM 成本、失败原因和终态。不得只报告 accepted candidates。

### 9.4 故障覆盖与恢复实验

Pilot 后，对 `timeout`、`nonzero_exit`、`invalid_json`、`invalid_protocol`、`wrong_parent`、`missing_prediction`、`invalid_label`、`budget_overrun`、`write_protected_parent` 分别运行独立故障试验。每类固定注入批次、使用独立 run；至少 3 个配对 seed 才汇总为 pilot 故障率。命令行故障配置写在私有 candidate config 的 `fault_injection` 映射中，例如 `{"3":"invalid_protocol"}`。

恢复试验至少覆盖批次 1、3 和 pilot 中点；每个中断点都与相同 seed/config 的不中断组配对。进程中断与子命令失败分开。验收要求 logical state 完全相同、重复/漏批和重复/漏反馈均为 0；耗时可以不同。

### 9.5 本地汇总与结果结论

先运行本协议第 6 节定义的字段完整性检查，再输出逐故障隔离表、seed 配对质量表、恢复等价表。最终报告必须分开回答：候选故障是否被隔离、候选是否改善了数据/任务质量、运行成本是多少、恢复是否逻辑等价。流程 smoke 通过不能写成候选方法有效；10～20 批 pilot 不能写成完整长周期 RSI 已验证。

当前候选 runner 不按批次自动重新提案，也不自动将候选提升为后续 active 版本。若研究问题要求递归候选调度和多轮晋升，必须先实现该 runtime，再新增协议版本和独立实验，不能用当前 shadow pilot 代替。
