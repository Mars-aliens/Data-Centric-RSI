# Data-Centric RSI 当前实验方案

版本：`experiment-plan-v2`

状态：当前执行方案。三个单项实验和跨技术集成 smoke 的代码已经在本地具备；真实 HelpSteer2、Task Model 和 GPU 实验需要在服务器执行。

## 一、研究目标

研究 Data System 在持续处理数据、更新模型和接收延迟反馈时，能否长期稳定运行并持续改进。

三个技术点固定为：

1. **数据策略隔离验证**：隔离试验候选数据处理策略，确认对数据质量和下游训练的影响后再接纳，降低失败更新干扰稳定流程的风险。
2. **数据经验持续治理**：跟踪经验与 Skill 的来源、适用条件和失效证据，及时更新或停用，减少过时规则长期误导数据处理。
3. **延迟反馈下的数据策略演化**：将反馈关联到产生该反馈的数据与系统版本，依据多轮效果调整策略，减少延迟反馈导致的错误判断。

三个技术点的关系是：先隔离一次候选数据策略更新，再治理长期积累的数据经验，最后利用来源正确的延迟反馈决定数据策略是否继续保留、暂停或回退。

## 二、统一实验条件

| 项目 | 固定设置 |
| --- | --- |
| 数据集 | HelpSteer2 固定 revision；同一 prompt 的 responses 聚合为 `preferred/rejected/tie` 三类 |
| 数据划分 | 按完整 prompt group 划分，不能拆分同一 group |
| 数据流 | development 数据按固定 seed 分批，每批 50 个 prompt groups |
| 反馈 | 当前批先预测，标签在设定延迟后释放；pilot 默认延迟 2 批 |
| Task Model | 所有需要训练的实验使用同一 Qwen Task Model、Tokenizer、LoRA 配置和训练预算 |
| Decision LLM | 只用于候选数据策略提案；模型、revision、Prompt hash 和调用成本全部记录 |
| 评价 | 逐批 held-in 预序列 Macro-F1、Accuracy、混淆矩阵、数据选择和运行成本 |
| 信息边界 | 未释放标签不能用于当前批预测或策略决定；held-out 只报告；final-test 在方案冻结前锁定 |
| 配对单位 | `seed + stream_seed + manifest + model revision + training config + feedback delay` |

smoke 只验证工程链路，不作为方法效果结论。pilot 用于估计效果、成本和方差；正式实验参数在 pilot 后冻结。

## 三、需要回答的实验问题

总共回答 10 个问题：

| 实验 | QA | 要回答的问题 |
| --- | --- | --- |
| 数据策略隔离验证 | QA-1 | 候选策略失败时，是否只影响候选分支而不污染稳定流程？ |
| 数据策略隔离验证 | QA-2 | 候选策略是否改善数据质量或下游任务质量，代价是多少？ |
| 数据策略隔离验证 | QA-3 | 运行中断后能否继续执行，且不重复或丢失数据和反馈？ |
| 数据经验持续治理 | QA-4 | 把经验变成可执行数据处理动作，是否比只写反思或不保存经验更有效？ |
| 数据经验持续治理 | QA-5 | 经验条件变化后暂停并重新验证，是否能减少过时经验误用？ |
| 延迟反馈下的数据策略演化 | QA-6 | 延迟或乱序反馈能否正确归因到产生它的数据和系统版本？ |
| 延迟反馈下的数据策略演化 | QA-7 | 多批次来源对比是否比单批决定更可靠？ |
| 跨技术组合 | QA-8 | 三个机制组合后，是否提高整条 Data System 的长周期稳定性？ |
| 跨技术组合 | QA-9 | 三个机制之间是否存在主效应或交互收益？ |
| 跨技术组合 | QA-10 | 候选策略被接纳后发生退化，系统能否检测、回退并恢复？ |

## 四、单项实验设计

### 4.1 数据策略隔离验证

**QA-1：候选失败是否污染稳定流程？**

对照为固定流程 `B0-fixed` 和同流 stable 分支。实验分支在独立 candidate 分支运行受限的数据策略；每次只注入一种故障：非法协议、超时、非零退出、非法 JSON、错误 parent、缺失预测、非法标签或预算超限。

每个故障单独运行，记录注入批次、failure code、candidate 终止状态、stable 完成情况、stable checkpoint 前后 hash、反馈重复/丢失情况和恢复结果。

主要指标：

- 故障检测率
- 故障隔离率
- stable 完成批次 / 计划批次
- stable checkpoint 不变率
- 重复反馈数、丢失反馈数
- candidate 失败检测延迟和额外成本

通过条件：candidate 被拒绝或终止，stable 完成计划批次，stable 状态和 checkpoint 不变，没有重复或丢失已提交反馈。

呈现：故障逐例表、stable/candidate 批次时间线图、各故障类型的检测率和隔离率图。

**QA-2：候选策略是否改善质量，代价是多少？**

固定 stable 流作为 baseline。candidate 从同一 parent 分叉，使用相同数据流、已释放反馈、Task Model、训练配置和预算。比较 stable/candidate 的来源批次 Macro-F1、Accuracy、混淆矩阵、输入/保留/丢弃数据和训练成本。

主要指标：

- `candidate_macro_f1 - stable_macro_f1`
- 每批和每 seed 的平均差、最差差值、正收益批次比例
- 各类别数据保留数和丢弃数
- 训练步数、训练时间、预测时间、GPU 显存
- Decision LLM 调用次数、token 和失败成本

呈现：stable/candidate 质量轨迹图、数据选择图、候选逐例结果表、seed 配对差值表。

该问题只判断候选策略的当前影响，不把 candidate 自动提升后的长期效果当作本问题结论。

**QA-3：中断恢复是否等价？**

对照为不中断运行；实验组在预定批次边界中断，再从同一 run state 恢复。另对命令失败和进程中断分别测试。

比较 scheduler cursor、已完成批次、feedback ID、预测 checksum、checkpoint parent 链、训练记录和最终指标。

通过条件：恢复结果与不中断运行一致，没有重复批次、重复反馈、丢失反馈或错误 parent；恢复过程和重试成本可审计。

呈现：恢复审计表、checkpoint lineage 时间线、不中断/恢复运行差异表。

### 4.2 数据经验持续治理

使用四个配对实验臂：

| 实验臂 | 行为 |
| --- | --- |
| `no_experience` | 不保存持久经验，使用全部已释放训练数据 |
| `reflection_only` | 保存文字式标签分布反思，但不改变训练数据 |
| `executable_unmanaged` | 经验首次观察后立即执行，条件变化后仍继续使用 |
| `lifecycle_governed` | 记录来源和条件，观察并通过覆盖校验后才执行；条件变化后暂停并重新验证 |

**QA-4：可执行经验是否有用？**

主要比较 `executable_unmanaged - reflection_only` 和 `reflection_only - no_experience`。所有实验臂使用相同数据流、模型、训练预算和反馈释放时间。

主要指标：逐批 Macro-F1、Accuracy、数据保留率、训练记录数、类别覆盖、训练和预测成本、完成率。

**QA-5：生命周期治理是否减少经验误用？**

主要比较 `lifecycle_governed - executable_unmanaged`。通过 `randomized`、`interleaved`、`abrupt`、`alternating` 四种受控数据流观察条件稳定、突变和反复变化。

主要指标：

- 条件变化后仍应用旧经验的批次数 / 应暂停批次数
- 经验暂停延迟
- 重新验证所需批数
- 经验状态转换：candidate、verified、needs_validation、suspended
- 突变后的质量下降和恢复批数
- 稳定流中的激活等待、丢弃数据和额外成本

呈现：四臂逐批质量图、经验状态时间线、类别数据保留图、seed 配对差值表、经验生命周期审计表。

如果某种数据流没有达到预先规定的类别比例变化门槛，则该流只能报告为无效流，不能当作漂移实验结果。

### 4.3 延迟反馈下的数据策略演化

该实验使用数据策略隔离实验的完整 source run，不重新训练模型。

对照和策略如下：

| 策略 | 行为 |
| --- | --- |
| `arrival_as_current` | 把反馈到达时的当前版本当作反馈来源，作为错误归因诊断 baseline |
| `single_source_batch` | 只使用一个来源批次决定是否保留候选 |
| `source_attributed_window` | 按来源版本匹配多个后续批次，再决定保留、继续观察或拒绝 |

**QA-6：延迟反馈归因是否正确？**

构造 0、2、5 批延迟，以及乱序到达、交错版本、未知版本、缺失来源字段和重复 feedback ID 的轨迹。

主要指标：来源版本匹配准确率、错归因数/率、字段完整率、unknown/pending 数、反馈从产生到可决策的批数。

缺失来源或未知版本的反馈必须进入 pending/unknown，不能猜测为当前版本，也不能参与版本选择。

呈现：来源批次到达批次矩阵、延迟×策略错配热图、逐反馈审计表。

**QA-7：多批次选择是否更可靠？**

用来源批次 `0–2` 形成候选决定，用不重叠的来源批次 `3–5` 验证后续结果。不能使用同一窗口既做决定又做验证。

主要指标：

- 错误保留率：建议保留但后续窗口失败
- 错误拒绝率：建议拒绝但后续窗口成功
- 决策可用批次
- 窗口平均 gain 和最差 gain
- 候选失败拒绝率
- 未来窗口不足、反馈 pending 和 candidate failure 数

呈现：版本质量轨迹图、策略决定与未来结果配对图、每 seed 决策表。

离线回放只能验证归因和选择建议，不能单独证明真实在线晋升后的回退成功率。

## 五、跨技术组合实验

三个二元因素组成 `2×2×2=8` 个组合：

| 因素 | 关闭 | 开启 |
| --- | --- | --- |
| 数据策略隔离 | 固定流程，不运行 candidate | candidate 独立分支、失败门控和状态保护 |
| 数据经验治理 | 不使用可执行经验 | 使用来源、条件、暂停和重新验证机制 |
| 延迟反馈策略选择 | 单批或固定版本决定 | 来源版本归因和多批次窗口选择 |

8 个组合必须使用同一数据流、seed、初始 checkpoint、模型、训练配置、反馈延迟和总预算上限。每个组合单独输出目录，不能删除失败组合后再汇总。

**QA-8：整体长周期稳定性是否提高？**

比较全关闭 baseline、单机制组合、两机制组合和三机制全开组合。

主指标：完成批次比例、无完整性违规运行比例、故障后恢复成功率、经验误用率、反馈错归因率、候选错误晋升数、Macro-F1 和总成本。

**QA-9：是否存在组合收益？**

对 8 个组合做配对 factorial 分析，报告三个因素的主效应、交互效应、seed 级差值和置信区间。不能只比较全关闭和全开启后声称每个技术点都有效。

**QA-10：接纳候选后退化能否回退恢复？**

在已晋升候选的运行中注入可复现退化，比较固定 stable 和带晋升/回退机制的系统。

主要指标：错误晋升数、退化检测延迟、回退成功率、恢复批数、恢复后的质量和 checkpoint lineage。

本地集成 runtime 已实现候选观察、晋升、退化回退和事件恢复；真实模型下的组合验证仍需服务器执行。

## 六、执行阶段

### 阶段一：本地代码验收

执行单元测试、组合 smoke 和结果汇总。组合 smoke 使用 fake executor，只验证接口和状态不变量。

```bash
cd "$LEGO_ROOT"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"

python scripts/run_integration_smoke.py \
  --all-configs --batches 4 \
  --output-dir runs/integration-smoke

python scripts/summarize_integration_matrix.py \
  --root runs/integration-smoke \
  --output runs/integration-smoke/summary.json
```

验收要求：8/8 组合完成 4/4，候选失败不停止稳定批次，晋升后退化可以回退，汇总器检查到 8 个配置。

### 阶段二：服务器单项 smoke

1. 数据策略隔离验证：clean 和登记故障各跑 4 批。
2. 数据经验持续治理：四个实验臂各跑 4 批。
3. 延迟反馈回放：等候选隔离实验产生完整 source summary 后，对一个 seed 跑回放 smoke。

服务器 smoke 只确认真实数据、模型、路径、GPU、反馈边界和结果字段可用，不作为方法效果结论。

### 阶段三：单项 pilot

- 数据策略隔离：每个 seed 10～20 批，覆盖预登记故障和中断恢复。
- 数据经验治理：四个实验臂，seed `17/23/29`，每个 run 10～20 批；四种数据流分别汇总。
- 延迟反馈选择：使用至少 6 个来源批次，建议每个 seed 20 批源运行，再用 3 批决定窗口和 3 批未来窗口验证。

pilot 只报告 seed 级效应量、成本和区间，不作正式显著性结论。

### 阶段四：真实组合 pilot

单项 smoke/pilot 通过后，使用真实 HelpSteer2 和固定模型跑 8 个组合。先使用 3 个配对 seed 和 10 批，确认成本、状态恢复和结果字段，再决定是否扩展到 20 批或更长周期。

## 七、实际运行入口

服务器执行者按以下顺序运行。所有命令都在项目根目录执行，并先设置：

```bash
cd "$LEGO_ROOT"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
```

1. **服务器准备和固定流程 smoke**：先执行[服务器快速开始](../docs/server-quickstart.md)，完成代码、数据、Task Model、固定流程 smoke 和 Decision LLM API smoke。
2. **数据策略隔离验证**：执行[技术一执行手册](../docs/technology-one-safe-evolution-runbook.md)，先 `--preflight-only`，再完成 clean smoke 的 4/4；然后再做登记故障和中断恢复实验。
3. **数据经验持续治理**：执行[技术二协议](technology_two/protocol-v1.md)，四个实验臂各跑 4 批 smoke；通过后运行四种流、seeds `17 23 29` 的 10～20 批 pilot。
4. **延迟反馈下的数据策略演化**：等技术一有完整 source run 后，执行[技术三协议](technology_three/protocol-v1.md)的离线回放和汇总，不需要 GPU。
5. **跨技术组合**：先运行下面的 8 组合 contract smoke；真实组合 pilot 需在三项单独 smoke/pilot 通过后执行。

本地/服务器集成 contract smoke：

```bash
"$TRAIN_PY" scripts/run_integration_smoke.py \
  --all-configs --batches 4 \
  --output-dir "$LEGO_ROOT/runs/integration-smoke"

"$TRAIN_PY" scripts/summarize_integration_matrix.py \
  --root "$LEGO_ROOT/runs/integration-smoke" \
  --output "$LEGO_ROOT/runs/integration-smoke/summary.json"
```

该命令使用 fake executor，只验证统一 runtime、8 个配置、候选失败、晋升、回退和汇总器，不替代真实数据和模型实验。

技术二 smoke 的直接命令见[技术二 README](technology_two/README.md)；技术三回放命令见[技术三 README](technology_three/README.md)；技术一变量创建、预检和 4/4 命令见[技术一执行手册](../docs/technology-one-safe-evolution-runbook.md)。

每个 run 完成后检查 `summary.json`：`status=complete`、批次数完整、`held_out_used=false`、`final_test_used=false`。失败目录保留，修复后使用新的 run ID 和输出目录。

## 八、固定结果文件

每个 run 至少保存：

```text
summary.json
run_state.json
batch_metrics.jsonl
data_selection.jsonl
experience_events.jsonl
candidate_events.jsonl
feedback_events.jsonl
recovery_events.jsonl
environment.json
```

每个结果必须包含数据 manifest hash、代码 commit、模型 revision、实验臂、seed、反馈延迟、配置 hash、批次完成数、held-out/final-test 标志和失败/恢复记录。

汇总前必须检查：配置匹配、批次完整、feedback ID 无重复、source version 可追溯、结果未使用 final-test。检查失败的 run 保留在结果目录，但不得进入主结果表。

## 九、结论边界

- HelpSteer2 是受控数据流，不代表真实用户在线漂移。
- smoke 通过只说明工程链路正确，不说明方法有效。
- pilot 结果用于估计效果和资源，不直接代表正式统计结论。
- local shadow 和本地 integration smoke 不等于任意代码的操作系统级安全隔离。
- 离线延迟反馈回放不等于真实在线晋升、回退和恢复。
- 只有真实服务器组合实验完成后，才能回答三个机制是否共同提高 Data System 的长周期稳定性。
