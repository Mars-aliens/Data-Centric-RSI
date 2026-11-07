# 数据经验持续治理实验协议 v1

版本：`t2-experience-lifecycle-v1`
状态：工程 pilot 协议；正式统计方案待 pilot 后冻结。

统一 QA、baseline、指标与图表呈现见[实验总蓝图 v1](../experiment-blueprint-v1.md)。

## 研究问题

带来源和适用条件的可执行数据经验，在 HelpSteer2 长周期流式处理里，能否在复用经验的同时控制数据分布变化引起的错误复用，并维持任务质量、运行连续性和可审计性。

## 实验臂

每个随机种子运行以下四臂。四臂使用同一数据流、模型、训练配置和资源预算，分别使用独立输出目录。

| 臂 | 持久状态 | 对后续训练数据的作用 |
| --- | --- | --- |
| `no_experience` | 不保存经验 | 使用全部已释放训练数据 |
| `reflection_only` | 保存批次标签分布反思 | 只记录反思，不更改训练数据 |
| `executable_unmanaged` | 保存可执行多数类上限 Skill | 首次观察到已释放标签后启用 2:1 类别重放上限；遇到数据条件变化仍继续使用 |
| `lifecycle_governed` | 保存带来源条件的 Skill 及状态 | 至少跨两个反馈批次观察且覆盖校验通过后才应用；数据版本/标签分布条件变化时暂停并重新校验 |

第四臂用于区分“可执行经验本身”与“生命周期治理”的贡献。治理臂首次经验及每次条件变化后，必须在同一条件连续观察两个反馈批次并通过覆盖校验后才允许复用。反思臂是文字反思对照，`no_experience` 是无持久经验基线。

当前可执行 Skill 是固定的确定性数据选择规则，不由 LLM 生成：对累计训练记录按类别计数，将各类别保留数限制为当前最少类别的 2 倍，并保留较新的记录。无治理臂首次观察后即执行；治理臂先保持 candidate，至少在两个反馈批次观察到兼容条件且覆盖校验通过后才执行。Smoke 使用 4 批流，因此预期只能观察到治理经验成为 verified；10 批 pilot 才能观察到经验被实际执行后的影响。该规则为本实验提供可复现的经验动作；结论只适用于该操作和本任务适配，不代表已验证任意 Memory/Skill。

## 固定任务与公平性

- 数据：HelpSteer2，revision `990b2711a36180dd19d9c94b8627844866f8982a`，使用仓库长周期任务定义的 group split 和预序列反馈。
- 模型：固定同一 Qwen Task Model 快照、Tokenizer、LoRA 参数和训练步数；路径与 revision 写入服务器私有 JSON 配置。
- 反馈：pilot 默认延迟 2 批；不得读取未释放标签。
- 配对键：`seed`、`stream_seed`、数据 manifest 哈希、批大小、反馈延迟、模型 revision、训练配置、代码 commit 完全一致。四臂的策略臂标识除外。
- `held_out` 仅作报告，`final_test` 不访问。每个 summary 必须报告 `held_out_used=false`、`final_test_used=false`。
- 每个臂单独执行和存储；同一 GPU 上串行运行，避免显存竞争。

## 阶段与参数

1. **工程 smoke**：`seed=17`、`stream_seed=17`、`groups_per_batch=50`、`feedback_delay=2`、`max_batches=4`；四臂各跑一次。确认完成 4/4、治理臂有逐批审计、held-out/final-test 未使用。
2. **Pilot**：smoke 全通过后，每臂使用配对种子 `17, 23, 29`，`max_batches=10`。资源允许时可扩到 20 批，但不得中途改变配置或复用输出目录。
3. **正式实验**：pilot 后另行发布协议 v2，冻结完整流长度、种子数、统计检验、主要终点、预算、停止条件和资源安排。v1 结果不作为正式显著性结论。

工程 smoke 使用 `max_steps=1` 的单独模型配置；pilot 使用冻结的训练配置（示例模板 `max_steps=100`，执行者须先确认资源可行）。四臂不得混用 smoke 与 pilot 模型配置。Pilot 运行器会把同一 task seed 初始样本与到当前批为止已释放的全部反馈作为累计 replay 候选集，再由经验治理策略选择训练记录。

## 指标

- 任务质量：每臂按批次及全流汇总的预序列 macro-F1、accuracy。
- 生命周期：经验启用批次、适用条件变化次数、暂停/重新验证次数、条件不匹配时的误复用次数、来源完整率。
- 数据与长周期：每批训练记录数及丢弃数、类别分布、累计训练数据量、批次完成率、反馈重复/丢失数、checkpoint 恢复成功率、训练/预测时间和硬件资源。

主比较顺序：`lifecycle_governed` 对 `executable_unmanaged`（治理的增量作用）；`executable_unmanaged` 对 `reflection_only`（执行经验的作用）；`reflection_only` 对 `no_experience`（文字记忆作用）。任务质量和稳定性并列报告，不以单一分数宣称长周期稳定性提升。

## 输出和验收

每个 run 目录保存 `run_state.json`、`summary.json`、逐批预测与命令账本。summary/config 哈希必须区分实验臂；批次历史记录经验状态、是否应用和数据保留数量。

smoke 验收：每臂 `status=complete` 且 4/4；各臂数据 manifest 哈希、模型 revision、批参数和随机种子一致；治理臂记录经验来源/条件/状态，条件变化时先重新验证；反思臂训练数据未被改变；held-out 与 final-test 未参与决策。失败时保留失败目录，用新 run ID 重跑。

## 结论边界

本实验检验 HelpSteer2 受控流中的一条类别重放经验及其失效治理。标签分布条件是已释放数据的确定性摘要，不代表所有真实场景漂移。当前生命周期校验检查数据覆盖，不是经验策略下游效果的独立因果验证；任务质量仍需由四臂预序列结果比较。smoke 或单种子 pilot 不能证明普遍 RSI 能力。

## 逐步执行方案

### 1. 冻结数据流

同一批原始 HelpSteer2 和任务定义生成四种 `stream_regime`：

| `stream_regime` | 顺序规则 | 验证用途 |
| --- | --- | --- |
| `randomized` | 按 seed 对全部 development prompt groups 确定性哈希排序 | 保留现有受控随机流基线 |
| `interleaved` | 以 group 的 dominant label 为类别轮转排序 | 近似稳定混合比例 |
| `abrupt` | 按 dominant label 分块排序 | 形成一次显著类别比例变化 |
| `alternating` | 按类别轮流放入两批大小的连续区块 | 形成反复类别变化 |

`dominant label` 定义为 group 内数量最多的派生标签，平票按固定标签顺序打破。四种流只重排同一个 development group 集，不增删或重复 group；离线构造可读取标签，运行时模型/policy 仍受反馈释放边界约束。每次运行保存 `stream_regime`、`stream_order_algorithm`、`stream_schedule_sha256` 和 `development_batch_label_counts`。检查流之间 group ID 集合完全相同，再开始训练。

顺序算法只能重排现有 group，不保证一定产生目标分布。运行前计算每批真实标签比例：`interleaved` 的各类别批次比例极差应不超过 0.10；`abrupt` 与 `alternating` 的预定变化窗口间总变差距离 `0.5 × Σ|p_before-p_after|` 应至少为 0.20。阈值在启动模型实验前固定。如果真实 manifest 达不到门槛，停止该流条件，不训练、不把它记成漂移实验；先报告 HelpSteer2 group 构成不足，之后再决定是否另建合法的 group-level 受控构造任务。不得拆分 group 或重复样本来制造变化。

命令示例：

```bash
for REGIME in randomized interleaved abrupt alternating; do
  "$TRAIN_PY" scripts/prepare_long_horizon_task.py \
    --train-jsonl "$TRAIN_JSONL" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --output-dir "$LEGO_ROOT/runs/t2-stream-manifests/$REGIME" \
    --stream-seed 17 --stream-regime "$REGIME" \
    --groups-per-batch 50 --feedback-delay 2
done
```

当前 runner 在每次执行时即刻构造并写入 summary 的 task manifest；上面的准备命令用于提前审计流结构，实际 run 仍须核对自己的 `stream_schedule_sha256` 与审计版一致。

### 2. 工程 smoke

固定 `seed=17`、`stream_seed=17`、`groups_per_batch=50`、`feedback_delay=2`、`max_batches=4`。先用 `interleaved` 流跑四臂；每臂单独目录、同 GPU 串行。按 README 命令执行并确认四个 summary 都是 `complete`、4/4、manifest/config 配对一致、final-test 未访问。4 批只验生命周期状态字段和运行路径，不作为下游效果证据。

### 3. 经验治理 pilot

使用通过流构造门槛的流条件 × 四个经验臂 × seeds `17,23,29`，每 run 20 批，反馈延迟固定 2 批。最多 48 runs；不满足分布门槛的 regime 先排除并说明原因。如 20 批超出资源预算，可先以 10 批做运行成本 pilot，但须检查治理经验确实经历 `candidate → verified → needs_validation → verified`；未出现完整状态转移的流/seed 不计为治理有效性证据。每个 seed、每种流内，四臂完全配对。

运行命令在 README 的 pilot loop 中增加：

```bash
--stream-regime "$REGIME" --max-batches 20
```

并以外层 `for REGIME in randomized interleaved abrupt alternating` 包住 seed/arm 循环。每个 regime 使用独立汇总根目录，例如 `runs/t2-pilot/$REGIME/`，其下沿用 `$ARM/seed-$SEED/`，这样每个 regime 可单独传给现有四臂汇总器。每臂实际 `run_id`、输出路径必须唯一，pilot 配置和 smoke 配置分开。

### 4. 逐批验收和指标计算

每批 `training_data_policy` audit 必须包含 `observed_regime`、`conditions_before`、`conditions_observed`、`context_changed`、`experience_status_before`、`experience_status`、`experience_source_batch_index`、`experience_source_version_id`、`experience_applied`、`records_before/selected/dropped`、`record_ids_selected_sha256`。缺字段则该 run 不进入效果汇总。

- **执行经验作用**：`executable_unmanaged - reflection_only` 的 seed 配对 Macro-F1 差；同时报告保留率、类别训练量和成本。
- **治理增量**：`lifecycle_governed - executable_unmanaged` 的配对差；对每次突变报告变化发生、反馈可见、暂停生效、重验完成四个批次。
- **经验误用**：条件已改变仍 `experience_applied=true` 的批次数 / 条件改变后应停用的批次数。
- **暂停延迟**：分别报告 input regime change → feedback visible、feedback visible → policy suspended、regime change → suspended。
- **重验时长**：policy suspended → verified and applied 的批数；若直到流结束未重验，作为右删失/未重验报告，不填零。
- **平稳代价**：interleaved 中治理相对无治理的启用延迟、Macro-F1 差、数据丢弃和训练成本。

### 5. 汇总、图表和停止规则

对每种流分别运行 README 中的 `summarize_technology_two.py`；不同 regime 不得放入同一汇总根目录混算。画四臂逐批 Macro-F1 图（标注 regime change 与 experience state）、逐批类别样本保留图、每 seed 的 paired difference 图。表格列出每 seed/流/臂的完成率、误用数、暂停/重验批数、质量差、成本和 manifest hash。

若任一臂的输入 manifest、模型 revision、训练配置或反馈延迟不匹配，停止该组比较并修正后使用新 run ID。若 20 批内没有触发预期经验状态变化，只能报告“该流未激活测试条件”，不得将零误用解释为治理有效。不得查看 held-out/final-test 后调规则或阈值。
