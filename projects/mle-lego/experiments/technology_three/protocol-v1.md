# 延迟反馈归因与多批次版本选择实验协议 v1

版本：`t3-delayed-feedback-replay-v1`
状态：离线回放 pilot 协议；在线晋升/回滚与正式统计方案尚未实现或冻结。

统一 QA、baseline、指标与图表呈现见[实验总蓝图 v1](../experiment-blueprint-v1.md)。

## 研究问题

系统更新后，能否把延迟返回的质量信号正确关联到产生该信号的数据批次和系统版本，并根据多个后续批次决定候选更新是否值得保留，避免单批偶然波动或错配反馈导致不稳定决策。

## 当前可执行范围

本实验复用已完成的候选修改隔离与运行连续性实验的 stable/candidate 配对 HelpSteer2 流运行结果，不再次训练模型。`scripts/run_technology_three_replay.py` 从源 run 的 `summary.json` 提取每个来源批次的 stable/candidate 预序列指标和可用批次，生成 `feedback-trace.json`，再对同一轨迹离线比较三种决策规则。它产生保留/继续观察建议，不在线切换 Task Model，也不覆盖源实验输出。

输入的源 run 必须完整结束，且包含 stable 与 candidate 的来源对齐指标，并以 `final_test_used=false` 完成。本实验不读取 held-out/final-test 来选择版本。若没有足够候选配对批次，回放器报错或给出观察不足，不制造对比数据。

## 对照策略

| 策略 | 决策信息 | 用途 |
| --- | --- | --- |
| `single_source_batch` | 第一个来源批次的 stable/candidate 差值 | 单批次选择基线，用来测量对偶然波动的敏感性 |
| `arrival_as_current` | 按反馈到达顺序，把信号当作当前版本反馈 | 不安全诊断基线；只统计错归因和可能决策，不作为可部署方案 |
| `source_attributed_window` | 按原始来源版本配对，观察多个后续批次 | 本实验的主要方法：均值收益需超过阈值，且窗口内单批回退不超过界限；候选失败则拒绝 |

默认 pilot 参数：`window_batches=3`、`min_mean_gain=0.0`、`max_regression=0.05`（Macro-F1 绝对差）。这些值仅用于工程/方法 pilot；正式实验前须根据预注册目标冻结，不得查看 held-out 后调整。

## 数据、模型与配对

- 数据与任务沿用 HelpSteer2 固定 revision、group split、预序列评价及源实验的反馈延迟设置。
- Task Model、candidate 由相同源 run 产生，不新增模型或依赖。
- pilot 对每个源实验 seed 独立回放，建议 seed `17, 23, 29`；每个输入 run 固定代码 commit、manifest、Task Model/Decision 配置和预算。
- 同一 seed 的候选和稳定版本必须使用相同来源批次、记录数和评价边界。不同 seed 的 manifest 可以不同。
- candidate 失败的批次作为失败观测处理，不把缺失指标当作零分。

## 指标与判读

- 反馈归因：反馈事件总数、版本/来源字段完整率、按到达时归因的错配数。
- 版本选择：单批/多批策略建议、观察窗口长度、均值收益、最差批次收益、候选失败拒绝数。
- 长周期证据：候选与 stable 的逐批质量轨迹、质量回退幅度、决策延迟批数、失败信号到达前的暴露批数。
- 系统完整性：源 run/manifest/config 哈希、feedback trace 可复现性、held-out/final-test 隔离。

当前离线回放不会真正将 candidate 提升为 stable，因此不能据此报告线上恢复时间或在线回滚成功率。该结论只能说明在已有源实验轨迹上，版本归因与多批选择规则会产生什么决策。验证真实在线晋升后的回滚，还需实现 runtime 闭环并开展独立故障/漂移实验。

## 运行步骤

1. 先完成候选修改隔离与运行连续性实验的 stable/candidate 长周期 run，并确认 summary 为 `status=complete`、包含来源版本指标且 `final_test_used=false`。
2. 每个 seed 单独回放：

```bash
cd "$LEGO_ROOT"
export T1_ROOT="$LEGO_ROOT/runs/technology-one-pilot"
export T3_ROOT="$LEGO_ROOT/runs/technology-three-pilot"
mkdir -p "$T3_ROOT"

for SEED in 17 23 29; do
  "$TRAIN_PY" scripts/run_technology_three_replay.py \
    --t1-summary "$T1_ROOT/seed-$SEED/summary.json" \
    --output-dir "$T3_ROOT/seed-$SEED" \
    --window-batches 3 --min-mean-gain 0.0 --max-regression 0.05
done
```

3. 汇总三种配对 seed：

```bash
"$TRAIN_PY" scripts/summarize_technology_three.py \
  --root "$T3_ROOT" --seeds 17 23 29 \
  --output "$T3_ROOT/technology-three-summary.json"
```

4. 人工检查每个 `feedback-trace.json` 的来源批次、版本和反馈可用批次，并核对到达顺序诊断策略标出的错配反馈。

## Smoke 与 Pilot 门槛

工程 smoke：用一个完整源实验 seed run 回放；必须产出非空比较轨迹、反馈到达顺序、三种策略结果，来源归因窗口不能使用未释放评分，held-out/final-test 均为 false。

Pilot：三个配对 seed 全部有完整源 run；汇总脚本检查来源完整且不混用不同 run；结果作为描述性方法 pilot，不作正式显著性结论。正式实验另发协议 v2，冻结候选生成/失败定义、反馈延迟条件、窗口阈值、种子数、统计检验、预算和在线晋升/回滚边界。

## 逐步执行与可验证输出

### 1. 准备源轨迹

版本选择离线评估需要每个 seed 至少 `decision_window_batches + outcome_window_batches` 个来源批次；默认 3+3=6 批，正式建议源实验至少 20 批以留出多次观察窗口。只含 2 批 smoke 的源 run 不能评估后续窗口。

先完成候选修改隔离实验，并检查每个源 `summary.json`：`status=complete`、stable/candidate 指标按 source batch 对齐、记录 `available_from_batch`、候选失败显式表示为缺失分数、`held_out_used=false` 和 `final_test_used=false`。每个 seed 独立目录，保存 source run、manifest、config、模型 revision 和代码 commit 的 hash。

### 2. 离线回放

每个 seed 运行一次，决策窗口和后续结果窗口不可重叠：

```bash
"$TRAIN_PY" scripts/run_technology_three_replay.py \
  --t1-summary "$SOURCE_ROOT/seed-17/summary.json" \
  --output-dir "$OUTPUT_ROOT/seed-17" \
  --window-batches 3 --outcome-window-batches 3 \
  --min-mean-gain 0.0 --max-regression 0.05
```

此处 pilot 阈值仅为占位参数。正式阈值必须在查看后续结果前冻结。输出应包含 `feedback-trace.json`、`summary.json` 中的 `future_window_evaluation`，并分别列出 decision source batches 与 outcome source batches；集合交集必须为空。future window 不足或 candidate 失败时标记 `outcome_complete=false`/不可评分，不把缺失值当零。

### 3. 反馈归因压力条件

先对同一 source trace 做 0/2/5 批延迟敏感性测试；这只改变事件 available time，用来检查归因逻辑。再对源实验用 `feedback_delay=0/2/5` 实际重跑，评估延迟对训练轨迹和选择建议的端到端影响。两类结果分开报告。

必须额外提供构造轨迹覆盖交错版本到达、同批多个版本、未知 version、缺 `source_batch_index`、缺 `source_version_id`、重复 feedback ID 等边界。缺来源或未知版本事件应被隔离为不可用于决策；重复 ID 应被拒绝。逐事件表保存 source/available/target/matched version、处理规则和是否进入决策。

### 4. 判定多批选择是否有预测价值

每个完整 seed 以 0–2 批的到期来源反馈形成建议，用后续 3–5 来源批次作为独立结果窗口。对单批、到达顺序诊断策略和来源归因窗口分别计算：有完整结果窗口的建议数、错误保留建议数/率、错误拒绝建议数/率、决策可用批次、后续窗口平均与最差 Macro-F1 差。错误保留率分母只包含“建议保留且可评分”的 run；错误拒绝率分母只包含“建议保持 stable 且可评分”的 run。尚无足够未来窗口、candidate failure 或反馈仍 pending 时列为不可评分，不作插补。

结果图：source batch 质量轨迹并标决策可用点、反馈延迟；延迟×策略错误归因率热图；每 seed 决策建议与未来结果配对点图。结果表：每策略有效分母、false accept/reject count/rate、未评分数、窗口 gain、延迟和源 hash。

### 5. 汇总与结论边界

运行：

```bash
"$TRAIN_PY" scripts/summarize_technology_three.py \
  --root "$OUTPUT_ROOT" --seeds 17 23 29 \
  --output "$OUTPUT_ROOT/technology-three-summary.json"
```

汇总器必须校验三个 seed 的源 run 唯一、输入完整、held-out/final-test 未使用，并输出未来窗口有效分母。结论只支持“离线选择建议是否与不重叠的后续观测一致”。真实线上晋升、回滚成功率、暴露时长和恢复时间，必须等版本 runtime 执行器与回退状态机实现后另做在线故障实验；不得从本回放推断。
