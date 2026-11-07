# 延迟反馈归因与多批次版本选择实验

本实验当前执行的是候选修改隔离实验长周期结果上的离线反馈回放，不需要重新加载模型或 GPU。它比较单来源批次选择、按到达顺序错误归因的诊断基线、以及按来源版本归因的多批次选择。完整边界和步骤见[实验协议 v1](protocol-v1.md)。

前置条件：每个 seed 已有完整的候选修改隔离实验 run，summary 包含 stable/candidate 的来源批次指标。源实验的运行环境按 [`docs/server-quickstart.md`](../../docs/server-quickstart.md) 准备；本实验的离线回放只需要项目 Python 依赖。

为每个 seed 分别运行一次：

```bash
"$TRAIN_PY" scripts/run_technology_three_replay.py --t1-summary "$T1_ROOT/seed-17/summary.json" --output-dir "$T3_ROOT/seed-17" --window-batches 3 --outcome-window-batches 3 --min-mean-gain 0.0 --max-regression 0.05
```

将上面命令中的 `seed-17` 分别替换为 `seed-23`、`seed-29`，再汇总：

```bash
"$TRAIN_PY" scripts/summarize_technology_three.py --root "$T3_ROOT" --seeds 17 23 29 --output "$T3_ROOT/technology-three-summary.json"
```

每个 seed 产出 `feedback-trace.json` 和 `summary.json`。离线回放不会实际晋升候选版本，因此不能据此报告线上回滚效果。
