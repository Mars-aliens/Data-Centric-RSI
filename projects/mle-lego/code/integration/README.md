# Data-Centric RSI 集成工作流

状态：端到端集成 smoke 入口已实现；真实 HelpSteer2/Qwen 长周期运行仍需在服务器执行。

本目录作为基于共享运行时 `core_system.mle_core` 组合三个技术模块的稳定集成边界。各技术模块必须保持可独立测试并提供带版本的契约；集成代码负责跨模块编排、配置和审计来源信息，不重复实现各技术模块。

当前入口包括 `DataSystemEvolutionRuntime` 和 `IntegrationOrchestrator`：T1 安全检查通过的候选先进入观察窗口，T3 按后续批次结果决定是否晋升，晋升后检测到退化可恢复到上一个稳定版本；T2 的经验状态写入同一事件账本。`PairedMeanRetentionPolicy` 是可替换的运行基线。

本地集成 smoke：

```bash
PYTHONPATH=code python scripts/run_integration_smoke.py \
  --all-configs --batches 4 --output-dir runs/integration-smoke
PYTHONPATH=code python scripts/summarize_integration_matrix.py \
  --root runs/integration-smoke \
  --output runs/integration-smoke/summary.json
```

该 smoke 验证 2×2×2 组合、候选失败后的批次连续性、候选晋升、晋升后回退和结果汇总。它使用确定性 fake executor，不替代真实 Task Model、数据集或服务器 GPU 实验。

详见 [长周期自进化框架](../../docs/evolution-runtime-architecture.md)。
