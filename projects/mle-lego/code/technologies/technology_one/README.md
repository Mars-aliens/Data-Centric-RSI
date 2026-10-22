# 技术一代码

当前实验入口和环境配置见 [`docs/server-quickstart.md`](../../../docs/server-quickstart.md)。正式运行矩阵见 [`experiments/technology_one/formal-protocol-v1.md`](../../../experiments/technology_one/formal-protocol-v1.md)。

当前代码模块：

- `long_horizon_task.py`：数据分组、批次和延迟反馈；
- `long_horizon_runner.py`：固定流水线和运行恢复；
- `candidate_proposal.py`：候选 bundle 生成与校验；
- `candidate_sandbox.py`、`candidate_shadow.py`：隔离运行、输出验证和故障注入；
- `qwen_preference.py`：Task Model 预测、LoRA 训练和 checkpoint 续训。

旧 Harness 实验材料位于 `archive/legacy_lego/`。部分旧算法模块仍被共享适配器和历史测试引用，暂留在本包中；它们不是当前实验入口。

注意：周期性多候选调度尚未实现。单候选 smoke 不能作为正式递归实验结果。
