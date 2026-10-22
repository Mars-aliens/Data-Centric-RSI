# 技术一代码

当前实验入口和环境配置见 [`docs/server-quickstart.md`](../../../docs/server-quickstart.md)。运行顺序为 4 周期部署 smoke、10～20 周期 pilot，再根据结果确定正式规模。

当前代码模块：

- `long_horizon_task.py`：数据分组、批次和延迟反馈；
- `long_horizon_runner.py`：固定流水线和运行恢复；
- `candidate_proposal.py`：候选 bundle 生成与校验；
- `candidate_sandbox.py`、`candidate_shadow.py`：隔离运行、输出验证和故障注入；
- `qwen_preference.py`：Task Model 预测、LoRA 训练和 checkpoint 续训。

注意：周期性多候选调度尚未实现。单候选 smoke 不能作为正式递归实验结果。
