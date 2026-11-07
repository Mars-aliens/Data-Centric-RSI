# 技术二：数据经验持续治理

技术二研究 Data System 如何从运行数据中形成可复用经验，并根据来源、适用条件和后续数据变化决定经验是否继续参与数据选择。

## 当前实现

- `experience_lifecycle.py`：经验来源记录、验证、失败计数、上下文变化失效、冲突处理、过期及事件恢复接口。
- `experiment_policy.py`：可复现的四臂实验策略。可执行经验是固定的多数类 2:1 重放上限规则；治理臂按 HelpSteer2 revision 与已释放标签分布匹配，并在条件变化时重新做标签覆盖校验。
- `technology_one/long_horizon_runner.py`：新增训练数据策略扩展点；策略状态随批次状态提交及恢复，逐批 summary 记录输入/保留/丢弃条数、经验来源与状态。
- `scripts/run_technology_two_qwen.py`：使用固定 HelpSteer2 长周期任务和本地 Qwen Task Model 执行一个实验臂，可断点续跑。
- 执行协议：[`experiments/technology_two/protocol-v1.md`](../../../experiments/technology_two/protocol-v1.md)。

运行不需要 Decision LLM、Docker 或 Kubernetes。Task Model 路径、不可变 revision 与 GPU 资源由服务器私有配置提供；依赖沿用项目 `requirements-server.txt`。目前代码及合成输入测试可在本地验证，真实数据/GPU smoke 尚未运行。

该实现验证一种明确的数据选择经验，不代表任意 Memory/Skill 已实现。治理臂的自动验证是标签覆盖检查；它不能代替与对照臂的下游效果比较。
