# 服务器交接与运行顺序

本手册面向负责服务器实验的贡献者。所有正式实验必须关联到已审查的 commit 和 GitHub Issue；不得将服务器私有配置、数据集、模型权重、凭据或运行结果提交到公开仓库。

## 当前基线状态

本地固定基线是可复现的确定性脚手架，并非论文中的最终 ReAct 基线。外部对照候选及论文、代码和许可证核验记录在 `code/technologies/technology_one/baseline_matrix.json` 和 `baseline_research.md`。外部方法的复现门槛未解决前，不要安排正式服务器运行。

服务器配置样例为 `code/technologies/technology_one/server_profile.evolution.qwen.example.json` 和 `server_profile.baseline_matrix.qwen.example.json`；Action System 接口契约见 `code/technologies/technology_one/server_action_contract.json`。规范配置块为 `action_system`；旧私有配置中的 `execution` 仍作为兼容别名读取，并会发出弃用警告。

- `server_preflight.py`：不加载模型，检查数据、哈希、切分锁、方法和未填写的模型占位符。
- `server_adapters.py`：为 Decision LLM 策略、Task Model 训练器和评估器提供 JSON stdin/stdout 桥接。
- `server_runner.py`：预检通过后运行已登记方法；拒绝未解析占位符和 shell 字符串命令。
- 数据协议：`data_protocol.json`。
- 数据文件：`data/helpsteer2-train.jsonl` 和 `data/helpsteer2-validation.jsonl`，需在服务器单独准备。
- 数据来源：`nvidia/HelpSteer2@990b2711a36180dd19d9c94b8627844866f8982a`；许可证为 `CC-BY-4.0`。

冻结的本地脚手架报告 held-in accuracy `0.36393196598`、held-out accuracy `0.36502463054`。这些数字仅用于检查流水线，不是方法效果结论。

## 首次部署顺序

第一阶段先运行冻结的技术一固定对照，再开展更大的 Decision LLM 对照矩阵。正式使用 final-test 前，依次完成：

1. 准备源码、归一化 JSONL、协议、依赖、冻结清单和基线配置；不要复制凭据或本地虚拟环境。
2. 填写私有 Qwen profile。基线矩阵使用 `experiment_scope: core_matrix`；`final_test_unlocked` 必须为 `false`，Task Model 路径须指向已固定版本的本地模型。自进化实验使用 evolution profile，并保持相同的数据、模型、seed 和候选预算。
3. 运行基线预检和导入检查。
4. 使用 `server_technical_one_baseline` 运行一个 pilot seed。
5. 检查 `summary.json` 中的训练组哈希、selector 截断审计、held-in/held-out 指标、模型来源信息和预算。
6. pilot 审查通过后，使用 `run_technology_one_baseline_batch.sh` 运行配对随机种子。
7. 使用 `run_baseline_matrix.sh` 或 `run_baseline_matrix_batch.sh` 运行可部署的基线/控制/适配矩阵。外部论文方法必须标为任务适配，除非单独完成正式复现。
8. 使用相同配置方案、随机种子、组预算和评估限制运行最终 natural-support 候选矩阵。
9. 固定对照与候选矩阵均完成且选择/统计分析冻结后，才能解锁 `final_test`。

服务器环境变量、预检、Decision LLM 连通性检查、冒烟测试和正式运行命令见[贡献者实验执行手册](contributor-execution-runbook.md)。技术一代码目录中的 `code/technologies/technology_one/server_handoff.md` 是本文件的兼容入口，不再维护重复内容。

## 统一验收条件

- 仅 `held_in` 可用于候选选择。
- 候选选择期间，`held_out` 只用于报告回归情况。
- `final_test` 仅可在明确解锁后用于最终报告。
- 每个候选都记录父 Harness 版本和原子化接受/拒绝记录。
- 被拒绝候选不得改变当前 Harness 或检查点。
- 延迟反馈必须保留来源批次与释放批次。
- 每次运行保留数据集、Harness、模型、代码和检查点标识。
- 在实现真实 LLM 策略前，真实服务器基线应与 `deterministic-scaffold` 分开标注。
- Action System 负责提供许可操作、模型/训练器/评估器适配器、预算和审计要求；不表示每次模型执行本身都是 Harness 修改。
