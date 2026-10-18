# MLE-LEGO

MLE-LEGO 是 Data-Centric Recursive Self-Improvement（RSI）的模块化研究项目，研究数据系统如何利用执行反馈、可复用经验和长程评估来改进自身的数据处理 Harness。目前实现集中于技术一；技术二、技术三及整体集成流程已预留目录，但尚未实现。

本仓库仅包含 LEGO 方向，不含 Aladdin 实验、模型权重、私有数据集、GPU 环境、凭据或既有运行结果。

## 项目架构

```text
code/
├── core_system/mle_core/          共享接口、运行时、数据与预算
├── technologies/
│   ├── technology_one/             已实现的 Harness 演化
│   ├── technology_two/             可执行经验复用（规划中）
│   └── technology_three/           连续反馈选择（规划中）
├── integration/                    跨技术编排（规划中）
└── local_experiments/lego/         兼容现有命令的导入路径
experiments/                        协议、清单和结果索引
docs/                               研究、服务器与贡献者指南
scripts/                            启动、审计和报告入口
data/                               仅含说明；数据集放在仓库之外
```

现有 profile 和服务器命令继续支持 Python 导入路径 `local_experiments.lego`。新代码应从 `technologies.technology_one` 导入。

## 研究范围

Harness 负责组织数据生成、筛选、过滤、训练、工具调用、反馈和经验复用。核心循环如下：

```text
execute -> observe feedback -> propose Harness edit -> evaluate candidate
-> accept on held-in improvement or roll back -> continue
```

技术一比较独立组件修改与依赖闭包联合修改。当前正式联合模式名为 `dependency-aware-joint`，要求提案合法地同时修改多个组件，例如 `generator + selector`。决策模型保持固定；允许变化的是可执行 Harness 及其被接受的版本。

## 当前实现

- 带严格校验的受限 JSON 提案协议；
- `independent` 和 `dependency-aware-joint` 演化模式；
- 候选分支、held-in 接受、held-out 报告和回滚；
- Harness 版本、来源记录、成本核算、检查点和恢复；
- 固定 ReAct 与 ReAct+Reflexion 对照运行器；
- 确定性的组件交互矩阵；
- 批量运行器、审计、汇总和本地集成测试。

## 证据边界

最近一次联合修改冒烟运行在执行链路层面成功完成：

- `independent`，seed 17、2 批次：generator 提案因 `held_in_regression` 被拒绝；
- `dependency-aware-joint`，seed 17、2 批次：generator+selector 提案因 `held_in_regression` 被拒绝；
- 两次运行均以状态 `0` 完成，final-test 保持锁定。

这说明联合提案、候选评估和回滚链路可以运行，但不能据此声称该方法提高了模型平均质量。多随机种子正式比较和长程联合验证仍是独立实验。

`docs/current-status.md` 记录了较早完成的组件矩阵和 60 批次耐久运行。这些结果只支持各自对应的结论；耐久运行是执行稳定性证据，并非持续性能提升的证明。

## 从这里开始

项目主要入口文档如下：

- **README.md：** 安装和实验入口。
- **服务器手册：** [docs/server-handoff.md](docs/server-handoff.md) 介绍服务器准备、配置、模型适配器和执行顺序。
- **协作指南：**
  [docs/contributor-execution-runbook.md](docs/contributor-execution-runbook.md)
  说明贡献者流程、证据包和审查交接。
- **Issue 模板：**
  [.github/ISSUE_TEMPLATE/experiment-execution.yml](.github/ISSUE_TEMPLATE/experiment-execution.yml)
  用于向贡献者分配可复现实验。
- **报告模板：**
  [docs/run-report-template.md](docs/run-report-template.md)
  记录执行结果和证据链接。
- **研究协议：**
  [docs/technical-one-protocol.md](docs/technical-one-protocol.md)
  定义研究问题、实验组、对照、指标和结论边界。

## 本地验证

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

本地测试覆盖协议校验、提案约束、报告、回滚和模拟命令集成，不需要 GPU 或在线 Decision LLM 服务。

## 服务器准备

服务器使用现有 Python 环境运行 Task Model，并通过兼容 OpenAI API 的 vLLM 服务调用 Decision LLM。私有配置、数据集、模型路径、日志和结果不得提交到 Git。

```bash
export LEGO_ROOT="$(pwd)"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"

cp "$LEGO_ROOT/code/technologies/technology_one/server_profile.evolution.qwen.example.json" \
   "$LEGO_ROOT/server_profile.evolution.qwen.json"

export LEGO_POLICY_API_URL="http://127.0.0.1:8000/v1/chat/completions"
bash "$LEGO_ROOT/scripts/check_lego_install.sh"
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
  bash "$LEGO_ROOT/scripts/check_decision_model.sh"
```

正式实验前先运行两批次冒烟测试：

```bash
export CUDA_VISIBLE_DEVICES=0
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=40

OUTPUT_ROOT="$LEGO_ROOT/runs/technology-one-joint-smoke" \
GPU_ID=0 BATCHES=2 SEEDS="17" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

正式实验协议和必需对照见 `docs/technical-one-protocol.md`。协议、统计分析计划和证据审计冻结前，不得解锁 `final_test`。

## 协作

修改协议或实验运行器前请先阅读 `CONTRIBUTING.md`。提案校验、接受规则、数据切分、预算上限或模型版本变更，都需要记录协议说明并重新运行冒烟测试。不要提交私有配置、凭据、数据集、模型检查点、`runs/`、`logs/`、临时文件或生成的归档包。

服务器协作请遵循
`docs/contributor-execution-runbook.md`，其中说明 GitHub Issue 交接、固定 commit/profile 要求、冒烟测试、正式运行、证据包和结果报告格式。仓库还提供 `.github/ISSUE_TEMPLATE/experiment-execution.yml`，用于向贡献者分配可复现实验。

## 许可证

本软件包未声明许可证。公开再分发前，应先确定并添加许可证。
