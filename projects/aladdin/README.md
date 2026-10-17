# MLE-Aladdin

Aladdin 是面向生成式机器学习任务的 Data-Centric RSI 研究项目。这类任务
在开始时通常只有部分明确的需求和质量判断。本目录只包含 Aladdin 实现；
MLE-LEGO 是独立项目，特意不包含在这里。

## 研究范围

Aladdin 研究数据管线如何将少量 seed 数据和有限反馈转化为更好的训练数据。
本项目包含：

- 需求假设和需求状态更新；
- 经过校准的人工/模型反馈协议；
- 固定查询预算下的主动反馈选择；
- 带泄漏检查的 HelpSteer2 和 HH-RLHF 适配器；
- 数据合成、选择和下游效用评估；
- 冻结基线和配对策略比较；
- 可使用 vLLM-compatible endpoint 的多轮集成管线。

当前实验用于验证单个机制，不能视为任何外部论文的正式复现，也不能将其
合并报告为一个结果。数据集、模型权重、checkpoint、服务器 profile、
凭据和运行输出都必须保留在 Git 之外。

## 仓库结构

~~~text
Aladdin/
  core_system/                 Aladdin 使用的共享动作系统基础组件
  local_experiments/aladdin/   Aladdin 适配器、协议、运行器和分析器
  local_experiments/tests/     单元测试和协议测试
  docs/                        实验和贡献者文档
  scripts/                     仓库检查和打包工具
  requirements.txt             CPU 测试和基线依赖
  requirements-qwen.txt        可选的 Qwen/LoRA 依赖
~~~

## 快速验证

在仓库根目录执行：

~~~bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python -m pytest local_experiments/tests core_system/tests -q
python -m compileall -q core_system local_experiments
python scripts/check_aladdin_install.py
~~~

测试使用合成 fixture，不需要 GPU、下载数据集或模型 endpoint。

## 集成管线 smoke test

集成管线可以使用准备好的 JSONL replay 离线运行。数据文件和 requirements
文件由运行者提供：

~~~bash
export PYTHONPATH="$PWD"
export TRAIN_PY=python
export DATA_PATH=/path/to/seed-development-held-in.jsonl
export REQUIREMENTS_PATH=/path/to/requirements.json
export FEEDBACK_MODE=offline-replay
bash local_experiments/aladdin/integrated_pipeline/server_preflight.sh
~~~

使用真实 endpoint 执行一轮：

~~~bash
export ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_DIR=/path/to/runs/aladdin-round-001
export RUN_ID=aladdin-round-001
bash local_experiments/aladdin/integrated_pipeline/server_run_round.sh
~~~

只有在确实有人审查展示的候选数据时，才使用 FEEDBACK_MODE=interactive。
只有在反馈文件由独立流程准备时，才使用 FEEDBACK_MODE=file。
offline-replay 中已有的数据集标签不属于新的人工反馈。

完整服务器交接说明位于
local_experiments/aladdin/integrated_pipeline/SERVER_HANDOFF.md。

HelpSteer2 replay、Qwen 下游验证、HH-RLHF 诊断和集成管线的冻结参数位于
docs/experiment-parameters.md。这些实验的指标不能合并成一个结果。

## 主动反馈和下游验证

HelpSteer2 和 HH-RLHF 运行器接收运行者提供的数据集路径，并将结果写入
仓库外的输出目录。协议会分离 seed、development、held-in、held-out 和
final-test 数据。final-test 标签必须一直封存到预注册的最终评估。

进行 Qwen LoRA 验证时，使用包含 requirements-qwen.txt 的环境，设置
ALADDIN_TASK_MODEL_BASE_PATH，并将运行器指向本地模型 snapshot。
Qwen 适配器位于 local_experiments/aladdin/qwen_preference.py，不依赖
独立的 LEGO 包命名空间。

## 可复现性规则

每次报告的运行都应记录仓库 commit、数据集 SHA-256、模型标识/revision、
split seeds、policy seeds、反馈预算、训练预算、endpoint 模型名称以及
是否访问 final-test。不得提交原始用户反馈、私有服务器 profile、模型
权重或大型运行构件。

## 协作流程

GitHub 仓库是协议和代码变更的唯一依据。执行贡献者应：

1. 检出 issue 中指定的精确 commit；
2. 执行列出的 preflight 和 smoke 命令；
3. 只使用仓库外的数据/模型路径运行指定实验；
4. 返回 commit、命令、日志、输出 manifest、哈希和摘要；
5. 提交不包含数据、checkpoint、秘密或生成运行目录的 pull request。

详见 CONTRIBUTING.md、docs/contributor-execution-runbook.md 和
docs/run-report-template.md。
