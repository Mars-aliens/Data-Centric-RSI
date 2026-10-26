# MLE-LEGO

MLE-LEGO 是面向 Data-Centric Recursive Self-Improvement（RSI）的模块化研究项目，研究 Data System 如何在持续运行中改进自身，并保持长期稳定执行。优化主体是 Data System 的整体运行方法，其中 Harness 是主要改进载体，同时覆盖数据构建、数据选择与清洗、模型更新、评价、反馈、Memory 和 Skill 等环节。

项目的三个技术点都服务于同一个目标：让 Data System 在长周期运行中能够持续改进、控制错误累积、正确使用延迟反馈，并在更新失败时恢复。既有正式实验仍集中于技术一；目前已新增 T1/T2/T3 集成接口骨架和本地测试，但技术二、技术三的研究策略、真实任务适配及长周期收益尚未实现或验证。

本仓库仅包含 LEGO 方向，不含 Aladdin 实验、模型权重、私有数据集、GPU 环境、凭据或既有运行结果。

## 项目架构

```text
code/
├── core_system/mle_core/          共享接口、运行时、数据与预算
├── technologies/
│   ├── technology_one/             长周期任务流、固定流水线与候选安全门
│   ├── technology_two/             经验生命周期管理（接口骨架）
│   └── technology_three/           后续表现驱动的版本保留（接口骨架）
├── integration/                    跨技术运行时（接口骨架）
experiments/                        协议、清单和结果索引
docs/                               研究、服务器与贡献者指南
scripts/                            启动、审计和报告入口
data/                               仅含说明；数据集放在仓库之外
```

当前代码统一从 `technologies.technology_one` 导入。

## 研究范围与当前路线

研究对象是 Data System 的整体运行方法：数据构建与质量、Task Model 更新、评价、反馈、Memory/Skill、工具和 Harness 控制流程。RSI 周期采用混合方式：离线固定可复现初始系统，之后按任务批次运行、接收已到期反馈并更新。

新研究按 T1 → T2 → T3 构建：T1 隔离候选修改并让失败不污染稳定运行；T2 对可执行经验/Skill 做来源追踪、适用性验证、冲突治理和失效处理；T3 将延迟反馈归给正确版本，根据多个后续批次决定保留、继续观察或回退。三者最终在同一运行框架中组合，且每一项都单独评估长周期稳定性。

当前共同任务定义是 HelpSteer2 prompt-group 质量分类的受控流式回放。数据集没有自然时间戳或用户会话，延迟反馈是对已有评分标签的实验控制，不是真人在线反馈。数据划分、信息边界和评价规则以
[长周期任务定义](docs/long-horizon-task-definition.md) 为准。

## 当前实现与边界

- 新任务流：完整 prompt-group 切分、确定性批次顺序、无标签当前批输入、0/2/5 批延迟反馈、版本绑定的预序列评分和状态恢复。
- 固定流水线 baseline：先用当前 checkpoint 预测本批；再释放到期标签，并用 seed + 已释放反馈续训 parent checkpoint。
- 运行可靠性：逐批原子状态、命令尝试日志、检查点来源链和配置/任务哈希。
- T1：本地已接通可配置 Decision LLM 候选提案、源码 bundle provenance、固定镜像 digest 的 Kubernetes Job sandbox、运行时父版本绑定、候选输出安全检查和故障注入接口；尚未在真实服务器/Kubernetes/GPU 上验证。
- T2/T3：经验生命周期、反馈归因和多批次版本选择仍是机制骨架，尚未完成真实 runner 端到端集成和长周期实验。

本地测试不需要 GPU，但真实任务模型 smoke、服务器真实数据 manifest、正式故障注入和多随机种子评估都尚未完成。当前代码状态与下一道门槛见
[docs/current-status.md](docs/current-status.md)。

新框架边界见 [docs/evolution-runtime-architecture.md](docs/evolution-runtime-architecture.md)。

## 从这里开始

项目主要入口文档如下：

- **README.md：** 安装和实验入口。
- **服务器手册：** [docs/server-quickstart.md](docs/server-quickstart.md) 是当前服务器执行入口。
- **技术一实验：** 按[实验协议 v1](experiments/technology_one/protocol-v1.md)完成 4 周期部署 smoke，再运行 10～20 周期 pilot；根据稳定性、隔离、恢复和资源成本决定正式实验规模。服务器配置与命令见[执行手册](docs/technology-one-safe-evolution-runbook.md)。
- **协作指南：** [docs/server-quickstart.md](docs/server-quickstart.md) 说明新服务器从克隆、安装、数据准备到 smoke 的顺序。
- **报告模板：**
  [docs/run-report-template.md](docs/run-report-template.md)
  记录执行结果和证据链接。
- **当前任务与研究范围：** [docs/long-horizon-task-definition.md](docs/long-horizon-task-definition.md) 和 [docs/research-scope.md](docs/research-scope.md)。

## 本地验证

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-server.txt
python -m pytest -q
```

CPU 测试环境需要 PyTorch CPU wheel，因为部分模型协议测试会导入 PyTorch/Transformers/PEFT，但不需要 NVIDIA GPU 或在线 Decision LLM 服务。服务器 GPU 安装方式见[服务器快速开始](docs/server-quickstart.md)。本地测试通过不等于已经验证 Data System 的长周期稳定性。

## 服务器准备

新服务器不能只执行 `pip install -r requirements.txt` 后就开始 GPU 实验。它还需要匹配服务器 CUDA/驱动的 PyTorch、Transformers/PEFT、外部 HelpSteer2 数据、可本地训练的 Task Model、Decision LLM API、Kubernetes 访问权限和服务器私有配置。完整的从克隆到 smoke 的步骤见[技术一安全自进化执行手册](docs/technology-one-safe-evolution-runbook.md)。
新路线首轮建议使用本地 `Qwen/Qwen3-8B` Task Model；T1 Decision LLM 先用百炼 Qwen3.8-Flash 做 pilot，依据提案质量和成本决定正式冻结 Flash 或升级评估 Qwen3.8-Max。正式 run 不能切换模型。不需要同时使用所有 Qwen 型号、Kimi 或额外 evaluator LLM。各服务器自行配置模型快照路径、revision、GPU、API 凭据和 endpoint。模型选择理由、非必需候选及后续执行顺序见[模型配置](docs/model-configuration.md)和[技术一运行手册](docs/technology-one-safe-evolution-runbook.md)。新服务器尚未完成真实 API/权重 smoke，不能把建议当作已验证结果。

`final_test` 在研究协议冻结前继续锁定。

## 协作

修改协议或实验运行器前请先阅读 `CONTRIBUTING.md`。提案校验、接受规则、数据切分、预算上限或模型版本变更，都需要记录协议说明并重新运行冒烟测试。不要提交私有配置、凭据、数据集、模型检查点、`runs/`、`logs/`、临时文件或生成的归档包。

服务器协作从 `docs/server-quickstart.md` 开始；运行报告格式见 `docs/run-report-template.md`。

## 许可证

本软件包未声明许可证。公开再分发前，应先确定并添加许可证。
