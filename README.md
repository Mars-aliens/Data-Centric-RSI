# Data-Centric RSI

Data-Centric RSI 是一个研究总项目，关注系统如何在多轮运行中持续改进数据构建与模型开发。项目目前包含两个相互独立的研究方向：MLE-Aladdin 与 MLE-LEGO。跨项目整合将作为后续明确的工程工作推进，不预设代码已经共享。

## 子项目

| 项目 | 研究重点 | 当前状态 |
|---|---|---|
| MLE-Aladdin | 在需求部分未知的情况下构建生成式机器学习数据，包括校准反馈和主动反馈选择 | 独立本地 Git 仓库；提交 `6571f7d`；102 项测试通过；独立 GitHub 发布待完成 |
| MLE-LEGO | 自我改进的数据 Harness：依赖关系引导的修改、经验复用与连续反馈 | 已发布于 `projects/mle-lego/`；技术一已实现；技术二、三仍在规划 |

两个子项目有各自的源码、协议、依赖和研究结论。Aladdin 源码不复制到本仓库，以其独立仓库作为唯一事实来源。

## 仓库结构

```text
Data-Centric-RSI/
├── projects/
│   └── mle-lego/                 # MLE-LEGO 项目快照
├── docs/
│   ├── architecture.md           # 项目边界与整合规则
│   └── collaboration.md          # 两人协作与服务器实验流程
├── .github/ISSUE_TEMPLATE/       # 总项目 GitHub 任务模板
├── CONTRIBUTING.md               # 跨项目贡献规范
└── README.md
```

本仓库不包含 Aladdin 项目目录，因为该项目由独立 Git 仓库维护。不要将其源码复制到此处形成第二份事实来源。在两个项目明确共同接口之前，不建立总项目级共享代码包。

## 从这里开始

- 修改项目边界或共享接口前，先阅读[项目架构](docs/architecture.md)。
- 分配实现和服务器运行工作前，先阅读[协作流程](docs/collaboration.md)。
- MLE-LEGO 的安装、实验和现有证据见[项目说明](projects/mle-lego/README.md)。
- 跨项目进度及待办事项见[项目状态](docs/project-status.md)。
- 可复现的 MLE-LEGO 服务器实验请遵循[贡献者执行手册](projects/mle-lego/docs/contributor-execution-runbook.md)。

## 运行 MLE-LEGO 测试

```bash
cd projects/mle-lego
python -m pytest -q
```

数据集、模型权重、私有配置、检查点和运行结果均不纳入 Git。准备方式见 MLE-LEGO 的数据说明和服务器交接文档。

## 项目状态

本仓库包含 MLE-LEGO 项目快照。Aladdin 在独立仓库开发，不复制到此处。总项目仓库已发布至 `Mars-aliens/Data-Centric-RSI`；当前实现范围和实验结论边界见[项目状态文档](docs/project-status.md)。
