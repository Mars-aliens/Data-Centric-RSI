# 当前状态

更新日期：2026-10-22。

## 当前路线状态

研究重点是通过技术一、二、三共同保障 Data System 长周期稳定运行。当前先验证技术一候选隔离流程，再依据 pilot 结果确定正式运行安排。

## 研究方向

MLE-LEGO 的统一目标是保障 Data System 的长周期稳定执行与持续改进。优化主体是 Data System 的整体状态和运行方法，Harness 是主要改进载体，并覆盖数据构建与质量、模型更新、评价、反馈、Memory/Skill、工具和运行控制。技术一、二、三都必须分别提出并验证长周期稳定性机制；稳定性不能只作为最后一个技术点的附加实验。

当前已具备本地长周期任务流、固定流水线、受限候选策略与 `local_shadow` 独立分支执行接口；真实服务器数据、Task Model、Decision LLM/GPU 运行以及完整 T1/T2/T3 集成仍待验证。`local_shadow` 不要求 Kubernetes，且不运行任意候选代码或提供容器级隔离。先完成 4 周期 smoke，再开展 10～20 周期 pilot；根据运行稳定性和资源成本决定后续规模。正式协议冻结前，`final_test` 保持锁定。

## 证据位置

本源码包有意排除运行结果。请在执行服务器的带版本 `runs/` 目录下保留结果，并在实验报告中记录目录、profile 哈希、代码版本、模型版本、随机种子和预算。

## 新版运行框架骨架

本地已实现 T1 受限数据策略提案、候选 bundle manifest/哈希校验、Decision LLM 兼容接口、固定可信 runner、candidate 私有 checkpoint/output、stable parent 哈希核验和故障注入。`local_shadow` 的隔离证据是候选分支输出与 stable 状态分开；它不是任意代码沙箱，CPU/内存限制不作强制保证。Kubernetes backend 保留为可选增强。长周期任务流支持通过 `max_batches` 设置 pilot 周期数。候选通过安全检查后进入观察，不会自动提升为稳定版本；候选运行失败时稳定 runner 继续。跨技术版本晋升由 T3 的多批次策略负责。

T2/T3 当前代码属于可测试的机制骨架；配对均值策略是占位策略。T1 runner 尚未在真实服务器上验证模型 endpoint、GPU 和真实数据运行；三个技术点的组合及正式长周期实验均未完成。设计说明见 `docs/evolution-runtime-architecture.md`，T1 具体步骤见 `docs/technology-one-safe-evolution-runbook.md`。

## 新长周期任务流

Task Definition v1 已冻结 HelpSteer2 的 prompt-group 标签规则、10/60/15/15
train-group 划分、每批 50 个 prompt groups、流式顺序和反馈释放边界。`long_horizon_task.py`
提供无标签 development 输入、可信反馈释放、版本绑定的预序列评测、终态离线汇总及状态恢复；
`long_horizon_qwen_command.py` 与 `long_horizon_runner.py` 加入严格训练记录校验、A/B/C 标签映射、
parent checkpoint 续训、先预测后更新和逐批原子状态保存。run 工件还包括命令尝试账本、每批可信预测及 checksum、
已释放反馈上的合并预序列指标和任务/模型/命令配置哈希。

本地 fake-command 测试覆盖反馈时序、断点恢复、增量训练数据边界、流前缀限制、提案校验、候选版本绑定和候选分支隔离契约；另有确定性、默认策略的 T1 fixture 用于验证可信 runner 部署链路，它不属于 RSI 实验候选。尚未在真实 HelpSteer2 文件及真实 GPU 上 smoke。`local_shadow` 故障注入目前由本地测试覆盖，服务器端到端故障运行仍待执行。T1/T2/T3 runtime 的端到端组合也未完成。

实验按[技术一实验协议 v1](../experiments/technology_one/protocol-v1.md)推进：先完成 4 周期 `local_shadow` smoke，再运行 10～20 周期 pilot。根据稳定性、候选分支隔离、恢复能力和资源成本决定正式实验规模。周期性候选提案调度、多候选恢复和服务器真实数据/模型/GPU 预检仍待完成；pilot 结果不作为正式方法结论。
