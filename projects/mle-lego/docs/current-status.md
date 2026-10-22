# 当前状态

更新日期：2026-10-22。

## 当前路线状态（2026-10-22）

研究路线已转向保障 Data System 长周期稳定执行的 T1/T2/T3 整体框架。此前的
Harness 组件演化实验及其结果不再是当前实验的前置步骤，也不作为新框架的有效性证据。
旧实现、运行脚本和配置不包含在当前 GitHub 项目中。

## 研究方向

MLE-LEGO 的统一目标是保障 Data System 的长周期稳定执行与持续改进。优化主体是 Data System 的整体状态和运行方法，Harness 是主要改进载体，并覆盖数据构建与质量、模型更新、评价、反馈、Memory/Skill、工具和运行控制。技术一、二、三都必须分别提出并验证长周期稳定性机制；稳定性不能只作为最后一个技术点的附加实验。

既有服务器实验和正式证据仍来自旧 Harness 组件修改、候选接受/回滚和运行检查点。新路线已具备本地长周期任务流、固定 baseline runner，以及技术一的候选提案到隔离运行路径；真实 HelpSteer2/Docker/GPU 端到端证据、完整 Data System 级执行和连续反馈治理实验尚未完成。

## 此前工作简述

当前协议要求的技术一本地实现已完成，包含：

- strict `dependency-aware-joint` proposal validation;
- `independent` versus dependency-closed joint editing;
- candidate evaluation, held-in acceptance, rejection, rollback, checkpoint,
  and resume;
- baseline runners, component matrix, batch execution, audits, and tests.

## 此前实验记录

- 在冻结的 HelpSteer2/Qwen 协议下完成十随机种子组件矩阵。该矩阵中 selector+filter 组合表现最好，held-out 平均差值为 `+0.0404`，配对胜出 `8/10`。
- 完成一次 60 批次耐久运行，进度 `60/60`，无命令重试，未使用 final-test。这仅是执行稳定性证据。
- 最近一次联合修改冒烟测试完成 seed 17 的两个模式、共两批次。联合模式生成了合法的 `generator + selector` 提案；两个候选均因 held-in 指标退步而被拒绝。

## 尚未声称的结论

- 冒烟运行不能证明方法优越性。
- 这些证据不作为新 T1/T2/T3 路线的实验要求或有效性证据。
- 新路线的正式参数与统计分析冻结前，final-test 继续保持锁定。

## 证据位置

本源码包有意排除运行结果。请在执行服务器的带版本 `runs/` 目录下保留结果，并在实验报告中记录目录、profile 哈希、代码版本、模型版本、随机种子和预算。

## 新版运行框架骨架

2026-10-22 在旧实验路径之外新增并本地测试了 T1 候选提案、候选 bundle manifest/哈希校验、Decision LLM 兼容接口、Docker shadow runner、真实父版本绑定、候选输出核验和故障注入。新增长周期任务流的 `max_batches` 流前缀参数，可固定限制 smoke 规模，并将完整可用批数、排程批数和限制写进 manifest；默认不设该值，仍运行完整 development 流。候选通过安全检查仍只进入观察，不会提升为稳定版本；候选运行失败时稳定 runner 继续。跨技术版本晋升仍由 T3 的多批次策略负责。

这次新增内容不改变旧 `HarnessEvolutionController`、既有协议或历史实验结论。T2/T3 当前代码属于可测试的机制骨架；配对均值策略是占位基线。T1 新 runner 尚未在真实服务器上验证百炼/Qwen/Kimi endpoint、Docker 镜像、双卡隔离和真实数据运行；三个技术点的组合及正式长周期实验均未完成。设计说明见 `docs/evolution-runtime-architecture.md`，T1 具体步骤见 `docs/technology-one-safe-evolution-runbook.md`。

## 新长周期任务流

Task Definition v1 已冻结 HelpSteer2 的 prompt-group 标签规则、10/60/15/15
train-group 划分、每批 50 个 prompt groups、流式顺序和反馈释放边界。`long_horizon_task.py`
提供无标签 development 输入、可信反馈释放、版本绑定的预序列评测、终态离线汇总及状态恢复；
`long_horizon_qwen_command.py` 与 `long_horizon_runner.py` 加入严格训练记录校验、A/B/C 标签映射、
parent checkpoint 续训、先预测后更新和逐批原子状态保存。该路径独立于旧 `helpsteer2.py`
与旧技术一实验，未改写或迁移旧结果。run 工件还包括命令尝试账本、每批可信预测及 checksum、
已释放反馈上的合并预序列指标和任务/模型/命令配置哈希。

本地 fake-command 测试覆盖反馈时序、断点恢复、增量训练数据边界、流前缀限制、提案校验、候选版本绑定和安全隔离契约；另有确定性、无行为变化的 T1 候选 fixture 用于验证部署链路，它不属于 RSI 实验候选。尚未在真实 HelpSteer2 文件及真实 GPU 上 smoke。Docker 故障注入目前由本地 mock/契约测试覆盖，仍需在服务器镜像上实际执行各故障类型。T1/T2/T3 runtime 的端到端组合也未完成。

工程 pilot 仍按[技术一实验协议 v1](../experiments/technology_one/protocol-v1.md)执行。正式长周期规模已在[正式协议 v1](../experiments/technology_one/formal-protocol-v1.md)中预注册为 100 cycles、10 个配对 seeds、8 类故障及 5 个恢复 runs；机器可读矩阵为 `experiments/technology_one/formal-run-matrix-v1.json`。这只是本地设计冻结，不表示服务器实验已经开始或完成。

正式递归实验仍被周期性候选提案调度、多候选 lineage/resume 集成及 100-cycle 故障隔离集成测试阻塞。当前 runner 一次只执行固定的一份 candidate bundle。服务器真实数据/模型/Docker/GPU/预算预检也未完成。以上门槛通过前，只运行 4-batch pilot 或单候选工程验证，不得将其报告为正式 T1 RSI 结果。
