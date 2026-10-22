# 技术一正式长周期实验协议 v1

版本：`t1-formal-long-horizon-v1`
状态：本地研究设计已冻结；服务器数据、模型和资源预检完成前不得启动正式运行。
配套工程 pilot：[工程协议](protocol-v1.md)；机器可读矩阵：[formal-run-matrix-v1.json](formal-run-matrix-v1.json)。

## 1. 研究问题与结论范围

本实验检验候选修改隔离机制能否让 Data System 在长周期数据流中持续运行，并在候选失败时保护 stable 分支、状态和 checkpoint。任务质量是共同报告指标，不把单次分数提高作为 T1 安全机制有效的充分证据。

HelpSteer2 development stream 是受控回放，不是自然时间流或真人反馈。结论限于固定 revision、既定分组/标签规则和已测试的故障模型；不得外推为任意代码安全或真实在线用户系统稳定性。

## 2. 周期定义与运行长度

- 一个周期（cycle）定义为处理一个 development batch；默认每批 50 个完整 prompt groups。
- 正式目标为每个 run **100 个周期**，即至少 5,000 个不重复 development prompt groups。不得复制样本延长运行。
- 服务器 manifest 若不足 100 批，先停止并审查数据 revision、切分与可用 groups；不得静默降成短实验或重复数据。任何改用较短流的协议必须另行编号。
- pilot 为 4 批，只验证工程链路；不是正式结果。100 个数据周期不等于 100 次 RSI 更新。
- 当前 runner 在整个 run 中固定一份候选 bundle；因此当前实现只能验证单一候选在长流中的隔离，不能据此声称已验证递归多次自进化。

## 3. 正式条件与运行规模

正式随机种子固定为 `17, 23, 29, 31, 37, 41, 43, 47, 53, 59`。主比较按 seed 配对，使用相同数据 manifest、流顺序、初始 Task Model、训练 recipe、反馈延迟、代码 commit 和预算。

| 条件 | 次数 | 长度 | 目的 |
|---|---:|---:|---|
| `B0-fixed` | 10 seeds | 100 cycles | 固定系统长期运行基线 |
| `T1-shadow-clean` | 10 seeds | 100 cycles | stable 与隔离候选并行；无故障 |
| `T1-shadow-fault` | 8 类故障 × 5 seeds = 40 runs | 100 cycles | 故障在 cycle 50 注入；候选终止后 stable 必须继续至 100 |
| `T1-resume` | 5 seeds | 100 cycles | cycle 50 边界中断并恢复；检查 exactly-once 状态 |

计划总量为 65 runs，所有 run 的 stable 流均目标完成 100/100。每个故障 run 中 candidate 在注入后终止是预期结果，stable 仍需继续完成。故障种子使用前五个正式 seed，每种故障均有独立 run、candidate ID 和输出目录。恢复 run 必须从原 run ID、配置和输出目录续跑。

故障类型固定为 `timeout`、`nonzero_exit`、`invalid_json`、`invalid_protocol`、`wrong_parent`、`missing_prediction`、`invalid_label`、`budget_overrun`。`write_protected_parent` 作为额外容器权限/契约测试单列，不和应用层故障率合并。

## 4. 关键实现门槛

本协议不定义不安全的 `T1-naive`：候选不能获得 stable 状态写权限，也不能在真实共享目录中关闭隔离。若需消融隔离/门控，须实现仅作用于 disposable copy、无网络、无秘密、无 stable 挂载的受控仿真臂，另发协议版本。

正式主张是递归自进化时，正式运行前必须实现并通过本地契约测试：

1. 在 100 个周期中按预注册时点产生多份新的候选；建议每 10 个已完成周期生成一次，共 10 个提案机会。
2. 每个候选均基于当时 active stable parent，生成不可变 bundle、candidate ID 和审计 manifest；失败候选不得改变后续 stable parent。
3. 候选运行、反馈、资源预算、重试和恢复具备逐候选状态与 exactly-once 语义；候选通过安全检查不等于晋升，晋升/回退由 T3 负责。
4. fault injection runner 能在候选终止后继续 stable stream，且不丢失已提交批次。

现有 `run_long_horizon_t1_qwen.py` 每次只接收一份 candidate config，尚未实现周期性提案调度。因此，在该功能及其集成测试完成前，只能执行 pilot/单候选隔离测试；不得启动或宣称完成本协议的 recursive-T1 正式效能实验。固定 baseline 和单候选 100-cycle 压力测试可作为工程稳定性证据，但不等同于 RSI 方法有效性实验。

## 5. 公平性和数据边界

- `feedback_delay=2` 为主条件；延迟 `0` 与 `5` 是单独的敏感性实验。
- 主实验固定 `groups_per_batch=50`、`stream_seed=seed`、同一完整数据 manifest。
- Task Model、tokenizer、LoRA recipe、Decision LLM API model ID/revision、提示词哈希、Docker 镜像 digest、GPU 分配和每 run 总预算必须在服务器 smoke 后登记。变更任一项须另开协议版本和输出根。
- `held_out` 只由可信评测器报告；不得供候选、提案、经验检索或选择逻辑使用。`final_test` 在所有条件和分析冻结前完全锁定；所有 summary 的 `final_test_used` 必须为 `false`。
- 失败尝试、候选生成、Docker、Task Model 训练、评估、恢复和额外存储均计入成本；分别报告宿主监控值与候选自报值。

## 6. 指标与预注册分析

主要稳定性终点：

- stable 100-cycle completion：完成 100 批且无丢批的 run 比例；
- fault containment：候选被检测/停止，stable 仍完成 100 批且 stable checkpoint/state 哈希连续的故障 run 比例；
- state integrity：批次、反馈释放、候选 lineage 和 checkpoint 无重复、缺失或来源错配；
- resume success：恢复后 run signature 不变，批次/反馈 exactly-once，最终完成 100 批。

次要结果包括逐批 prequential Macro-F1/accuracy、候选失败前后差异、提案有效性、恢复耗时、预算超限、调用/token、GPU/CPU/存储成本。使用同 seed 配对差值，报告逐 seed、均值、标准差、中位数和 95% bootstrap 区间；二元稳定性结果报告分子/分母及精确区间。

## 7. 启动与停止门槛

启动正式矩阵前必须全部通过：

1. 实际数据 manifest 至少包含 5,000 个唯一 development groups、100 批及正确反馈计划；
2. 4-batch `B0-fixed`、`T1-shadow-clean`、单故障和边界恢复 pilot 通过；
3. 固定模型 revision、Decision API、Docker digest、资源预算和代码 commit；
4. 周期性候选调度、多候选 lineage/resume、各故障类型 contract tests 及跨批失败后继续的集成测试通过；
5. 10 seeds × 100 cycles 的成本在服务器预算内，`final_test` 未访问。

正式 run 遇到状态完整性违规、反馈/测试泄漏、stable checkpoint 污染、签名漂移或不可解释丢批即停止。修复后使用新 commit、新 run ID 和新输出根重跑受影响的完整配对块，禁止覆盖旧证据。

## 8. 结论边界

65-run 矩阵全部完成且审计通过后，才能报告该任务与故障集合上的正式 T1 长周期结果。若周期性候选调度未实现，只能报告 fixed stream 或单候选 shadow 的工程稳定性；若没有合法、同任务适配的外部方法实现，不得声称优于 Self-Harness、ReAct/Reflexion 或 DataEvolver。
