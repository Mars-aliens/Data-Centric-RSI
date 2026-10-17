# 当前状态

更新日期：2026-10-16。

## 实现情况

当前协议要求的技术一本地实现已完成，包含：

- strict `dependency-aware-joint` proposal validation;
- `independent` versus dependency-closed joint editing;
- candidate evaluation, held-in acceptance, rejection, rollback, checkpoint,
  and resume;
- baseline runners, component matrix, batch execution, audits, and tests.

## 已完成的证据

- 在冻结的 HelpSteer2/Qwen 协议下完成十随机种子组件矩阵。该矩阵中 selector+filter 组合表现最好，held-out 平均差值为 `+0.0404`，配对胜出 `8/10`。
- 完成一次 60 批次耐久运行，进度 `60/60`，无命令重试，未使用 final-test。这仅是执行稳定性证据。
- 最近一次联合修改冒烟测试完成 seed 17 的两个模式、共两批次。联合模式生成了合法的 `generator + selector` 提案；两个候选均因 held-in 指标退步而被拒绝。

## 尚未声称的结论

- 冒烟运行不能证明方法优越性。
- 最新严格联合模式仍需进行十随机种子正式比较，并审计每个种子至少一项联合提案。
- 100/300 批次长程联合验证是独立于冒烟测试和十随机种子比较的实验。
- 协议和统计分析冻结前，final-test 继续保持锁定。

## 证据位置

本源码包有意排除运行结果。请在执行服务器的带版本 `runs/` 目录下保留结果，并在实验报告中记录目录、profile 哈希、代码版本、模型版本、随机种子和预算。
