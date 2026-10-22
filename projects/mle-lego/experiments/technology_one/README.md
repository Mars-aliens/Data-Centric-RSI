# 技术一实验

当前新技术一是“安全自进化与故障隔离”。先按[工程 pilot 协议](protocol-v1.md)完成 4 批部署验证；之后按[正式长周期协议 v1](formal-protocol-v1.md)运行 100 批正式矩阵。正式矩阵为 10 个 seed 的 fixed/shadow 配对、8 类独立故障和 5 个恢复 run，共 65 runs。机器可读计划见[formal-run-matrix-v1.json](formal-run-matrix-v1.json)。

注意：当前 T1 runner 每个 run 只运行一份固定候选，周期性生成 10 份候选的功能仍是正式递归实验的代码门槛。该门槛通过前不得把 100 批单候选运行称作完整 RSI/T1 正式验证。

本目录保存冻结后的实验协议版本、审计摘要和外部证据包的链接及校验值。HelpSteer2 数据、模型权重、profile、日志、运行结果和凭据不得提交到仓库。

此前开展过 Harness 组件修改及 `independent`/`dependency-aware-joint` 比较；相关代码、脚本和运行结果不包含在当前 GitHub 项目中，也不作为新 T1 的运行前置步骤或稳定性证据。
