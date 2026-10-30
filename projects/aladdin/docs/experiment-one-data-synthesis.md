# 静态合成组件比较

本文说明现有的静态合成组件 runner。它不是当前主实验的必跑部分，也不是反馈驱动 RSI 的正式运行手册。主实验直接把选定的现成合成方法作为固定组件，在固定流水线和递归流水线中使用；无需完成本文件所述的 24 单元比较才能开展主实验。整体对照、更新消融和反馈选择对照见 [`feedback-driven-update-experiment.md`](feedback-driven-update-experiment.md)。

## 当前 runner 能做什么

`experiments/experiment_one_synthesis.py` 可在固定 seed 上比较 `seed-only`、`fixed-prompt`、`self-instruct`，并训练 Qwen LoRA 模型，在 PRISM development 数据上做离线评价。`seed-only` 是该旧 runner 支持的可选参考组，不是当前主实验要求。runner 不实现数据清洗、真人反馈采集、策略更新或多轮控制，因此其结果不能作为完整 RSI 实验结论。

模型固定为 `Qwen/Qwen3-8B`，revision 为 `b968826d9c46dd6066d109eabc6255188de91218`。生成和训练均使用该 revision。实际运行前仍需先完成 API 预检，并确认每次响应的 token usage 可被完整记录。

## 运行限制

- 不使用 final-test 数据或标签。
- PRISM development 标签仅用于离线开发集评价，不能当作当前轮真人反馈。
- 生成候选必须满足预先配置的独立样本数；不足时该运行失败，不能重复样本冒充独立候选。
- 生成和训练 token 都不得超过各自预算；运行结果必须记录实际消耗和超额值。
- 每个正式矩阵使用全新输出目录，并保留配置、数据哈希、代码哈希和完整日志。

在上述预算边界和候选数量检查完成前，不要使用旧结果补跑或合并成确认性矩阵。即使组件比较成功，也仍需实现并运行 [`反馈驱动更新主实验`](feedback-driven-update-experiment.md)，才能回答递归更新是否有效。
