# 合成 Baseline 上游来源

下表列出所参考的上游仓库及固定版本。上游工作副本不随项目分发；Aladdin 实际运行的是 `experiments/synthesis/` 中针对当前任务改造的实现，不代表完整复现论文训练流程。

| 方法 | 官方来源 | 固定版本 | 许可说明 | Aladdin 中的改造 |
|---|---|---|---|---|
| Self-Instruct | [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct) | `0b26ccaa415992100fa32df62d41b994cf928e23` | 上游代码为 Apache-2.0；发布数据集各自有单独许可，本项目不打包或使用。 | 先生成 task-input，再生成 response；改造自论文的 instruction/instance 自举流程。 |
| Evol-Instruct / WizardLM | [nlpxucan/WizardLM](https://github.com/nlpxucan/WizardLM) | `cd4b4cc022b6426aee62d51f7ae99a4a6ec9aaef` | WizardLM 代码许可为 Apache-2.0；上游数据和模型另有条款，本项目不打包或使用。 | 演化 task-input 后生成 response；只有确认任务适配 instruction/task-input 演化后才运行。 |

需要检查固定版本的上游代码时，运行 `scripts/fetch_synthesis_baselines.ps1`。不要直接运行上游生成脚本：Self-Instruct 使用旧版 OpenAI SDK；WizardLM 的 Evol-Instruct 脚本包含占位 API key、硬编码输入/输出路径和旧版 API 调用。Aladdin 适配器使用项目配置的 OpenAI-compatible endpoint，并在 run manifest 中记录请求和 token 用量。

当前测试使用确定性 mock generator，只验证适配器行为，不证明模型质量，也不复现论文结果。
