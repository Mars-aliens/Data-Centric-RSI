# 合成 Baseline 上游来源

下表列出所参考的上游仓库及固定版本。上游工作副本不随项目分发；Aladdin 实际运行的是 `experiments/synthesis/` 中针对当前任务改造的实现，不代表完整复现论文训练流程。

| 方法 | 官方来源 | 固定版本 | 许可说明 | Aladdin 中的改造 |
|---|---|---|---|---|
| Self-Instruct | [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct) | `0b26ccaa415992100fa32df62d41b994cf928e23` | 上游代码为 Apache-2.0；发布数据集各自有单独许可，本项目不打包或使用。 | 先生成 task-input，再生成 response；改造自论文的 instruction/instance 自举流程。 |
| Evol-Instruct / WizardLM | [nlpxucan/WizardLM](https://github.com/nlpxucan/WizardLM) | `cd4b4cc022b6426aee62d51f7ae99a4a6ec9aaef` | WizardLM 代码许可为 Apache-2.0；上游数据和模型另有条款，本项目不打包或使用。 | 演化 task-input 后生成 response；只有确认任务适配 instruction/task-input 演化后才运行。 |

## 获取固定版本的上游代码

Linux：

```bash
bash scripts/fetch_synthesis_baselines.sh
git -C baselines/synthesis/self-instruct-upstream rev-parse HEAD
git -C baselines/synthesis/wizardlm-upstream rev-parse HEAD
```

PowerShell：

```powershell
& .\scripts\fetch_synthesis_baselines.ps1
git -C baselines/synthesis/self-instruct-upstream rev-parse HEAD
git -C baselines/synthesis/wizardlm-upstream rev-parse HEAD
```

两条 `rev-parse` 命令应分别输出上表中的固定 commit。脚本使用 sparse checkout，只取论文说明、许可和相关算法目录；上游副本被 `.gitignore` 排除，不会被提交。若目标目录已存在但 commit 不匹配，脚本会停止，不覆盖本地内容。

## 在 Aladdin 中运行的 baseline

实验一数据合成使用项目内适配器，不直接执行上游仓库脚本：

| 实验组 | Aladdin 实现 | 实验命令中的策略名 |
|---|---|---|
| 不合成 | 保留固定 seed | `seed-only` |
| 固定生成 | 冻结提示词直接生成新输入和回答 | `fixed-prompt` |
| Self-Instruct 思路 | 先生成新任务输入，再生成回答 | `self-instruct` |
| Evol-Instruct 思路 | 按固定演化操作改写输入，再生成回答 | `evol-instruct`，需显式确认任务兼容 |

正式矩阵、数据、Qwen3-8B revision、共同生成 token 预算、LoRA 参数和结果汇总命令见[实验一数据合成运行步骤](../../docs/experiment-one-data-synthesis.md)。在该实验中，所有组都从相同 seed 出发；合成环节只做结构检查，不请求人工反馈、不做人工质量筛选。固定 development 标签仅用于生成和训练完成后的离线结果指标。

## 论文复现范围

当前可运行的是 Aladdin 的任务适配比较，不是 Self-Instruct 或 WizardLM 原论文的完整实验复现，也不能将结果称为原论文结果。原因是目标任务是 PRISM 对话回答，而原方法面向 instruction 数据生成/演化；上游脚本还依赖原论文的数据构造、API/模型、过滤和训练环境。Aladdin 固定 Qwen3-8B 和统一 token 预算，运行本项目中的适配策略，目的是在同一目标任务和资源预算下比较合成方法。

原论文的 Self-Instruct 流程包括 instruction 生成、任务类型识别、实例生成、过滤/整理和下游 instruction tuning；WizardLM 的 Evol-Instruct 流程依赖其演化操作、源指令数据、生成 API 和后续模型训练。上游 README 与脚本给出了各自的原始步骤，但本项目没有打包原论文数据、旧模型/API、训练 checkpoint 或原始评价环境；Self-Instruct 脚本固定旧版 GPT-3 API 调用，WizardLM 源码还含占位 API key 和旧版调用。因此本项目不提供“原论文结果一键复现”承诺，也不把 Qwen3-8B 适配实验叫作原论文复现。

若后续确实要复现原论文结果，需在独立环境中按上游 README 获取许可允许的数据、恢复论文使用的模型/API 和完整生成/过滤/训练/评价流程，并逐项记录与原文的偏差。仅运行上游生成脚本不等于复现论文的模型结果。当前协作者可直接执行的是 Aladdin 的任务适配对照，命令见上文链接。

当前测试使用确定性 mock generator，只验证适配器行为，不证明模型质量，也不复现论文结果。
