# LEGO 对照方法研究与复现计划

## 证据标准

arXiv 论文只能证明方法以预印本形式发表，不能证明存在官方基准要求、官方实现或代码复用许可。每项结果必须标注为以下类别之一：

- `official reproduction`：使用作者实现和原始协议进行的正式复现；
- `author-code adaptation`：修改作者代码以适配 LEGO 任务；
- `independent reimplementation`：根据论文独立实现；
- `project control/ablation`：为 LEGO 设计的对照或消融，不声称复现外部方法。

不得将适配实验报告成原论文结果。

## 主要比较方法

| 方法 | 原始来源 | 作者实现证据 | LEGO 中的定位与适配 |
|---|---|---|---|
| 固定 Harness + ReAct 风格控制器 | Yao 等，*ReAct: Synergizing Reasoning and Acting in Language Models*，[arXiv:2210.03629](https://arxiv.org/abs/2210.03629)，ICLR 2023。论文元数据于 2026-10-02 对照 arXiv API 核验。 | [ysymyth/ReAct](https://github.com/ysymyth/ReAct)，MIT；`master`，commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`，仓库活跃。原代码针对原论文 QA/交互环境，不是本 HelpSteer2 数据流。 | 项目对照。使用相同 Decision LLM 和操作/工具接口，但持续冻结 Harness、程序和 Skill 版本。不可称为 ReAct 基准的精确复现。当前本地运行仅为确定性 TF-IDF 脚手架。 |
| ReAct + Reflexion | Shinn 等，*Reflexion: Language Agents with Verbal Reinforcement Learning*，[arXiv:2303.11366](https://arxiv.org/abs/2303.11366)，NeurIPS 2023。元数据于 2026-10-02 核验。 | [noahshinn/reflexion](https://github.com/noahshinn/reflexion)，MIT；`main`，commit `218cf0ef1df84b05ce379dd4a8e47f17766733a0`。其原始基准/任务实现不是 LEGO 数据流。 | 可采用作者代码适配或忠实独立实现：冻结可执行 Harness；根据已释放反馈保存文字反思；只在后续暴露符合条件的历史反思。不持续改写代码、prompt 或 Skill。反思 Token 计入预算。当前本地实现是玩具/规则版本，不是论文实现。 |
| General Self-Harness | Zhang 等，*Self-Harness: Harnesses That Improve Themselves*，[arXiv:2606.09498](https://arxiv.org/abs/2606.09498)，v3 日期 2026-08-20。论文元数据和仓库链接于 2026-10-02 对照 arXiv API 核验。 | [qzzqzzb/Self-Harness](https://github.com/qzzqzzb/Self-Harness)，`main`，commit `2720dbb3f52283684f4b85a1065d642df1779dd8`，未归档。GitHub 元数据未声明许可证；复用前须检查许可证和具体版本。未声明许可证不代表可以复制或修改。 | 适配其弱点挖掘、与失败关联的最小提案和回归验证，但移除 LEGO 依赖引导。保持可编辑组件、轨迹、候选预算和验证数据相同。若不能合法或技术性地使用其代码，应依据论文独立实现并准确标注。 |
| CurateEvo | Wang 等，*CurateEvo: Data-Curation Evolving for Agentic Post-Training*，[arXiv:2607.06140v1](https://arxiv.org/abs/2607.06140)，2026-07-07 提交。标题、作者、版本和摘要页于 2026-10-02 核验。 | arXiv 页面未提供作者 GitHub 链接；官方实现和许可证**未核实**。不要用名称相似的第三方仓库替代。 | 当前 LEGO 方案要求将其作为适配方法。实现前核实原始整理循环、变异/评估协议、许可、依赖和数据假设。该方法根据 held-out 失败演化可执行整理代码，并包含成本感知剪枝；适配到相同数据流时应保留这些关键思想。代码/协议审查完成前，不得声称已完成该对照。 |

## 可选相关工作

这些方法与研究相关，但不是 `idea/MLE-LEGO.md` 当前对照表中的明确必做项；它们属于扩展候选，不是初始核心矩阵的门槛。

| 方法 | 来源与代码 | 建议定位 |
|---|---|---|
| DataEvolver | Deng 等，*DataEvolver: Automatic Data Preparation for Large Language Models through Multi-Level Self-Evolving*，[arXiv:2606.07001](https://arxiv.org/abs/2606.07001)，v2（2026-06-10）。[ruc-datalab/DataEvolver](https://github.com/ruc-datalab/DataEvolver)，MIT，核验 commit `2f02be1267ba97164ac6c5326d1e583af5532104`。 | 可选。原协议在 Alpaca、ARC-Easy/Challenge、GSM8K/MATH 和 Spider/BIRD 上进行 seed 引导的原始数据到 SFT 数据准备。应作为数据准备对照，而非原生长程 LEGO 复现。 |
| HarnessEvolve | Jiang 等，*HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution*，[arXiv:2609.00829](https://arxiv.org/abs/2609.00829)，v1（2026-09-01）。元数据和 HTML 实验设置于 2026-10-02 核验。 | 检查的 arXiv 记录未提供代码链接，官方仓库/许可未核实。论文使用 SearchQA、OfficeQA、SpreadsheetBench、CloudCoreNetwork-QA 和 Wireless-QA，后两者为内部数据集。可选但高度相关；其参考轨迹对齐和质量/性能门槛不同于 LEGO 依赖感知的数据组件分组。公开适配可考虑 OfficeQA 和 SpreadsheetBench；不得声称复现内部数据集或等同于 LEGO 方法。 |

## 项目内对照与消融

以下不是外部论文基线，须标注为项目对照或消融：

- `independent-edit`：每次只修改一个组件，用于检验联合修改的作用。
- `random-joint-edit`：联合候选规模相同，但不使用依赖关系引导。
- `one-batch-selection` 与 `three-batch-selection`：比较后续多批次验证的价值。
- `same-memory-no-process-update`：比较持续过程更新与仅访问相同初始经验。
- `dependency-aware-joint-edit`：本项目提出的方法，不是外部基线。

## 统一服务器协议

每项比较运行前，冻结并计算以下信息的哈希：

1. Decision LLM 名称/版本、解码设置、system prompt、上下文长度和全部 API/模型版本。
2. Task Model 初始检查点及所有实验组相同的训练方案。
3. 数据集来源版本、归一化数据哈希、按组切分、数据流批次顺序、反馈释放计划和标签构造。
4. 工具/API 接口及权限。对照方法可以策略不同，但数据和工具访问权限必须一致。
5. 总预算，包括决策/演化 LLM 调用和 Token、候选执行、失败候选、Task Model 训练、评估以及人工反馈请求/耗时。
6. 候选接受逻辑、held-in 选择规则、held-out 回归策略及 final-test 锁定。最终报告前不得向系统暴露 `final_test`。
7. 多个固定随机种子/运行和配对统计分析计划。

各候选分支应从相同的初始模型、数据和 Memory 状态分叉。不得共享分支中新生成的检查点、反思或经验。模拟反馈结果必须与新采集的人类偏好结果分开报告。

## 方法适配器验收标准

以下是外部方法适配器进入服务器比较前的协议测试，并不意味着原论文在 HelpSteer2 上评估过。

### ReAct + Reflexion

- 在固定 Harness 接口下，按顺序记录 ReAct 行动选择和工具结果。
- 仅在符合条件的反馈/结果事件后生成反思，并将文字保存在情景记忆中。
- 后续决策可以读取符合条件的历史反思，但 Harness 代码、组件图和持久 Skill 不变。
- 反思 Token 和调用计入与 LEGO 相同的决策预算。

### General Self-Harness

- 提案前应对执行轨迹或失败记录进行弱点聚类；单独的原始分数不能作为失败诊断。
- 每项提案都应尽量小、对应已诊断失败，并保留父 Harness 版本。
- 接受前按 held-in/held-out 策略执行回归验证，不向方法提供依赖关系分组或 LEGO 专属图启发式。
- 失败提案不得改变当前 Harness、任务检查点或 Memory。

### CurateEvo

- 演化对象必须是可执行的数据整理代码，而不只是自然语言反思或模型检查点。
- 反馈来自 held-out 开发失败，并在成本感知目标下驱动增强、过滤、细化或剪枝。
- 重新整理数据及后续训练使用与其他实验组相同的固定方案；只改变整理策略。
- 记录适配器中哪些部分来自原始 CurateEvo，哪些是 LEGO 数据流协议所要求的。

### DataEvolver（可选）

- 将 seed 引导的数据理解与操作级、流水线级演化区分开。
- 操作候选和流水线候选均需执行检查与 held-out 验证；不得将两个层级折叠成任意单一修改。
- 除非任务目标和原始数据到处理后数据的接口确实等价，否则只作为可选数据准备比较报告。

### HarnessEvolve（可选）

- 参考轨迹须与普通失败轨迹区分，并且只能按声明协议提供。
- 评估、优化和门控应保持为可分离模块。
- 性能门槛前先检查质量、数据泄漏和 prompt 膨胀门槛；记录近期批次回归和 epoch 结束时的 held-out 选择。
- 不得将此适配器标为等同于 LEGO 依赖感知修改；两种方法使用的信号和门槛不同。

## 服务器运行前的完成门槛

- [x] 记录候选论文标识和已知作者仓库链接。
- [x] 区分已发表方法的适配与 LEGO 项目控制/消融。
- [x] 审查 Self-Harness、CurateEvo、DataEvolver 和 HarnessEvolve 的公开 HTML 方法结构，并记录版本、算法和结论。完整 PDF/附录审查仍须纳入各适配器实现审查。
- [ ] 核验所有待运行外部方法的官方仓库链接、commit、许可证文件和复现说明。
- [ ] 逐项决定使用作者代码适配还是独立重实现。
- [ ] 编码前明确方法到 LEGO 的映射及验收测试。
- [ ] 冻结真实决策/任务模型、prompt、预算、数据流协议、重复次数、主要指标和统计分析。
- [ ] 在安排高成本运行前，于本地或服务器冒烟任务中实现并测试真实模型适配器。

本说明中的仓库和论文元数据核验日期为 2026-10-02。服务器执行前，应重新检查可能变化的 GitHub 分支和许可证。
