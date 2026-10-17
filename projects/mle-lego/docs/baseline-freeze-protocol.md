# LEGO 技术一：Baseline 补全与冻结协议

状态：历史冻结协议文档。最新服务器进度、实际结果和未完成事项见
`实验记录/LEGO-技术一-当前完整进度汇总-2026-10-14.md`。本协议仍作为实验规则和公平性约束的历史依据。

## 一、冻结顺序

1. selector `40/80/120/160` 配对消融已作为开发诊断完成；不再追加 sweep seed。
2. 只依据 `held_in` 选择最终 selector 规则；`held_out` 仅作回归报告。
3. 冻结 generator、selector、filter 的规则、训练预算和模型配置。
4. 用独立入口运行 fixed baseline pilot，并保存训练组审计。
5. pilot 通过后，用相同协议跑 fixed baseline、baseline/control/adaptation 矩阵和最终自然支持方法的完整 seed 集合。
6. 所有候选选择和回归报告完成后，才解锁 `final_test`。

## 二、主方法的处理

当前理论主方法仍是 dependency-aware joint edit。经验上表现最好的 `selector-filter-joint-edit` 暂作为强消融，不在没有重新冻结方法定义前改写为主方法。

如果 selector sweep 和后续理论定义都支持将 selector-filter 定义为新的最终方法，必须重新记录方法版本，并重新运行统一 baseline 矩阵。

## 三、当前技术一对照矩阵

当前服务器阶段只冻结技术一的核心可辨识对照。外部方法适配不作为本批次部署门槛：

核心对照：

- `fixed-harness-react`：固定 Harness 的 ReAct 控制；
- `react-reflexion`：固定 Harness 加文字反思，不持久修改 Harness；
- `independent-edit`：单组件修改；
- `random-joint-edit`：不使用依赖关系的联合修改；
- `dependency-aware-joint-edit`：技术一主方法；
- `selector-filter-joint-edit`：强组件消融；
- `generator-selector-filter-joint-edit`：三组件消融；
- `one-batch-selection` 与 `three-batch-selection`：选择视野消融；
- `same-memory-no-process-update`：记忆收益与过程修改收益的区分。

`general-self-harness` 和 `curateevo-adaptation` 在当前服务器协议中以 HelpSteer2 task adaptation 形式接入，可用于本任务对照，但不得写成原论文 official reproduction。`dataevolver-adaptation` 和 `harnessevolve-adaptation` 仍属于可选后续扩展，不进入当前 required baseline 矩阵。

## 四、公平性约束

所有方法固定：

- 相同 HelpSteer2 revision、split 和 source-group 规则；
- 相同 Qwen Task Model、tokenizer、revision 和训练配置；
- 相同训练 group budget、候选预算和评估 batch；
- 相同 seed 集合；
- `held_in` 只用于候选选择；
- `held_out` 只用于回归报告；
- `final_test` 只用于最终一次性报告。

Natural-support 与 common-support 是两个不同 estimand，必须分开汇报，不能混合排名。

## 五、最终报告至少包含

- held-in 和 held-out 的均值、标准差、逐 seed 值；
- 相对 fixed 的 paired delta；
- 最佳方法胜出次数；
- selector-filter 与 dependency-aware 方法的直接比较；
- tie / non-tie 的分层指标；
- 每个方法的训练 group、label、margin 和文本截断诊断；
- 是否使用 final_test；
- 外部 baseline 是否为完整复现或 adaptation。

## 六、当前不能声称的内容

- 不能声称三组件联合必然最优；
- 不能声称 LEGO 已优于所有外部 baseline；
- 不能把 deterministic candidate matrix 写成 Decision-LLM 驱动的完整 LEGO evolution；
- 不能在 selector 或 baseline 规则仍根据 held_out 调整时使用 final_test。
