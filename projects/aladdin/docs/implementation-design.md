# Aladdin 实现设计

本文将当前实验方案转化为代码层面的实现计划。文中会区分已经存在的模块与待实现模块，避免把设计写成仿佛已经可运行。当前可运行部分包括数据集工具、`experiments/experiment_1a.py` 中的静态合成对照 runner，以及 Qwen LoRA 训练和 development 评价；实验 1B、清洗、反馈选择和 RSI controller 仍是后续开发目标。

本项目在公开 monorepo 中的路径为 `projects/aladdin/`。以下路径均相对于该目录。

## 1. 系统运行结构

将三个数据操作放进同一个 pipeline，由一个外层 RSI controller 管理：

```text
seed 样本
  -> SynthesisStrategy.generate()
  -> Cleaner.clean()
  -> FeedbackSelector.select() -> AnnotationStore.release_labels()
  -> PolicyUpdater.propose() -> PolicyReviewer.confirm()
  -> 下一轮
```

Controller 负责轮次顺序、预算、版本 ID 和审计日志。每个阶段不能访问属于后续阶段的字段。尤其是 selector 只能接收候选内容和可见特征，不能读取隐藏标签；只有查询被选中后，才由标签存储释放对应反馈。开发和策略更新期间，controller 不加载 final-test 数据或标签。

## 2. 目标代码目录

```text
experiments/
  synthesis/                 已有静态合成适配器和 runner
  cleaning/                  确定性清洗和 judge 清洗器
  feedback/                  查询对象、选择策略、标注输入输出
  rsi/                       轮次 controller、策略版本和更新日志
  evaluation/                Qwen SFT、下游评价与统计报告
scripts/
  prepare_*.py                数据集专用准备脚本
  run_component.py            冻结组件对照入口
  run_rsi.py                  多轮系统对照入口
  analyze_run.py              manifest 校验和结果报告
configs/
  datasets.json               固定数据源和使用限制
  pilot.example.json          与机器无关的实验参数示例
tests/
  test_synthesis_*.py
  test_cleaning_*.py
  test_feedback_*.py
  test_rsi_*.py
  test_evaluation_*.py
```

服务器路径、endpoint、密钥、GPU 编号和下载数据不能写入跟踪的配置文件。允许本机配置通过环境变量覆盖路径；提交的配置示例只能包含占位值。

## 3. 共用数据格式

数据记录使用 JSONL，运行配置和 manifest 使用 JSON。每个模块边界都要校验 schema，并拒绝未知的 schema 版本。

### Candidate 记录

```json
{
  "schema_version": 1,
  "record_id": "stable-id",
  "source_group_id": "prompt-or-user-group-id",
  "parent_record_id": "seed-or-parent-id",
  "input": "任务输入",
  "context": [],
  "output": "候选回答",
  "round": 0,
  "synthesis_strategy": "fixed-prompt",
  "policy_version": "synthesis-v0",
  "generation": {"model": "pinned-model-id", "requests": 1}
}
```

不得把来源评分、隐藏偏好标签、标注者身份或 final-test 结果放进此记录，应分别保存在标签文件或事件文件中。所有派生记录都要保留 source group 和 parent，以便检查数据划分及 lineage。

### 清洗决策

每个输入候选对应一条决策记录，包含 `record_id`、`cleaner_id`、`cleaner_version`、`keep`、`reason_codes`、`duplicate_of`、确定性指标、judge 分数（若有）和时间戳。清洗器同时输出保留记录 ID 列表和完整决策日志；被拒绝的记录仍须保留，以供误删审计。

### 反馈查询与事件

查询记录包含 query ID、单条候选或同一输入下的一对候选、source group、轮次、选择策略/版本、预计评审秒数、评分分量和随机化后的显示顺序。反馈事件包含 query ID、假名化标注者 ID、可用性判断、原因标签、要求清晰度、可选修改建议、适用时的成对偏好、开始/结束时间戳、实际评审秒数和是否请求澄清。不同输入的回答不能伪造为成对比较。`cannot_judge` 和平局都是有效反馈。

### 策略更新事件

记录所用反馈事件 ID、前后策略版本、拟议变化、依据、人工确认状态、生效轮次和新版本生成的记录 ID。LLM 可以起草变更，但指定审核者确认前不能启用。该日志用来验证 RSI 更新是否真的发生且可追溯。

## 4. 实施顺序

### 步骤 A：冻结一份试验配置

每次正式运行通过 CLI 固定数据路径、PRISM split manifest、生成/训练预算、随机种子和指标定义。实验 1A 的生成模型与下游任务模型统一固定为 `Qwen/Qwen3-8B`，revision 为 `b968826d9c46dd6066d109eabc6255188de91218`；服务器 API 必须部署同一 revision，生成 API 和训练 chat template 均关闭 thinking 模式。运行 manifest 保存模型 ID、revision 和实际解析到的训练模型 commit。后续实验如需更换模型，须在新实验协议中预先冻结，不得混入同一对照矩阵。

每次运行都将不可变配置副本写进 run 目录，并记录数据源文件哈希、代码版本、Python/依赖版本、CUDA 信息（如使用）、随机种子以及开始/结束时间。

### 步骤 B：完成实验 1A（静态数据合成）

沿用 `experiments/synthesis/` 适配器。正式 runner `experiments/experiment_1a.py` 使用确定性 seed group 子集和共同服务端 token 上限。正式运行要求 endpoint 对每次请求返回 usage；Self-Instruct/Evol-Instruct 的所有请求均计入预算。预算末尾以完整候选结束并记录有限超额；候选安全上限先触发时运行失败。

每种方法从相同 seed 出发生成候选；仅使用各臂相同的 schema 检查和文本规范化，不进行语义清洗或反馈更新。各臂使用相同训练记录数与 Qwen LoRA 配置，并按真实输入 token 数达到共同的下游训练预算。Development 评价仅读取 PRISM development 的独立标签 sidecar，输出 chosen/rejected 条件 log-probability 排序准确率和参考回答 NLL；禁止读取 final-test。实现见 `experiments/experiment_1a.py` 与 `experiments/evaluation/qwen_sft.py`。

### 步骤 C：实现清洗（实验二）

实现共用 `Cleaner` 接口，在完全相同的合成候选池上运行以下对照：

1. `NoOpCleaner`：解析成功的候选保留；各臂都统计解析失败和空记录。
2. `DeterministicCleaner`：先规范化并精确去重，再按字符 5-gram TF-IDF cosine 相似度阈值 `0.92` 过滤近重复。平分时采用确定性规则；每条删除记录都记录保留代表项和 `duplicate_of`。
3. `QualityJudgeCleaner`：先运行确定性清洗，再交由固定版本 judge。只有 rubric 平均分不低于 `4/5` 且每个维度不低于 `3/5` 才保留。

启用第三种方法前，按冻结的抽样规则抽取至少 50 条候选做人类校准，且要从被拒绝样本中随机抽样。报告误删率、保留率、保留样本中的低质量残留率和 judge 成本。校准门槛未通过时，整体系统使用方法 2，并报告质量过滤器尚未验证。当前活动代码中还没有清洗器实现。

### 步骤 D：实现反馈选择（实验三）

一个标注请求对应一条候选；仅当两个回答具有同一任务输入时才使用盲测成对比较。实现以下策略：

1. `RandomSelector`：使用固定 seed，在符合条件的 query group 中均匀抽样。
2. `PreferenceUncertaintySelector`：使用在 development 数据上校准的偏好模型，为同输入的比较打分；正式选择前冻结模型。
3. `DisagreementSelector`：可选诊断策略，根据固定 judge 或 requirement 假设之间的分歧打分；不得将分歧当作正确性标签。
4. `ImpactPerReviewCostSelector`：计算 `uncertainty * affected_pool_count / max(predicted_review_seconds, 1)`。在释放任何查询标签前，按冻结的 cluster 估算 affected-pool 数量。评审时间在试验前校准，不得使用当前隐藏答案更新预测值。

所有策略按相同的累计实测人工分钟预算停止，不只对齐查询数量。随机化成对显示顺序，向标注者隐藏实验组别和选择分数。释放标签前先写入 query 和 annotation event。历史标签回放只能检查选择机制；真实人工时间必须在新生成候选的独立 live study 中测量。当前活动代码尚无 selector 或标注界面。

### 步骤 E：实现策略更新和实验 1B

先做 synthesis-only RSI，以便识别更新机制。固定实验 1A 选出的合成 baseline，至少运行两轮，比较：

- `R0 Fixed`：随机选择反馈，并作为相同的下游监督提供；但反馈不能改变下一轮合成策略。
- `R1 Random-update`：随机选择反馈，并允许据此更新合成策略。
- `R2 Aladdin-update`：按单位评审成本影响力选择反馈，再通过相同 updater 更新合成策略。

三组的候选数、反馈分钟、清洗器、任务模型训练 token 和 development 评价预算必须一致。策略更新只允许使用人工确认可用的示范、明确修改建议和经确认的 requirement。保存策略差异及其来源事件。单独的 `unusable` 标签不能作为负训练目标。如果没有合理的更新，就记录 no-op，不要为了让 RSI 看起来发生而编造变更。当前尚无 updater 和轮次调度器。

### 步骤 F：集成实验四

只有实验一至三选出并冻结组件后，才运行 `feasibility-pilot-protocol.md` 中的四组：A 仅 seed；B 固定流程/随机反馈/不更新策略；C 随机反馈 RSI；D impact-per-review-cost RSI。所有组的 source group、候选数、judge 预算、任务模型版本、训练 token 和累计人工分钟必须相同。要声称合成与清洗分别受益，还须增加“仅更新合成”和“仅更新清洗”消融。

冻结配置、阈值、标注者 rubric 和报告代码之前，不加载 final-test 文件。最终只做一次测试，报告成对偏好胜/平/负比例、按任务/source group 和标注者聚类的不确定性、标注者一致性、成本、每轮策略版本和排除记录。

## 5. 各阶段必须通过的测试

- 合成：seed 子集稳定、schema 合法、lineage 正确、随机种子可复现、请求/token 总量正确、endpoint 故障可处理。
- 清洗：规范化去重正确、代表项稳定、阈值边界正确、保留/删除日志完整、误删审计分母正确。
- 反馈：prompt group 不拆分、随机选择可复现、查询前不能访问隐藏标签、只构造同输入成对比较、按实际耗时正确停止。
- RSI：能区分更新/不更新对照、策略版本不可变、反馈事件能追溯到下一轮样本、预算一致、final-test 路径从未被打开。
- 分析：在独立 split/user 层级做配对比较、保留平局与 `cannot_judge`、提供置信区间和成本。

当前已实现代码可运行以下本地检查：

```bash
python -m pytest -q
python -m compileall -q experiments scripts tests
```

当前自动化测试覆盖已实现的数据工具、合成适配器及实验 1A 的预算、抽样、泄漏防护和结果写出逻辑；它不替代目标服务器上的真实模型 API 与 GPU 运行验证。

## 6. 当前实现边界

**已实现：** 固定版本数据集下载器、PRISM/HelpSteer2 数据准备和审计、四种合成适配器、实验 1A 的预算匹配矩阵 runner、Qwen LoRA 训练、development 代理评价和分析脚本。

**后续实验仍须实现：** 清洗器和 judge 审计、反馈选择与真人标注事件采集、策略更新器/版本存储、多轮 controller、实验 1B 和系统对照。实验 1A 正式 runner 已完成本地确定性测试；还须在协作者服务器上验证真实模型 endpoint 和 GPU 训练。
