# 技术一：HelpSteer2 历史反馈 Replay

## 目的和范围

本实验只检验 idea/MLE-Aladdin.md 中的第一项技术：
结合需求分歧、对下游选中数据集合的预期变化、审查成本和固定随机探索保留，
决定下一次请求哪一条反馈。

这是对已有 HelpSteer2 人工评分的历史 replay，不会收集新的人工标注、
调用 LLM、测试技术二或技术三，也不能证明完整递归 RSI 系统。

## 冻结协议

完整参数表和复现实验命令位于 docs/experiment-parameters.md 的第 1 节。
下面重复列出关键值，使服务器交接文档可以独立使用。

- 数据集：nvidia/HelpSteer2，revision 为
  990b2711a36180dd19d9c94b8627844866f8982a。
- 只使用 helpsteer2-train.jsonl 中的 seed、held_in 和 held_out split，
  不打开 validation/final-test 文件。使用 source_group_id 保持 group 完整。
- 每次运行前固定候选需求解释：helpfulness-only、correctness-only 和
  coherence-only。这些是由 HelpSteer2 现有人类属性评分构造的半合成偏好
  profile，不是新询问得到的用户需求。
- Seed 示例用于训练五个 TF-IDF 属性模型，并估计每个候选回答的属性值。
  Profile 后验只使用 seed groups 初始化。
- 一次 query 是一个完整的双回答 prompt group。只有当选择策略请求该
  group 后，才释放其历史标签。Query pool 与下游数据选择 pool 不重叠。
- 查询成本是明确但未经校准的代理：
  5 秒 + 可见输入词数 / 每分钟 220 词。各方法共享相同时间上限和最大请求数，
  同时记录实际成本和请求数。
- Impact 分数为：
  disagreement * expected_selection_change / estimated_review_seconds。
  Expected selection change 是对每个可能反馈答案计算反事实 top-k 下游集合
  与当前 top-k 集合的对称差，再按后验加权。
- impact-aware-random 将配置比例的反馈时间预算保留给 random 查询，其余
  使用 impact score。
- 下游指标是 held-out、具有明确偏好的 pair 上相对于模拟目标 profile 的
  group-level pairwise accuracy。Tie 不计入准确率，但单独计数。
- final_test 永远不能用于选择、训练、参数选择或评估。只有协议和方法冻结后
  才能解锁。

冻结的 factorial 为 4 个 split seeds（17、23、29、37）x 3 个 policy seeds
（17、18、19）x 3 个目标 profile x 5 个策略 = 180 个条件。限制为：
64 个 seed groups、256 个 query-pool groups、512 个 selection-pool groups、
512 个 held-out groups、1,800 秒反馈时间、最多 48 次查询、64 个选中 groups、
25% random reserve 和 tie margin 0.0。

## 策略

- seed-only：使用相同 seed groups 训练，不请求新反馈，也不增加选中的训练 group。
- random：在共同成本上限内随机请求反馈。
- disagreement-only：最大化每个估计审查秒数对应的需求分歧。
- impact-aware：最大化分歧乘以下游选择变化的预期大小，再除以估计审查秒数。
- impact-aware-random：提议方法，使用固定随机探索保留，再执行 impact-aware 请求。

在每个 split_seed、policy_seed、target_profile 组合内，所有方法使用相同的
seed、query、下游 selection 和 held-out groups。四种反馈请求方法共享相同的
时间上限、最大请求数和下游选择大小。主要配对比较是
impact-aware-random 对 random、disagreement-only 和 seed-only。
置信区间使用 split 层配对差异；在推断前，每个 split 内先平均 target
profiles 和 policy seeds。

## 本地验证

从项目根目录运行：

~~~powershell
.\.venv\Scripts\python.exe -m pytest local_experiments\tests\test_active_feedback_helpsteer2.py -q
~~~

运行小型真实数据调试矩阵：

~~~powershell
.\.venv\Scripts\python.exe -m local_experiments.aladdin.active_feedback_helpsteer2 `
  --data data\helpsteer2-train.jsonl `
  --output tmp\technology-one-helpsteer2-debug.json `
  --split-seeds 17 `
  --policy-seeds 17 18 `
  --seed-group-limit 32 `
  --query-group-limit 32 `
  --selection-group-limit 64 `
  --heldout-group-limit 64 `
  --feedback-time-budget-seconds 300 `
  --max-queries 8 `
  --selection-budget 16
~~~

调试通过后，运行冻结的本地矩阵：

~~~powershell
.\.venv\Scripts\python.exe -m local_experiments.aladdin.active_feedback_helpsteer2 `
  --data data\helpsteer2-train.jsonl `
  --output tmp\technology-one-helpsteer2-matrix.json `
  --split-seeds 17 23 29 37 `
  --policy-seeds 17 18 19 `
  --seed-group-limit 64 `
  --query-group-limit 256 `
  --selection-group-limit 512 `
  --heldout-group-limit 512 `
  --feedback-time-budget-seconds 1800 `
  --max-queries 48 `
  --selection-budget 64 `
  --random-reserve-fraction 0.25
~~~

报告会记录数据集 checksum、配置、每次请求诊断、后验更新、选中 group ID、
已释放记录数量、pool ID 和重叠审计、pairwise 指标、配对差异及 final-test
访问标志。

## 本地矩阵结果

冻结的本地矩阵已完成全部 180 次运行（4 个 split seed x 3 个 policy seed
x 3 个历史偏好 profile x 5 个策略）。Held-out pairwise accuracy 平均值为：

本地 train 文件的精确 SHA-256：
5bd88e03cb93527131d2a75b30940ea8a969a23096f8f529eb0e9bb114f4d6e9。
复现时，请将这个完全一致的规范化 helpsteer2-train.jsonl 作为单独数据
构件传输并验证 checksum。不要静默替换为较旧或格式错误的服务器副本。

| 策略 | 平均 pairwise accuracy |
|---|---:|
| seed-only | 49.44% |
| random | 52.53% |
| disagreement-only | 53.18% |
| impact-aware | 53.18% |
| impact-aware-random | 53.09% |

impact-aware-random 相对于 random 的 split-level 配对差异为 +0.56 pp
（95% t CI [-0.48, +1.60]，4 个 split seed），相对于 disagreement-only
为 -0.09 pp（95% t CI [-0.79, +0.61]），相对于 seed-only 为 +3.65 pp
（95% t CI [+1.03, +6.27]）。这些是本地探索性 replay 结果，不是独立的
人工验证。Impact-aware 在该矩阵中没有优于 disagreement-only。虽然部分
条件下查询轨迹不同，但 impact-aware 和 disagreement-only 在全部 36 个
配对条件中选择了相同的下游数据 group。Random-reserve 变体相对于
disagreement-only 在 36 个条件中的 13 个选择了不同的下游集合。

实现通过了泄漏和预算审计，但该机制尚未显示出相对 disagreement-only 的
增量价值。应保留这一负结果；不得根据这些 held-out 分数调节 profile 定义
或预算，再将同一矩阵作为确认性证据。

## 服务器运行

实验只需要 Python、NumPy 和 scikit-learn，不需要 GPU、vLLM、Transformers
或模型 API。服务器上在选定 Python 环境中安装 requirements.txt，设置
PYTHONPATH 为解压后的包根目录，并使用与本地运行相同的数据集 revision 和
checksum。规范化数据集必须放在源码归档之外，启动前验证 checksum。

先运行 debug 命令，只有通过后才运行冻结矩阵，并将输出写入新的运行路径。
不要将部分写入的输出当作完成运行的证据。由于本地完整矩阵已经完成，服务器
重跑用于环境复现或单独登记的协议，不是解释本地矩阵的前置条件。

## 解释限制

该 replay 可以检验在受控、预算匹配的历史反馈条件下，策略实现是否改变了
请求顺序。它不能证明估计时间等于真实人工投入，不能证明三个手工指定 profile
代表真实用户不断变化的需求，也不能证明 LLM/人工界面会得到相同反馈。
TF-IDF 是轻量级下游协议基线，不是最终生成模型结果。积极的本地结果只能
支持开展独立受控的人工或模型实验，不能宣称完整 Aladdin 系统已经验证。
