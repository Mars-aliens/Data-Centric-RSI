# 影响感知主动反馈：本地协议

本 active-feedback-impact-local-v2 原型用于在加入 Qwen 或人工标注基础设施
之前，实现并检查技术一的预期主张。这是协议测试和调试基准，不是
HH-RLHF 性能证据。

## 研究问题

在相同反馈成本预算下，按照查询对未来数据选择的预期影响来选择查询，
是否优于 random 和 disagreement-only，并在下游 held-out 任务上取得更好结果？

## 操作定义

本地基准维护多个候选需求假设。每个假设都会为每个候选项预测标签和效用
分数。实验包含三个互不重叠的 pool：查询 pool、下游选择 pool 和
held-out 评估 pool。对查询候选项 x：

~~~text
disagreement(x) = 各需求假设之间的标签分歧
impact(x) = 当前 top-k 集合与反事实 top-k 集合之间的预期对称差
value(x) = disagreement(x) * impact(x) / query_cost(x)
~~~

反事实集合在独立的下游选择 pool 上计算，使用每个可能的反馈答案以及
对应的需求假设后验。selector 永远看不到隐藏基准标签。查询后，已观察到的
反馈会更新后验，下一轮重新计算 value。被查询项目不会自动加入下游训练集。

## 策略

- random：在成本预算内随机查询；
- disagreement-only：按单位成本的高分歧程度查询；
- impact-aware：将分歧程度乘以下游选择变化的预期大小；
- impact-aware-random：保留 25% 成本预算用于随机查询，其余按 impact-aware value 分配。

所有策略共享相同的生成基准、seed、候选 pool、轮数、每轮成本预算和最大
选择数量。由于查询成本不同，策略实际选择的候选数量可能不同，但任何策略
都不得超过相同的成本预算。

报告同时包含总成本和成本利用率。不能因为某个策略使用的查询更少，就仅凭
更低的实际花费声称它更好；必须同时报告下游指标和实际花费。

## 本地运行

~~~powershell
python -m local_experiments.aladdin.active_feedback_impact   --output local_experimentsaladdinunsactive-feedback-impact-local.json   --seeds 17 18 19 20 21 22 23 24 25 26   --rounds 2   --cost-budget-per-round 12   --selection-budget 12   --random-reserve-fraction 0.25
~~~

报告记录每轮查询的 ID、查询成本、分歧程度、反事实受影响选择数量、
下游最终选择的 ID、后验更新以及 held-out 最近邻准确率。每次运行还会
保存 seed、query、下游 selection 和 held-out pool 的 ID 以及成对重叠数量，
因此可以从报告审计 pool 隔离。报告还记录 held-out 标签未用于选择、相对
random 的配对准确率差异和实际成本利用率。

## 当前本地诊断

v2 实现通过了重点测试（6 个测试）和独立本地测试集（62 个测试）。
10-seed 诊断使用一轮、每次运行成本上限 4、最多 4 次查询和 25% random
reserve。held-out KNN 准确率均值为：

| 策略 | 平均准确率 | 平均成本 | 平均查询数 | 平均成本利用率 |
|---|---:|---:|---:|---:|
| random | 66.42% | 3.58 | 3.00 | 89.4% |
| disagreement-only | 67.75% | 4.00 | 4.00 | 100.0% |
| impact-aware | 66.04% | 4.00 | 4.00 | 100.0% |
| impact-aware-random | 66.87% | 3.58 | 3.40 | 89.4% |

相对于相同 10 个合成 seed 的 random，平均准确率差异为：
disagreement-only 为 +1.33 pp，impact-aware 为 -0.38 pp，
impact-aware-random 为 +0.46 pp。这些只是机制诊断，不是确认性估计：
需求是玩具数据，反馈由 oracle 模拟，下游任务是合成的 KNN 任务。
尤其是，本次运行没有显示 impact-aware selection 的优势。由于查询成本
不可分割且剩余预算有时无法再购买一个候选项，random 使用的名义预算较少；
因此每次比较都必须报告实际成本。

impact 分数使用合成的下游 selector 估计预期 top-k 选择变化。它不是经过
验证的人力时间、人类效用或真实模型性能估计。下一步有意义的本地工作是：
在查询影响和标签分歧独立变化的受控合成案例中测试分数，并确认排序按定义
响应这两个因素。不要根据 held-out 准确率调参基准。

## 必须遵守的解释边界

本地运行可以证明选择分数已经实现，并且比较使用了匹配预算。它不能证明
人工可用性、LLM 判断质量或真实任务下游收益。后续服务器实验必须使用独立
定义的需求模型替换合成需求假设，并使用人工反馈或经过明确验证的 LLM 反馈
替换模拟反馈。
