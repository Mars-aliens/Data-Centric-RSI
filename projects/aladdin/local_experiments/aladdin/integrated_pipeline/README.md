# Aladdin 集成管线原型

这条新路径与归档的数据选择研究明确分开。它组合可替换的小组件，用于
smoke test 预期的循环：

~~~text
seed cases
  -> 验证并精确去重
  -> 生成候选数据
  -> 质量门控（accept / defer / reject）
  -> 在需求假设存在分歧且可能改变下游选择集合时请求反馈，
     按审查工作量计算，并保留随机查询
  -> 根据反馈更新假设权重，并保留直接人工编辑
  -> 可选：比较两个候选项，将偏好转换为记录级训练决策
  -> 使用 TF-IDF maximal marginal relevance 选择已接受的有标签数据
  -> 使用相同 trainer 比较 seed-only 和增强训练
  -> 在 development 上评估，并使用衰减 UCB1 分配后续合成
  -> 持久化需求、评估器、校准和分配状态
~~~

## 方法选择和限制

- 清洗使用确定性验证和精确去重。TF-IDF 余弦相似度高的样本对只标记为
  待审查，不会自动删除。
- 查询分数是项目特定策略，由需求分歧、下游 top-k 集合的反事实对称差以及
  估计人工工作量构成。这是可检验假设，不是已经证明胜出的方案。
- 质量决策需要分数和置信度阈值。本原型还通过独立质量审计通道和 Wilson
  一致性下界，分别为每个任务切片认证自动 accept 和 reject。训练 accept
  反馈不会再次作为质量证据。低于配置数量或一致性阈值的证据会进入人工
  审查（defer）。demo.py 中的 fixture 证据只覆盖这条控制路径；真实阈值
  仍需要保留的人类判断。
- 候选生成以及需求/质量评分都是注入式接口。demo.py 使用确定性 fixture，
  不解析自然语言需求，也不生成真实样本。
- 合成库包含 coverage、revision 和 contrastive 三种独立目标。LLM 适配器
  会将选中的目标和所需证据放入生成 prompt，不会仅凭策略名称就声称使用了
  不同方法。
- 成对反馈与二元 accept 分开表示。成对选择器和扩展辅助函数会保留偏好记录
  和审查成本；默认 smoke 路径仍使用单记录反馈，避免两种协议被无意混合。
- 可以组合多个质量评分器形成保守 ensemble。分数分歧会降低置信度，独立
  审计分歧会作为有版本的评估器 prompt 说明保留到后续轮次。
- TF-IDF MMR 使用已有的 maximal marginal relevance 方法
  （Carbonell 和 Goldstein，1998，<https://doi.org/10.1145/290941.291025>）。
- 合成策略分配使用有界、按成本归一化的 UCB1 奖励（development 准确率增益
  位于 [-1, 1]，生成成本至少为一个 token/call/example），遵循 UCB1
  （Auer 等，2002，<https://www.jmlr.org/papers/v3/auer02a.html>）对每单位
  实测生成成本的 development 增益进行分配。实际人工审查秒数会与声明的
  查询时间预算核对。smoke run 不能验证 regret 或收益。
- 下游效用在 development 上测量，不使用 held-out/final test。本地
  TF-IDF 分类器只是连接层基线，不是目标模型。

## 本地运行

在已激活项目环境的工作区根目录执行：

~~~powershell
python -m local_experiments.aladdin.integrated_pipeline.demo   --output-dir tmpaladdin-integrated-smoke
python -m pytest local_experiments	ests	est_integrated_aladdin_pipeline.py -q
~~~

报告会将自己标记为 smoke test。不要把其指标报告为方法改善真实下游任务
的证据。下一阶段会使用下面的配置适配器；在自动质量动作得到认证前，仍需要
真实且独立的人类判断。

## 配置好的 vLLM/OpenAI-compatible 运行

adapters.py 提供真实 JSONL 加载器、针对 OpenAI-compatible endpoint 的确定性
JSON 解析、带 seed-only lineage 的候选生成、需求和质量评分器、校准证据加载
以及明确的文件或控制台人工反馈。模型评分的质量不作为人工反馈，训练反馈也
不作为质量审计证据。生成记录会保留 seed parent 和 source group。

先执行 preflight。在默认 offline-replay 协议中，输入必须包含 seed、
development 和 held_in 行；final test 对此命令有意不可用。held_in 标签
只有在选中的反馈 query 释放后才可见。

最小输入和配置示例位于 examples/。空反馈文件和审计文件只是 schema 示例，
不提供人工证据，也不能作为校准材料。

~~~powershell
python -m local_experiments.aladdin.integrated_pipeline.preflight   --data path	rain-development.jsonl   --requirements pathequirements.json   --calibration pathcalibration.json   --endpoint http://127.0.0.1:8000/v1/chat/completions
~~~

然后使用新的空输出目录运行一轮 development。必须明确提供人工审查，
可以使用交互方式，也可以使用独立准备的反馈文件。

~~~powershell
python -m local_experiments.aladdin.integrated_pipeline.run_configured   --data path	rain-development.jsonl   --requirements pathequirements.json   --calibration pathcalibration.json   --interactive-feedback   --output-dir runsintegrated-round-001   --run-id integrated-round-001   --endpoint http://127.0.0.1:8000/v1/chat/completions   --model Qwen/Qwen2.5-7B-Instruct
~~~

使用 --quality-ensemble-size N 可运行 N 个独立 prompt 的质量评分器。
使用 --rounds N 和 --interactive-feedback 可运行多轮；运行器会在每轮
完成后写入 recursive-state.json 和 recursive-manifest.json。
recursive.py 还提供经过验证的状态加载器用于恢复。声称递归改进前，仍必须
单独冻结 final-test 协议。
