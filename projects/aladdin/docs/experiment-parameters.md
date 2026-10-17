# Aladdin 实验参数

本文档冻结当前 Aladdin 实验使用的参数。下面四个部分属于相互独立的协议，
不得将它们合并为一个结果。

## 1. HelpSteer2 技术一历史反馈 Replay

### 目的

检验在固定模拟反馈预算下，结合需求分歧、对下游选择的预期影响和随机探索
保留比例来选择反馈请求，是否会改变所选训练数据的质量。这是对已有
HelpSteer2 标注的历史 replay，不是新的人工反馈研究。

### 数据集和划分协议

| 参数 | 冻结值 |
|---|---|
| 数据集 | nvidia/HelpSteer2 |
| 数据集 revision | 990b2711a36180dd19d9c94b8627844866f8982a |
| 输入构件 | 规范化 helpsteer2-train.jsonl |
| 本地记录 SHA-256 | 5bd88e03cb93527131d2a75b30940ea8a969a23096f8f529eb0e9bb114f4d6e9 |
| 分组键 | source_group_id |
| 使用的 split | seed、held_in、held_out |
| Final test | 不加载；final_test_accessed=false |
| 目标 profile | helpfulness-only、correctness-only、coherence-only |
| Profile 权重 | 对应 HelpSteer2 属性使用 one-hot 权重 |
| Split seeds | 17、23、29、37 |
| Policy seeds | 17、18、19 |
| 条件总数 | 4 x 3 x 3 x 5 = 180 |

目标 profile 只用于 replay 的模拟目标、seed supervision、后验标签和
held-out 指标。它不是用户提供的需求，不能描述为系统发现的用户偏好。

### 数据和反馈预算

| 参数 | 冻结值 |
|---|---:|
| Seed group 上限 | 64 |
| Query-pool group 上限 | 256 |
| Selection-pool group 上限 | 512 |
| Held-out group 上限 | 512 |
| 反馈时间预算 | 1,800 秒 |
| 最大反馈查询数 | 48 |
| 选入训练的 group 上限 | 64 |
| Random reserve fraction | 0.25 |
| Tie margin | 0.0 |
| 审查成本代理 | 5 秒 + 可见词数 / 每分钟 220 词 |

审查成本公式只是未经校准的代理，不是实测人工标注时间。Query、selection
和 held-out groups 必须互不重叠。未查询的 selection-pool 标签必须保持隐藏。

### 对比策略

| 策略 | 定义 |
|---|---|
| seed-only | 只使用 seed groups；不查询反馈，也不增加选入的 group |
| random | 在共同时间/查询预算内随机请求反馈 |
| disagreement-only | 需求分歧除以估计审查成本 |
| impact-aware | 分歧 x 预期选择集合变化 / 审查成本 |
| impact-aware-random | 使用 25% random reserve 的 impact-aware 选择；主方法 |

### 指标和推断

- 主指标：具有明确偏好的 held-out pair 的 group-level pairwise accuracy。
- Tie 不计入准确率分母，单独报告。
- 主要比较：impact-aware-random 减去 random、disagreement-only 和 seed-only。
- 配对单位：split seed。每个 split 内先平均 target profiles 和 policy seeds，
  再进行配对比较。
- 置信区间：四个 split 差异上的双侧 95% 配对 t 区间。
- 必须完成的审计：查询标签隐藏、selection-pool 标签隐藏、query/selection/
  held-out 零重叠、选择和反馈预算相等，以及加载的 final-test 行数为零。

### 复现实验命令

从 projects/aladdin/ 或独立 Aladdin 根目录运行。数据和输出必须放在 Git 之外。

~~~bash
export PYTHONPATH="$PWD"
export TRAIN_PY=python
export DATA_PATH=/external/data/helpsteer2-train.jsonl
export OUTPUT=/external/runs/aladdin-helpsteer2-technique-one.json

"$TRAIN_PY" -u -m local_experiments.aladdin.active_feedback_helpsteer2   --data "$DATA_PATH"   --output "$OUTPUT"   --split-seeds 17 23 29 37   --policy-seeds 17 18 19   --target-profiles helpfulness-only correctness-only coherence-only   --seed-group-limit 64   --query-group-limit 256   --selection-group-limit 512   --heldout-group-limit 512   --feedback-time-budget-seconds 1800   --max-queries 48   --selection-budget 64   --random-reserve-fraction 0.25   --tie-margin 0.0
~~~

使用仓库指定的分析器分析完成的报告：

~~~bash
export ANALYSIS=/external/runs/aladdin-helpsteer2-technique-one-analysis.json
"$TRAIN_PY" -u -m local_experiments.aladdin.analyze_active_feedback_helpsteer2   "$OUTPUT"   --output "$ANALYSIS"
~~~

查看 held-out 分数后不得调参，再将同一次运行称为确认性实验。

## 2. Qwen LoRA 下游验证

本协议检验冻结的模拟反馈矩阵所选 groups 是否能改善下游偏好分类器。
它不增加新反馈，也不验证独立数据集。

| 参数 | 冻结值 |
|---|---|
| 基础/任务模型 | Qwen/Qwen2.5-7B-Instruct |
| 目标 profile | helpfulness-only |
| Split seeds | 17、23、29、37 |
| Policy seeds | 17、18、19 |
| 策略 | seed-only、random、disagreement-only、impact-aware-random |
| 条件数 | 4 x 3 x 4 = 48 |
| 训练步数 | 150 |
| 单设备 batch size | 2 |
| 梯度累积 | 1 |
| 最大序列长度 | 1,024 |
| 学习率 | 2e-4 |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0.05 |
| 评估 batch size | 16 |
| 服务器运行 GPU 显存上限 | 60 GiB |
| 评估 groups | 冻结矩阵中的 512 个 held-out groups |
| Final test | 不访问 |

所有策略使用相同的基础模型、训练预算、评估 groups 和优化器配置。输出必须
记录矩阵 SHA-256、数据集 SHA-256、模型 revision、adapter hash 和
final_test_accessed。

矩阵是由冻结反馈协议生成的输入构件，必须随实验 issue 提供。确认运行期间
不得使用改变后的 seed 或预算重新生成矩阵。

示例命令：

~~~bash
export PYTHONPATH="$PWD"
export TRAIN_PY=/path/to/qwen-environment/bin/python
export DATA_PATH=/external/data/helpsteer2-seed-development-clean.jsonl
export MATRIX_PATH=/external/runs/technology-one-multiround-matrix.json
export BASE_MODEL=/external/models/Qwen2.5-7B-Instruct
export OUTPUT=/external/runs/aladdin-helpsteer2-qwen-lora-matrix.json
export CUDA_VISIBLE_DEVICES=0

"$TRAIN_PY" -u -m local_experiments.aladdin.helpsteer2_qwen_lora_matrix   --data "$DATA_PATH"   --matrix "$MATRIX_PATH"   --output "$OUTPUT"   --base-model "$BASE_MODEL"   --split-seeds 17 23 29 37   --policy-seeds 17 18 19   --target-profile helpfulness-only   --max-steps 150   --batch-size 2   --eval-batch-size 16   --max-length 1024   --learning-rate 2e-4   --lora-r 8   --lora-alpha 16   --lora-dropout 0.05   --max-gpu-memory-gib 60
~~~

## 3. HH-RLHF Qwen 诊断

这是独立的 selector 标签下游诊断，不是 HelpSteer2 矩阵，必须单独报告。

| 参数 | 正式/默认值 |
|---|---|
| 数据集 | 外部提供的 Anthropic HH-RLHF train 构件 |
| Split 单位 | prompt |
| Split seed | 17 |
| Strategy seeds | 17 到 26 |
| 策略 | 默认 seed-only、random、aladdin-v2；Qwen selector 变体属于明确标记的诊断 |
| 候选反馈预算 | 32 groups |
| Seed/pool/evaluation 上限 | 64 / 256 / 512 groups |
| 训练预算 | 100 steps，equal-epochs 模式 |
| Batch / 最大长度 | 2 / 1,024 |
| 学习率 | 2e-4 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| 评估 | held-out pairwise accuracy，默认 batch size 8 |
| 反馈 endpoint | OpenAI-compatible Qwen endpoint |
| 官方 test | 封存 |

诊断可以比较 LLM 模拟标签和数据集 oracle 标签，但 oracle 标签只能用于
分析，不能作为可用反馈方法。结果只支持 selector 标签诊断，不支持人工
反馈结论。

## 4. 集成管线工程验证

集成管线用于验证更大的 Aladdin loop，是独立工程验证，不能替代冻结的
技术一矩阵。

| 参数 | 默认值 |
|---|---:|
| 直接 Python 入口的反馈模式 | interactive |
| 服务器 wrapper 未设置时的反馈模式 | offline-replay |
| 策略 | coverage、revision、contrastive |
| 每轮候选数 | 4 |
| 反馈预算 | 4 次请求 |
| 反馈时间预算 | 120 秒 |
| 生成 token 预算 | 1,536 |
| 生成上下文中的 seed 示例 | 16 个示例，每个 700 字符 |
| 选中数据上限 | 8 |
| Random feedback fraction | 0.25 |
| Model seed | 17 |
| 轮数 | 1 |
| 质量 ensemble 大小 | 1 |
| Endpoint 超时 | 120 秒 |

offline replay 要求输入包含 seed、development 和 held_in 记录，并且只在
选中查询后使用已有标签。interactive 和 file 模式才能表示新的人工反馈。
配置的 round runner 不接受 final-test 记录。

管线中的本地 TF-IDF trainer 只是连接层基线，其 smoke 指标不得作为生成
模型改进证据。

## 必填报告字段

每份服务器报告必须包含：

- 仓库 commit 和协议版本；
- 数据集路径占位符、SHA-256 和不可变数据集 revision；
- 模型标识/revision 和 GPU 环境；
- split seeds、policy seeds、策略以及所有预算参数；
- 已完成条件数和仓库外输出路径；
- 主指标、配对差异，以及适用时的置信区间；
- 泄漏审计和明确的 final_test_accessed 值；
- 反馈属于历史 replay、模型模拟、文件提供还是实际人工收集。
