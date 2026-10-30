# Aladdin 实现说明

本文说明当前本地代码实际执行的 PRISM 主场景流程。它描述的是可运行实现，不代表已完成 GPU/API 试跑、真人反馈或确认性实验。LCCC 中文续写场景目前只实现数据准备工具，尚未接入此 RSI runner。

当前示例配置使用 `feedback_mode: model`：第三阶段由固定 revision 的 Qwen3-32B 生成模拟反馈，每组每轮固定评审条数，保存请求耗时、token 和失败尝试。以下交互式真人采集器作为可选模式保留；模型反馈不能计为真人反馈。final-test 支持 Qwen3-32B 自动盲评，交换顺序复评；同一反馈模型的最终评价只能作为模型评价证据。

## 1. 运行流程

```text
固定 seed
  -> Qwen3-8B 按改造版 Self-Instruct 生成任务输入与回答
  -> 确定性清洗，输出保留池和隔离池
  -> 查询选择器从两池中选出待评审样本
  -> 第三阶段采集真人判断、问题类型、修改/新要求和实际耗时
     （model 模式由 Qwen3-32B 模拟生成，不计为真人）
  -> 更新下一轮合成、清洗或选择策略
  -> 使用 seed 与累计可用反馈重新训练 Qwen LoRA
  -> 仅在 development 数据上评价
```

阶段一、二不读取本轮人工反馈，也不要求人逐条判断。阶段三选中样本后才请求真人反馈；反馈只影响下一轮。主 runner 只读取 `seed`、`development` 和 development 标签，不打开 `final_test`。

## 2. 代码入口

| 路径 | 实际职责 |
|---|---|
| `experiments/synthesis/` | Self-Instruct 数据合成适配及 OpenAI-compatible API 调用 |
| `experiments/cleaning/rules.py` | schema/空文本/重复输入/重复片段/精确与近重复清洗；输出隔离样本和原因 |
| `experiments/feedback/selection.py` | development 用户组切分、选择器校准、随机/不确定性/反馈价值查询选择 |
| `experiments/feedback/annotation.py` | 交互式真人标注、实际耗时记录、未完成队列续采 |
| `experiments/rsi/policies.py` | 合成、清洗、反馈选择策略状态和版本更新 |
| `experiments/rsi/runner.py` | 实验组调度、数据边界、预算、检查点和结果写出 |
| `experiments/evaluation/qwen_sft.py` | Qwen3-8B LoRA SFT 和 development 成对偏好代理评价 |
| `experiments/evaluation/blind_pairwise.py` | final-test 输出生成、盲化、真人成对评价与汇总 |
| `configs/feedback-rsi-pilot.example.json` | 仅供工程试跑的参数示例 |

运行入口：

```bash
python -m experiments.rsi --config configs/feedback-rsi-pilot.example.json
python -m scripts.analyze_feedback_rsi runs/feedback-rsi-pilot/result.json
```

## 3. 主方法实现

### 数据合成

改造版 Self-Instruct 从 seed 选择父样本，先生成新任务输入，再生成对应回答。每条候选保留父样本、source group、轮次、合成策略版本、模型 ID 和 revision。接口必须逐请求返回 prompt/completion token 数，超过单组生成 token 预算时停止并报错。

### 数据清洗

`clean_candidates` 依次检查空/过短文本、回答重复输入、过度重复、精确重复和字符 n-gram TF-IDF 近重复。清洗器输出 `retained.jsonl`、`quarantine.jsonl` 和逐条决策。两类候选都能进入第三阶段查询池；pilot 配置默认将 25% 查询名额留给隔离池抽查，若某一池样本不足，剩余名额由另一池补足。

### 数据选择与反馈

每个查询是一条候选及其任务输入。随机策略在候选池内抽样；不确定性策略使用 development calibration 用户组拟合的字符 n-gram TF-IDF + Logistic Regression，按二元偏好熵排序；反馈价值策略使用

```text
偏好熵 × (1 + impact_weight × log(1 + 同源候选数)) ÷ 预计评审秒数
```

该实现是待验证的反馈价值代理分数，不是已校准的真实效用。查询前不可读取 development 隐藏标签以外的任何本轮反馈；实际标注记录 `usable / needs_revision / unusable / cannot_judge`、问题类型、要求清晰度、可选修改/新要求、匿名标注者和真实耗时。

人工时间预算是目标上限。选择下一条查询前会用已记录耗时和预估耗时检查剩余预算；单次实际评审仍可能超过预算，结果会单独报告超出秒数。每种比较中所有组的首轮共享同一生成候选池；后续轮次允许不同反馈产生不同策略和候选。

### RSI 策略更新

- 合成策略：去重后的 `new_requirement` 作为下一轮生成约束，最多保留最近 8 条。
- 清洗策略：一轮至少收到两条 `repetitive` 反馈时，下调下一轮的重复阈值。
- 反馈选择策略：按 `actionable` 反馈比例调整下一轮 `impact_weight`，限制在 `[0, 2]`。

策略变化和来源 query ID 写入 `policy-update.json`。固定策略组仍接收并使用相同流程的反馈训练监督，但不改变策略；消融组分别冻结一类更新。反馈不足时允许 `no_op`。

## 4. 比较入口和结果

`comparison` 可设为：

| 值 | 运行组 |
|---|---|
| `overall` | `fixed_policy`、`recursive_policy` |
| `ablation` | 完整递归组及冻结合成/清洗/反馈选择更新的三个消融组 |
| `selection` | 随机、不确定性、反馈价值三种查询策略 |
| `all` | 上述所有组 |

每个 seed/组/轮次在独立目录保存候选、清洗决策、查询队列、反馈、训练数据、策略更新、LoRA adapter（启用时）和 development 指标。`result.json` 保存输入文件 hash、split manifest hash、代码 hash、模型 revision、预算、状态和汇总。development 结果只作开发诊断，不作为最终效果证据。

中断后可使用 `--resume`。恢复要求 output 下已有未完成的 `result.json`，且配置、输入 hash、split manifest 和 runner 代码 hash 完全匹配。已完成查询不会重复提问，已完成轮次不会重复训练；同一 output 目录不得同时运行多个进程。配置、数据或代码有变化时必须使用新 output 目录重新运行。

final-test 盲评通过单独的 `scripts.final_blind_evaluation prepare/collect/analyze` 命令执行。只有在实验配置和分析方案冻结后才可读取 final-test；准备评审队列时，评分标签不会被加载。需要先保存待比较的两个 LoRA adapter。

## 5. 实现边界与验证

自动测试覆盖数据准备、清洗、反馈选择、策略更新、多轮运行恢复、查询池隔离、Qwen 训练预算边界和 final-test 盲评逻辑。运行：

```bash
python -m pytest -q
python -m compileall -q experiments scripts tests
```

当前代码可以运行工程 pilot，但仍需在目标 Linux GPU/API 环境完成模型加载、真实生成、LoRA 训练、终端人工标注和实际耗时检查。pilot 结果不能作为确认性实验结论；正式运行前仍需冻结样本量、seed 数、人工时间预算、开发/测试评价方案，并完成数据许可和隐私审查。
