# HH-RLHF Qwen 协议

本文档是当前 Aladdin 核心技术一实验的规范流程。旧的
formal-hh-rlhf-qwen-v1 运行作为历史基线保留；当前主策略是
aladdin-qwen-reliable，协议标记为 custom-diagnostic-v1。

~~~mermaid
flowchart TD
    A[Anthropic HH-RLHF train<br/>固定数据集 revision] --> B[按 prompt 划分<br/>预先声明 split seeds]
    B --> C{划分审计<br/>prompt 重叠 = 0}
    C -->|失败| X[终止运行]
    C -->|通过| D[共享 seed groups: 64<br/>held-in pool: 256<br/>held-out groups: 512]
    D --> E1[seed-only<br/>使用 64 个 seed groups]
    D --> E2[random<br/>选择 32 个反馈 groups]
    D --> E3[aladdin-qwen-reliable<br/>目标 rank=0.35 + 25% 新颖性]
    E2 --> F[固定 Qwen A/B 反馈<br/>顺序由 seed + group_id 决定]
    E3 --> F
    E1 --> G[Seed 标签]
    F --> H[LLM 标签记录]
    E2 --> I[Oracle 标签诊断]
    E3 --> I
    G --> J[Qwen LoRA 偏好分类器]
    H --> J
    I --> K[Qwen LoRA Oracle 诊断]
    J --> L[Held-out 成对准确率]
    K --> M[LLM 标签与 Oracle 标签差距]
    L --> N[跨 seeds 的配对分析]
    M --> N
    N --> O[报告趋势或稳定证据<br/>CI 跨过零表示不能声称有优势]
~~~

## 冻结控制项

| 组件 | 正式设置 |
|---|---|
| 策略 | seed-only、random、aladdin-qwen-reliable |
| Seeds | 每个 split 使用 17..26 |
| Split | prompt，预先声明 split seed，重叠即终止 |
| 反馈预算 | 每种反馈策略 32 groups |
| 训练暴露量 | equal-epochs，seed-only 参考为 100 steps |
| 评估 batch | split 17/23/29 为 8；split 37 为 16 |
| 反馈模型 | Qwen/Qwen2.5-7B-Instruct，固定 revision |
| 任务模型 | 相同 Qwen snapshot，使用 LoRA 偏好分类器 |
| 选择器 | 75% 以目标为中心的不确定性价值 + 25% prompt 新颖性；目标 rank 0.35 |
| 评估 | 512 个 held-out groups，成对准确率 |
| 测试集 | 官方 test 保持封存 |

## Split 安排

已完成的安排包含 prompt splits 17、23、29 和 37。每个 split 内使用
配对 seeds 报告；split 层平均值只作描述性统计，不作为独立观测的合并检验。

Split 17 属于探索性运行，因为 reliable target 是在较早诊断之后固定的。
Splits 23、29 和 37 是按冻结规则进行的确认运行。Split 37 的
Oracle-label 结果低于 random，应作为异质性证据保留，不能为了使用
LLM-label 结果而省略。

## 解释边界

这是单轮历史 replay。Qwen 使用已有 HH-RLHF pairs 模拟反馈。该运行不会
生成新答案、收集新的人工标签，也不会根据下游训练增益更新选择器。
它只测试选择规则是否改善下游偏好分类任务，不能证明完整递归 RSI loop、
技术二、技术三、SFT、DPO 或官方 HH-RLHF 测试集性能。
