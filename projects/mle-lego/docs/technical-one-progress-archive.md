# LEGO 技术一：当前完整进度汇总

更新时间：2026-10-14  
实验对象：HelpSteer2 Preference / Technology One   
当前阶段：真实 decision-LLM Harness evolution 正式多 seed 实验进行中

## 1. 当前结论

目前已经完成并冻结了技术一的第一阶段组件矩阵和 matched fixed baseline 对齐；真实 decision LLM 驱动的多批次 Harness evolution pilot 也已经成功跑通。

因此现在已经证明：

1. 数据协议、Task Model、Evaluator 和服务器运行链路可以稳定执行。
2. 组件依赖相关的 deterministic candidate matrix 可以在统一训练和评测预算下比较。
3. 真实 decision LLM 可以提出 Harness 修改，系统可以训练和评估候选分支，并在 held-in 下降时拒绝候选、执行 rollback。

目前还没有证明：

1. 完整 LEGO 在多 seed 上稳定优于 fixed Harness。
2. 真实 decision LLM 的长期演化平均带来性能提升。
3. LEGO 优于所有外部方法适配或所有 required controls。

正式 10-seed evolution 结果完成前，不使用 final test，也不写最终方法优越性结论。

## 2. 数据和协议冻结

### 2.1 数据集

- 数据源：`nvidia/HelpSteer2`
- source revision：`990b2711a36180dd19d9c94b8627844866f8982a`
- 任务：`helpsteer2-preference`
- 训练 JSONL hash：`5bd88e03cb93527131d2a75b30940ea8a969a23096f8f529eb0e9bb114f4d6e9`
- final-test JSONL hash：`54dd7be32c665b2d63da3caa488c9674d631ff87e46763b64871580b4a7a1e8c`
- protocol hash：`1cd04e9c057a8d79deff4619db774598be895cfc23aa716de83191778d01b3d1`
- freeze manifest hash：`a35dc0eb89a4cad100b38c699cf73992e66f9cfd8c98f74d18ed9d212f5074d7`

### 2.2 Split 和规模

| split | records | source groups | 说明 |
|---|---:|---:|---|
| seed | 14,296 | 7,148 | 可用于训练流 |
| held-in | 3,998 | 1,999 | 候选选择和接受判断 |
| held-out | 2,030 | 1,015 | 回归报告，只读 |
| final_test | 1,038 | 519 | 最终一次性报告，当前未使用 |

训练数据合计 20,324 条，source group 合计 10,161；绝大多数 group size 为 2，另有 1 个 group size 为 4。正式运行使用 train group budget `1000`，自然支持设计为主，未使用 common-support 结果作为主结论。

### 2.3 模型和训练配置

- Task Model：Qwen3-4B preference LoRA
- decision model：`Qwen/Qwen2.5-7B-Instruct`
- decision model revision：`a09a35458c702b33eeacc393d103063234e8bc28`
- LoRA：`r=8`，`alpha=16`，`dropout=0.05`
- learning rate：`0.0002`
- max length：`1024`
- training steps：每个 candidate `100`
- gradient accumulation：`4`
- per-device batch size：`2`
- Task Model GPU memory limit：`40 GiB`
- decision temperature：`0.0`
- final test：始终保持锁定

正式矩阵 seed 集合：`17 23 29 31 37 41 43 47 53 59`。

## 3. 第一阶段：10-seed component matrix

### 3.1 审计状态

10 个 seed 已完成，审计结果：

```text
status: PASS
run_count: 10
seeds: [17, 23, 29, 31, 37, 41, 43, 47, 53, 59]
missing: []
unexpected: []
budgets: [1000]
```

每个 seed 有 8 个方法臂：

- `fixed`
- `independent-edit`
- `selector-only-edit`
- `filter-only-edit`
- `dependency-aware-joint-edit`
- `generator-filter-joint-edit`
- `selector-filter-joint-edit`
- `generator-selector-filter-joint-edit`

这里的“8 个方法臂”指同一个 seed 下的 8 个 deterministic Harness 候选，不是 8 个独立数据集，也不是 8 次完整 LEGO 长程自进化。

### 3.2 10-seed 结果

指标为 10 个 seed 的均值；`delta_out` 是相对于同 seed `fixed` 的 held-out 配对差值均值；`wins` 是 held-out 超过同 seed fixed 的次数。

| method | held-in mean | held-out mean | delta_out | wins |
|---|---:|---:|---:|---:|
| fixed | 0.2451 | 0.2513 | +0.0000 | 0/10 |
| independent-edit | 0.2377 | 0.2455 | -0.0059 | 5/10 |
| selector-only-edit | 0.2453 | 0.2498 | -0.0015 | 5/10 |
| filter-only-edit | 0.2543 | 0.2562 | +0.0048 | 6/10 |
| dependency-aware-joint-edit | 0.2433 | 0.2472 | -0.0041 | 4/10 |
| generator-filter-joint-edit | 0.2637 | 0.2648 | +0.0134 | 6/10 |
| selector-filter-joint-edit | 0.2867 | 0.2917 | +0.0404 | 8/10 |
| generator-selector-filter-joint-edit | 0.2514 | 0.2551 | +0.0037 | 4/10 |

### 3.3 第一阶段解释

当前最稳定的 component-matrix 结果是 `selector-filter-joint-edit`：held-out mean `0.2917`，相对 fixed 的平均配对提升 `+0.0404`，10 个 seed 中 8 次胜出。

但 `dependency-aware-joint-edit` 在该矩阵中并没有优于所有组件候选，held-out mean 为 `0.2472`。这说明：

- 组件联合修改可能有效；
- selector/filter 相关修改在当前 HelpSteer2 配置下更重要；
- 不能仅凭这一阶段宣称 dependency-aware joint edit 是最优 LEGO 方法；
- 第一阶段没有 decision LLM，`llm_calls=0`，因此不能作为完整 LEGO 自我改进证据。

早期 7-seed 汇总已经被 10-seed 结果替代，不再作为最终统计结果。

## 4. Matched fixed baseline 对齐

10 个正式 seed 的 matched fixed baseline 已完成语义对齐审计：

```text
SEMANTIC_ALIGNMENT: PASS
```

已核对的内容：

- train group hash 一致；
- Task Model training configuration 一致；
- adapter configuration 语义一致；
- held-in / held-out 指标一致；
- `final_test_used=False`；
- target modules 的顺序差异属于 JSON 表示差异，语义内容一致。

需要保留的技术说明：不同运行目录的 `artifact_hash` 可以不同，因为 artifact JSON 中包含运行路径和运行元数据；真正的 LoRA adapter hash 在已核对的 seed 上一致。因此报告中应优先使用 adapter 内容 hash 和 semantic config，不应把 artifact hash 差异误写成模型不一致。

这一部分证明的是 fixed baseline 的可复现性和公平比较基础，不是 LEGO 性能提升本身。

seed 17 的 matched anchor 例子如下：

- matrix fixed held-in macro-F1：`0.1976785086`
- matrix fixed held-out macro-F1：`0.2014421959`
- matched baseline held-in macro-F1：`0.1976785086`
- matched baseline held-out macro-F1：`0.2014421959`
- actual adapter hash：两者一致，为
  `05207721331da27aad71680f74e05af25db8380e52ecbd631638eea75d80c04e`
- artifact hash：运行目录相关元数据不同，不作为模型一致性判断依据

## 5. 第二阶段：真实 decision-LLM evolution pilot

### 5.1 Pilot 配置

- run：`qwen-technology-one-evolution-seed-17-batches-3-20261013-173836`
- seed：`17`
- batches：`3`
- train group budget：`1000`
- Task Model GPU：GPU 6
- decision endpoint：vLLM OpenAI-compatible `/v1/chat/completions`
- output root：`runs/qwen-technology-one-evolution-pilot-retry-v2`
- 总耗时：`1404.3` 秒，约 `23 分 24 秒`

### 5.2 Pilot 已证明的机制

运行中真实调用了 decision LLM，并成功产生了候选 Harness 修改。候选分支被单独训练和评测，最后一个候选的 transition 为：

- active Harness version：`lego-harness-fa3f83cf4035`
- candidate Harness version：`lego-harness-5777c6051262`
- 修改组件：`generator`
- 修改内容：扩大 `allowed_buckets` 到 `0..9`
- active train group hash：`c05360f660175325bb638674c268b947b259bbdf7131b31b6f879accc8ac1bc8`
- candidate train group hash：`bfc42df85d4ed125b99f3a207e75c02c6ee9368dfbccbb539cc1576f341023a7`

### 5.3 最后一个候选的结果

| branch | held-in macro-F1 | held-out macro-F1 |
|---|---:|---:|
| active | 0.3292178886 | 0.3268155995 |
| candidate | 0.1975134882 | 0.2014421959 |

候选 held-in 明显下降，因此系统正确地：

- 将 candidate 标记为 `rejected`；
- 执行 rollback；
- 恢复到 `lego-harness-fa3f83cf4035`；
- 不使用 held-out 进行接受判断；
- 不使用 final test。

该 transition 的 decision policy cost 为 `llm_calls=1`、`llm_tokens=1113`。这证明当前链路不是预设 candidate matrix，而是真实的 decision-LLM proposal -> candidate evaluation -> reject/rollback 流程。

这个结果不是“pilot 失败”。它证明了系统可以拒绝有害修改；但单个 pilot 中的拒绝不能证明平均性能提升。

## 6. 当前正式多 seed evolution

正式运行目录：

```text
runs/qwen-technology-one-evolution-formal
```

配置为 10 个 seed、每个 seed 3 个 batch、GPU 6 串行执行。启动时状态为：

- seed 17 已启动；
- 其余 seed 等待串行执行；
- 每个 3-batch run 约包含 5 次 Task Model candidate train/eval；
- 单个 seed 预计约 15--30 分钟，受 GPU 和 evaluator 负载影响；
- 10 个 seed 串行预计约 4--5 小时。

截至本记录的原始批处理阶段，seed 17 和 23 已完成，seed 29 曾在 decision LLM 返回不满足依赖闭包的 proposal 后中止；随后已完成 vLLM 恢复和 seed 29 retry。seed 31--59 仍需按服务器当前状态检查。因此正式 evolution 的均值、标准差和 paired delta 尚不能填写。正式运行必须先通过 evolution audit，再进入统计汇总。

seed 29 的训练和 evaluator 均已完成；失败发生在下一轮 proposal 解析阶段。原 runner 将这个可恢复的模型协议错误当作致命异常。修复后，editor 会对语义无效 proposal 发送一次带具体错误原因的 schema retry；两次仍无效时记录 `proposal_validation_failed`、reject/rollback 并继续该 seed，而不会让整个多 seed batch 停止。依赖闭包校验本身没有放宽。

## 7. 已完成的代码和工程工作

### 7.0 长期 Harness 验证模式（本地已加入）

为匹配宏观研究目标，已在本地 runner 上加入长期运行能力；现有
`BATCHES=3` pilot 和 formal wrapper 的默认行为不变。新增内容包括：

- 每个 batch 后原子写入 `endurance_state.json` 和 `progress.json`；
- 保存 active Harness、proposal、transition、reflection 和累计 budget；
- 支持使用同一个 `run_id` 通过 `--resume` 继续运行；
- decision policy、trainer、evaluator 的临时命令失败支持指数退避重试；
- candidate 执行失败时记录 reject/rollback，不提交损坏版本；
- 记录 resume 次数、重试次数、候选执行失败次数和 checkpoint 写入次数；
- 新增 `scripts/run_technology_one_evolution_endurance.sh`，默认目标为 300 cycles；
- 新增 `scripts/audit_technology_one_evolution_endurance.py`。

长期模式的定位是系统级验证：连续运行、持续产生 proposal、接受有效变化、拒绝或回退无效变化，并能在中断后继续。它不是把 3-batch pilot 重新命名为长期证据；正式长期结果仍需在服务器上实际跑完并审计。

### 7.1 Evolution runner

已加入：

- decision LLM 真实 Harness proposal；
- 多 batch 运行；
- held-in-only acceptance；
- held-out report-only；
- reflection 传递到后续 proposal；
- accepted / rejected candidate；
- rollback 事件；
- final-test lock；
- LLM、训练和 evaluator budget accounting。

### 7.2 服务器适配器修复

修复了：

```text
ValueError: Unknown budget fields: ['training_records_seen']
```

原因是 Qwen trainer 将 `training_records_seen` 作为诊断字段放在 cost 中，而 adapter 只识别正式 `Budget` 字段。现在该字段会被忽略，其他未知字段仍然报错。

### 7.3 本地验证和报告工具

已新增：

- `code/local_experiments/lego/evolution_reporting.py`
- `code/local_experiments/lego/test_evolution_reporting.py`
- `scripts/audit_technology_one_evolution.py`
- `scripts/summarize_technology_one_evolution.py`
- `scripts/run_technology_one_evolution_formal_batch.sh`
- decision proposal schema retry 和 invalid-proposal continuation

本地测试结果：

```text
41 passed
```

新部署包：

```text
lego_server_bundle_formal_tools_20261014-021733.zip
SHA256: 31846F7CB445CAF72DB74EA975FFC526E5C449D30ACCB6474E87DD4C34F95DF1
```

## 8. 当前实验边界和未完成事项

### 8.1 主要运行目录

| 内容 | 目录 |
|---|---|
| 10-seed component matrix | `runs/qwen-aligned-component-matrix` |
| 10-seed matched fixed baseline | `runs/qwen-fixed-baseline-aligned` |
| evolution pilot | `runs/qwen-technology-one-evolution-pilot-retry-v2` |
| formal evolution | `runs/qwen-technology-one-evolution-formal` |

### 已完成

- HelpSteer2 数据和 split 冻结；
- Qwen Task Model fixed baseline 配置冻结；
- 10-seed 8-arm component matrix；
- 10-seed matched fixed baseline 语义对齐；
- 真实 decision-LLM 3-batch pilot；
- proposal、candidate evaluation、reject、rollback 链路；
- 正式 10-seed evolution 已启动。

### 未完成

1. 正式 10-seed evolution 的全部 summary 和统计汇总。
2. 与真实多 batch evolution 匹配的 fixed Harness control。
3. text-only reflection/memory control。
4. same-memory-no-process-update control。
5. 真正的 one-batch / three-batch selection horizon 对照。
6. 所有 controls 的统一 paired statistics。
7. final test 的一次性最终报告。

特别注意：当前 `server_baseline_matrix.py` 中的 `three-batch-selection` 是 deterministic one-shot adaptation matrix 中的标签，不能直接当作真实的三批次 branch-survival control。正式论文中必须把它标注为 adaptation/scaffold，或者在冻结协议后补写真正的多批次控制实现。

## 9. 正式报告的统计规则

正式 evolution 完成后，报告至少包括：

- 每个 seed 的 initial active 和 final active held-in / held-out macro-F1；
- final active 相对 initial active 的 paired delta；
- 相对于 matched fixed control 的 per-seed paired delta；
- mean、sample SD、95% confidence interval；
- accepted / rejected / rollback 次数；
- decision LLM calls、tokens、training steps、evaluator calls 和 wall-clock；
- proposal 的 component、patch、Harness version 和 parent version；
- 所有 final-test 使用状态；
- 失败候选和 rollback 案例。

选择规则：held-in 用于接受/拒绝，held-out 只作回归报告，final_test 在协议和方法选择完全冻结后只使用一次。

## 10. 当前下一步

1. 等正式 10-seed evolution 完成。
2. 使用新的 audit 工具确认 10 个 seed、3 个 batch、LLM calls、transition 和 final-test lock 全部通过。
3. 使用新的 summary 工具生成 per-seed 表和均值/标准差。
4. 在不修改已冻结主方法的前提下，运行 matched multi-batch controls。
5. 完成 controls 后冻结统计方案。
6. 最后一次性运行 final test，并将其与开发阶段结果分开报告。

## 11. 可引用的阶段性表述

当前可以安全写成：

> 我们首先在冻结的 HelpSteer2/Qwen 协议下完成了十随机种子的 natural-support 组件矩阵。矩阵已完成审计，各随机种子的匹配固定基线也通过语义对齐检查。随后，我们验证了真实的多批次 Decision LLM 演化链路：模型提出受限的 Harness 修改，候选分支经过训练和评估；当 held-in 表现下降时，系统拒绝候选并执行回滚。十随机种子正式演化研究仍在进行，final-test 评估继续锁定。

当前不能写成：

> LEGO 已经证明，在长程运行中显著优于所有基线并实现自我改进。

因为正式多 seed evolution 和匹配的多批次 controls 尚未完成。
