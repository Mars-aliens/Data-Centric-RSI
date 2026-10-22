# 模型配置

## 首轮主实验模型选择

新路线的主实验只固定两个模型角色，不需要把所有 Qwen 型号或 Kimi 都跑一遍：

注意阶段差异：先验证固定流式 baseline 时只需要 Task Model；开始 T1 候选生成时才需要 Decision LLM。Decision LLM API 未配置好，不应阻止先做数据审计和 Task Model baseline smoke。

| 角色 | 默认选择 | 部署方式 | 任务 |
| --- | --- | --- | --- |
| Task Model | `Qwen/Qwen3-8B` | 服务器本地 Hugging Face 权重 | HelpSteer2 质量分类、预测和基于 parent checkpoint 的 LoRA 续训 |
| Decision LLM | 首轮 pilot 用百炼 `Qwen3.8-Flash` | 百炼 OpenAI-compatible Chat Completions API | 读取允许的运行证据并提出有界 Data System 修改 |

建议首轮 pilot 使用本地 Qwen3-8B + 百炼 Qwen3.8-Flash。根据截图价格，Flash 输入约 0.8 元/百万 tokens、输出约 2.7 元/百万 tokens；Qwen3.8-Max 对应约 12/36 元，约贵 15 倍。由于 T1 需要多轮候选提案，先用 Flash 控制 pilot 成本，再用实际提案有效性决定正式模型。此建议不是效果排名，尚未在新服务器实测。Task Model 的本地快照路径与不可变 revision 必须由执行者填写。截图中的型号/版本标识不能未经 API 请求验证就当作 API `model_id`；执行者按控制台/API 文档填入真实调用 ID。

### 是否需要其他模型

- **Qwen3-8B Task Model：需要。** 当前 runner 需要本地 causal-LM 权重做 LoRA parent-checkpoint 续训。8B 是首轮固定基准，不代表任务指标最优；正式运行前必须验证 tokenizer 标签编码、LoRA target modules、训练/保存/恢复和显存。
- **Decision LLM：T1 候选提案需要。** 首轮 pilot 用 Qwen3.8-Flash 检查真实提案和成本；若产生不可解析、越界或无效修改，且经 prompt/协议修正后仍不达预设门槛，再用 Qwen3.8-Max 做升级对照。Max 成本较高，不能只因其是旗舰型号就认定任务效果更好。
- **额外 evaluator LLM：不需要。** 当前可信 runner 根据 Task Model 预测和到期标签计算指标；Decision LLM 不给自己的候选打分，也不接触 held-out/final-test。
- **Qwen3.8-Flash：首轮 pilot 推荐。** 若 pilot 通过预先定义的提案质量门槛，正式主实验继续固定 Flash；如需改用 Max，必须在正式运行前完成独立 pilot 并冻结 Max，不能在正式 run 中途切换。
- **百炼 `Decision Model`：当前不需要。** 卡片描述偏高频结构化决策；当前 T1 要生成代码修改。除非单独验证它能满足候选提案协议，否则不作为主模型。
- **Kimi K3：当前不需要。** 仅在研究 Decision LLM 选择的独立对照中使用，结果不并入 Qwen 主实验。
- **更大或更新的本地 Qwen 权重：当前不需要。** 只有在其权重确实可本地训练、架构被 Transformers/PEFT 支持、GPU 资源足够并通过完整 smoke 后，才可作为独立 Task Model 对照。能调用托管 API 不等于能做本地 LoRA 续训。

因此本研究当前不回答“哪个模型绝对最好”，而是固定一个可验证的模型组合来检验 Data System 方法。若将来比较模型，应独立注册实验，固定数据、预算、prompt、工具协议并使用独立 run；不能把模型效果差异误当成方法效果。

### Flash/Max 筛选门

这是正式实验前的工程筛选，不作为方法效果实验。先固定 10 条不含 held-out/final-test、只使用当时已释放信息的代表性提案输入；将同一输入、同一 system/user prompt 和同一解码参数分别交给 Flash 与 Max，pilot 目录独立保存。按以下规则决定：

1. API/JSON/schema 必须 10/10 成功；候选路径、大小和允许文件类型等机械校验不得出现越界。
2. 至少 8/10 个提案通过本地 proposal/bundle 校验，且改动后 bundle 源文件哈希确有变化。该机械检查只能排除空 patch 或原文未变，不能证明代码语义发生有效改变。
3. 对提案内容另行记录是否改动可执行逻辑、修改理由与观察证据是否对应、是否有明确可验证的预期；注释/格式变化不能算作有意义的候选。此内容审查不要求候选改善任务指标或被 T1 接受。
4. 记录每条输入的有效性、错误类型、input/output tokens、费用和延迟。正式决策预算必须在 pilot 前设上限；超过上限即不选该配置。

先测 Flash。满足上述门槛且成本在预算内，就冻结 Flash 为主实验 Decision LLM，不必再跑 Max。Flash 未通过时先修 prompt/协议并在同一固定输入上重测一次；仍未通过再评估 Max。若 Max 通过则正式实验冻结 Max；若两者均未通过，暂停 T1，修复提案协议，不进入正式运行。两模型筛选结果不计入正式实验，也不能用 held-in/held-out 的候选效果挑选模型。

## 模型角色

MLE-LEGO 使用两个独立模型角色。项目不预设所有执行服务器都装有同一型号，也不要求沿用先前服务器的模型或路径。执行者先盘点新服务器的本地权重、GPU 资源和可调用 API，再为本次实验选择模型；正式配对运行开始前固定模型标识、版本、运行位置和预算。

| 角色 | 用途 | 配置边界 |
| --- | --- | --- |
| Task Model | 在 HelpSteer2 流上预测，并用已释放标签做 LoRA 续训 | 当前 runner 从执行服务器本地 Hugging Face causal-LM 权重加载，要求 Transformers/PEFT 兼容 |
| Decision LLM | 读取允许观察到的运行证据，提出 Data System 候选修改 | 通过 OpenAI-compatible Chat Completions endpoint 调用，可选服务器可访问的托管 Qwen、Kimi K3 或兼容的本地服务 |

两个角色可以使用不同模型。Qwen 的托管 API 可以作为 Decision LLM，但当前 Task Model 训练器需要本地可加载权重，不能把纯 API 访问当作 LoRA 训练接口。Kimi K3 在当前实现中是托管 Decision LLM；除非另有可训练权重和兼容训练接口，否则不能作为本地 LoRA Task Model。旧实验中的 Qwen2.5-7B-Instruct、Qwen3-4B 等值属于历史配置，不覆盖本节的新路线默认组合。

## Task Model 选择

Task Model 依赖执行服务器提供的本地模型目录及不可变 revision。若选择 Qwen，不能只按系列名判断兼容：还要确认具体模型是当前训练器支持的 causal-LM 架构、Tokenizer 可将监督标签编码成单 token、PEFT target modules 存在且 GPU 显存足够。多模态或非 causal-LM 型号不默认兼容。换用其他架构也需先扩展/验证训练适配器，不能因为 Transformers 能加载就认定实验兼容。

把以下模板复制到服务器私有配置目录。模型 ID 默认采用 Qwen3-8B；路径和不可变 revision 按该服务器的实际快照填写，不提交到 Git：

```json
{
  "base_model_id": "Qwen/Qwen3-8B",
  "base_model_path": "/models/Qwen3-8B/snapshots/<immutable-snapshot>",
  "base_model_revision": "<immutable-snapshot-id>",
  "max_steps": 100,
  "learning_rate": 0.0002,
  "per_device_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_length": 1024,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"]
}
```

此模板不预设 GPU 显存额度。根据新服务器的实际分配，在私有配置中按需加入 `max_gpu_memory_gib`；该值是进程软限制，不等价于 GPU 隔离或容器显存硬配额。

`base_model_path` 和 `base_model_revision` 必须指向同一份冻结权重。同一配对实验中的稳定分支与候选分支使用同一 Task Model。改变模型 ID、权重 revision、Tokenizer 或 LoRA 配方都要创建独立 run；模型选择不同的贡献者结果不能直接合并成同一个方法对比。

## Decision LLM 服务

Decision LLM 使用模型服务商提供的 API 模型 ID，不使用模型市场网页地址作为 `model_id` 或 API endpoint。首轮候选为百炼托管 Qwen3.8-Flash；Max 是质量门槛不通过时的备选。型号卡片版本名不一定等同于调用 API ID，需确认控制台 API ID、账号权限、地域、endpoint 和结构化输出能力后再 smoke。百炼控制台的模型条目不自动保证支持当前兼容协议、JSON mode、上下文长度或输出限制。Kimi K3 仅作为后续独立模型对照，不替换主实验配置。

服务器私有 profile 的 `decision_model` 示例也提供为 `code/technologies/technology_one/decision_model.qwen.example.json`：

```json
{
  "provider": "bailian_openai_compatible",
  "model_id": "<百炼控制台/API 文档确认的 Qwen3.8-Flash 调用 ID>",
  "revision": "<服务端版本；若不提供则记录模型 ID 和调用日期>",
  "api_url_env": "LEGO_POLICY_API_URL",
  "api_key_env": "LEGO_POLICY_API_KEY",
  "temperature": 0.0,
  "top_p": 1.0,
  "max_output_tokens": 512,
  "json_mode": true
}
```

在服务器通过环境变量注入地址和凭据，密钥不得写入 profile、命令行参数、日志或 Git：

```bash
export LEGO_POLICY_API_URL="<服务商的兼容模式 Chat Completions endpoint>"
export LEGO_POLICY_API_KEY="<从密钥管理器注入>"
```

支持的 provider 标签包括 `openai_compatible`、`vllm_openai_compatible`、`dashscope_openai_compatible` 和 `bailian_openai_compatible`。这些标签都走相同的兼容请求协议，不会自动创建服务、查找可用模型或替账号开通权限。若 endpoint 不支持 `response_format: {"type":"json_object"}`，可在 profile 中设 `json_mode` 为 `false`；客户端仍会解析并校验 JSON。

托管模型没有服务端不可变 revision 时，应记录 API 模型 ID、profile 哈希、提示词哈希、调用日期和服务返回的 token 用量；不能声称权重级完全可复现。

## 多服务器协作约定

每台服务器可以有不同的绝对模型路径、GPU 编号和密钥，但同一组可合并的正式对照必须使用相同的 Task Model ID/revision、Tokenizer、训练配方、Decision LLM API model ID/服务版本、提示词、数据 revision、代码 commit、种子和预算。托管模型若不能固定权重 revision，应记录调用日期、服务地域/endpoint 类型、API model ID 和完整非敏感请求参数，并将不同时间或不同模型的结果分开报告。

本地路径、API key、账号信息和服务器私有 profile 留在服务器，不进 GitHub。共享实验报告只记录复现实验所需的非敏感模型标识、revision/日期、配置哈希、硬件类型和结果证据。新贡献者先按自己的服务器资源做 smoke；只有使用相同 Qwen3-8B 权重 revision/Tokenizer/训练配方和同一正式冻结的 Decision LLM API model ID/提示词协议，且通过兼容性检查后，才加入同一主实验矩阵。物理路径和 GPU 编号可以不同；模型或服务版本不同则单独报告。

## 后续执行顺序

1. **盘点新服务器**：确认本地 Qwen3-8B 权重、Python/PyTorch/Transformers/PEFT/CUDA 版本、实际 GPU 与显存、HelpSteer2 文件、百炼地域和 API 权限。路径、GPU 编号、API endpoint/key 均由执行者私下配置，不从旧服务器复制。
2. **审计真实数据**：生成 HelpSteer2 manifest，检查分组、划分、样本数、类别比例、批次数、数据哈希和反馈可见性；数据审计不通过就停止在这里。
3. **Task Model smoke**：用 Qwen3-8B 做短训练、预测、adapter 保存和 parent-checkpoint 恢复，确认标签 token/LoRA target modules/显存正常。固定同一 Task Model 给 stable/candidate 两个分支。
4. **Decision LLM smoke**：按控制台确认的 Qwen3.8-Flash API ID 运行 `scripts/smoke_decision_endpoint.py`，验证鉴权、JSON/schema 和真实 token 成本；随后用少量候选提案评估内容质量。若质量不达预设门槛，再对 Max 独立重复这一步并在正式实验前冻结一种模型。失败时排查 API ID、地域、权限、endpoint 或 JSON mode，不用换模型掩盖配置问题。
5. **固定流式 baseline**：先运行 `scripts/run_long_horizon_qwen.py` 短流，验证先预测、后释放反馈、再续训及断点恢复。baseline 不通过，不进入 T1。
6. **T1 安全验证**：构建 digest 固定的候选镜像，先 preflight、再无故障短 run，然后用独立 run 逐类注入故障，验证候选隔离、稳定分支继续和恢复审计。
7. **冻结并开展正式 T1**：冻结代码 commit、Task Model revision/配方、Decision API ID/提示词、数据哈希、预算、种子和统计方案后，才开始多种子长周期运行。
8. **按技术递进**：T1 长周期安全机制验证完成后做 T2 经验生命周期与长周期验证，再做 T3 延迟反馈归因和跨批次保留/回退，最后开展 T1+T2+T3 组合与消融。

模型可调用/测试通过只证明执行条件成立，不证明模型或方法最优；正式结论来自冻结配置下的长周期成对实验。

## 技术一候选

候选提案命令在可信本地进程中调用 Decision LLM，再把无凭据的候选源码 bundle 交给 Docker。Decision API 凭据不会传入容器。候选镜像必须用 digest 固定；稳定分支与候选分支若都使用 GPU，分配不同物理 GPU 或 MIG 设备。Docker 不提供 GPU 显存硬配额。

完整的 T1 参数、命令和结果判读见[技术一安全自进化执行手册](technology-one-safe-evolution-runbook.md)。
