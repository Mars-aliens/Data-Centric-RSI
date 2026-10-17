# Decision Model 选择

## 首次真实运行的冻结建议

`fixed-harness-react` 和 `react-reflexion` 均使用 `Qwen/Qwen2.5-7B-Instruct` 作为 **Decision LLM**。

这是首次服务器实验的务实选择：该开放指令模型有稳定的 Transformers/vLLM 使用路径，支持英文 HelpSteer2 数据，能力足以选择结构化数据操作，同时控制器成本不会高于 Task Model 实验。本项目不声称它是 ReAct 或 Reflexion 的官方模型。

通过版本固定的 vLLM OpenAI 兼容端点提供服务，并使用以下设置：

```text
temperature = 0.0
top_p = 1.0
max_output_tokens = 512
structured JSON output = required
```

首次实验前固定 Hugging Face revision 或本地模型目录校验和。两个对照必须使用相同的模型版本、prompt 哈希、上下文长度、工具接口及 LLM 调用/Token 总预算。

仓库中的 `policy_command.py` 是该端点的 stdin/stdout 客户端。配置 `LEGO_POLICY_API_URL`，并将凭据保存在服务器密钥管理器中。命令返回一条提案，并将调用次数和 Token 成本记入 LEGO 预算账本。

## 对照方法定义

`fixed-harness-react` 使用固定基线流水线，包括数据处理、Task Model 训练、评估和运行日志。控制器可以为当前批次选择一项操作，但不会持久修改代码、Prompt、Skill 或 Memory 规则。

`react-reflexion` 使用完全相同的设置，并将控制器受限的文字反思额外保存到 `reflections.jsonl`。记录从下一次策略决策起可见，保留来源批次以及模型/prompt 来源信息，并以确定性方式检索，最多四条。它仍是文本记忆，不能直接修改 Harness。

Task Model 是独立的实验选择，必须在训练配置中另行选择并固定版本。

固定基线流水线是共同起点。LEGO 三项技术分别为依赖关系引导的联合修改、可执行经验复用和连续运行反馈选择。这些名称指研究机制，不是流水线版本标签。
