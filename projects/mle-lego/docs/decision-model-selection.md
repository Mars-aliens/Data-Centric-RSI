# Decision Model 选择

本文记录当前长周期 T1/T2/T3 路线的模型选择建议。模型安排见[模型配置](model-configuration.md)：本地 Qwen3-8B Task Model；T1 pilot 先用百炼 Qwen3.8-Flash，再按预设质量门槛决定正式冻结型号。

## 首次真实运行的冻结建议

首轮 pilot 使用配置文件中登记的 Decision LLM 作为 **Decision LLM**。

模型应根据服务器可用接口、提案格式有效性和调用成本进行选择，并在 pilot 中固定版本。

通过版本固定的 vLLM OpenAI 兼容端点提供服务，并使用以下设置：

```text
temperature = 0.0
top_p = 1.0
max_output_tokens = 512
structured JSON output = required
```

运行前固定模型版本、prompt 哈希、上下文长度、工具接口及 LLM 调用/Token 总预算。

仓库中的 `policy_command.py` 是该端点的 stdin/stdout 客户端。配置 `LEGO_POLICY_API_URL`，并将凭据保存在服务器密钥管理器中。命令返回一条提案，并将调用次数和 Token 成本记入 LEGO 预算账本。

## Pilot 固定要求

同一 pilot 内固定模型版本、prompt 哈希、上下文长度、工具接口和调用预算。Task Model 与 Decision LLM 的职责、权重位置和调用方式见[模型配置](model-configuration.md)。
