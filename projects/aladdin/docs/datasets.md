# 数据集、许可与处理规范

数据源和固定版本以 `configs/datasets.json` 为准。下载数据和派生样本保存在 Git 忽略的 `data/` 目录中。

## PRISM

- 数据集：[HannahRoseKirk/prism-alignment](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)
- 固定版本：`18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`
- 下载文件：`conversations.jsonl`、`utterances.jsonl`、`metadata.jsonl`；明确排除 `survey.jsonl`。
- 许可条件不同：参与者编写的文本为 CC-BY-4.0，模型回复为 CC-BY-NC-4.0，且须遵守原模型提供方条款。
- 禁止尝试重新识别参与者。准备数据时，先依据 PII、语言和 moderation 元数据过滤。不得再分发下载的原始数据。
- 引用：Kirk 等，*The PRISM Alignment Project*，2024，[论文](https://arxiv.org/abs/2404.16019)，数据集 DOI：`10.57967/hf/2113`。

适配器只用历史上被选中的回答构建对话上下文。当前输入、response 或必需上下文缺少元数据，或被标记为 PII、非英语、moderation 不通过时，该候选会被排除。评分和 `if_chosen` 写入独立标签文件，供回放过程在查询选中后释放。按 `user_id` 切分，确保同一参与者的对话不会跨 split。由于原始公开数据仍可在本地读取，文件分离只是实验协议边界，不是安全隔离措施。

对话级 `open_feedback` 单独整理为反馈查询，不会复制到每条回答上。评论文本保存在标签 sidecar 中，仅对应 feedback-pool 用户，并应在匹配的对话查询被选中后才释放。

## HelpSteer2

- 数据集：[nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- 固定版本：`990b2711a36180dd19d9c94b8627844866f8982a`
- 下载文件：`train.jsonl.gz`、`validation.jsonl.gz`、`preference/preference.jsonl.gz`。
- 许可：CC-BY-4.0。保留署名和五个原始评分维度。
- 官方 validation 留作最终评价，不能用于策略、阈值或数据选择。

审计会检查 schema 完整性、重复 prompt 数和 train/validation prompt 重叠，只用哈希，不输出数据集文本。该方案要求重叠为零。不要把 HelpSteer2 和 PRISM 的结果合并后当作同一任务或同一人群的结果。

首轮 HelpSteer2 回放以 `helpfulness` 为偏好目标：较高分胜出，相同分数视为平局。其他四个维度，即 correctness、coherence、complexity、verbosity，分别报告。不要在未预注册规则的情况下把五维分数平均为一个标签。

## 命令

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.download_datasets --dataset all --confirm-research-terms
python -m scripts.prepare_prism --split-seed 17
python -m scripts.prepare_helpsteer2 --split-seed 17
python -m scripts.audit_datasets --dataset all
```

每次下载会写入本地 manifest，记录固定的数据集版本、文件大小和 SHA-256。PRISM 和 HelpSteer2 的处理脚本都会在读取数据前校验下载 manifest。HelpSteer2 按 prompt 组切分官方训练集，并将官方 validation 作为 `final_test`。五维评分写在独立标签文件中，首轮回放只使用 helpfulness，并保留并列关系。`data/` 已被 Git 忽略；提交前仍须检查忽略规则。

处理后的 manifest 记录数据集版本、划分规则、哈希和数量，不记录机器专属绝对路径。由于 PRISM 许可和隐私限制，且两个数据集均可从固定版本重新下载，仓库不提交数据集内容。
