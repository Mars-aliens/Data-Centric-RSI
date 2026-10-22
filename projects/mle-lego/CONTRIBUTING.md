# 贡献指南

## 范围

本仓库承载 Data-Centric RSI 总项目中的技术一实现。变更应聚焦于 Harness、评估协议、可复现性及相关文档。

## 提交变更前

1. 创建名为 `feature/<short-name>`、`fix/<short-name>` 或 `experiment/<short-name>` 的分支。
2. 在仓库根目录运行 `python -m pytest -q`。
3. 对修改过的脚本运行 shell 语法检查：`bash -n <script>`。
4. 行为或证据发生变化时，更新对应协议或状态文档。

## 实验变更

除非明确提交协议修订，否则以下项目必须保持不变：

- 数据切分和来源版本；
- 随机种子集合和组预算；
- Decision LLM 模型及版本；
- held-in 接受规则；
- held-out 仅报告规则；
- final-test 锁定状态；
- 候选训练和评估器预算。

如需变更其中任一项，记录原因、更新协议并使用新的输出目录。不得覆盖旧正式运行结果来制造可比的假象。

## 贡献者执行

服务器实验由维护者与执行者直接协商安排，并基于已记录的 commit 和冻结协议执行。执行者应遵循
`docs/server-quickstart.md`，保存运行报告和所需证据，并只报告事实，不解释方法优越性。维护者负责协议、结果解释及 `docs/current-status.md` 的更新。

## 不得提交到 Git 的文件

不要提交凭据、私有服务器配置、数据集、模型权重、
检查点、`runs/`、`logs/`、`tmp/`、`.venv/`、生成的 zip 文件或本地编辑器/LibreOffice 配置。仓库 `.gitignore` 覆盖常见情况，但提交前仍须检查 `git status`。

## 提交信息规范

提交标题使用简短的祈使式表达，例如：

```text
fix: reject incomplete dependency closures
test: cover joint proposal validation
docs: record formal comparison protocol
```
