# MLE-Aladdin 贡献指南

Aladdin 和 MLE-LEGO 在总仓库中分别位于 `projects/aladdin/` 和
`projects/mle-lego/`，各自拥有独立的命名空间、实验计划、依赖和结果目录。
不要把 LEGO 代码或 LEGO 运行产物加入 Aladdin 目录。`D:\RSI\Aladdin` 是
Aladdin 的独立本地开发副本；GitHub 协作以总仓库对应目录为准。

## 代码修改

- 数据、模型和输出使用相对路径或环境变量。
- final-test 的加载必须由明确且可审计的协议控制。
- 每次协议或选择逻辑变化都要补充或更新针对性测试。
- 不要提交数据集、模型权重、checkpoint、日志、服务器 profile、凭据或运行结果。
- 实验输出必须保留既有数据哈希、seed、预算和泄漏审计元数据。

## 执行贡献者流程

实验负责人冻结 commit，并在 Issue 中写明数据路径、模型路径、endpoint、GPU、命令和预期输出位置。贡献者不得静默修改这些值。

```bash
git checkout <frozen-commit>
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python -m pytest local_experiments/tests core_system/tests -q
python scripts/check_aladdin_install.py
```

执行指定命令，将日志保存到仓库外，并报告：

- 精确 commit 和环境信息；
- 精确命令，以及用占位符表示的外部路径；
- 状态和已完成条件数；
- 数据集/模型哈希或不可变 revision；
- final-test 访问标记；
- 输出 JSON 路径和简短事实说明。

报告模板是 `docs/run-report-template.md`。
