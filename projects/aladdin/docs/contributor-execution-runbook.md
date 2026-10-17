# 贡献者执行手册

本文档供服务器贡献者执行已经冻结的 Aladdin 实验。

## 执行前

~~~bash
git status --short
git rev-parse HEAD
export ALADDIN_ROOT="$PWD"
export PYTHONPATH="$ALADDIN_ROOT"
python -m pytest local_experiments/tests core_system/tests -q
python scripts/check_aladdin_install.py
~~~

确认数据集和模型位于代码检出目录之外，endpoint 提供的是预期模型，
并且所选 GPU 当前可用。不要把私有路径或凭据写入提交的 profile。

## 执行期间

使用实验 issue 中的精确命令。将日志和输出写入仓库外的目录，例如
$ALADDIN_RUNS。恢复运行时必须使用相同的 run identifier 和不可变输入文件。
在预注册的最终评估步骤之前，不得读取 final-test 标签。

## 执行后

检查输出状态、已完成条件数、数据哈希、预算和 final_test_accessed 标志。
运行指定的分析器，只将小型且不含敏感信息的摘要附加到 pull request 或
issue。除非负责人明确批准经过清理的构件，否则原始输出必须保留在服务器上。
