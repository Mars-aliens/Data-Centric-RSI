# 外部基线交接说明

本目录定义外部方法的服务器交接协议，这些方法在
idea/MLE-Aladdin.md 中被引用。本目录不声称在本地实现这些论文方法。

官方数据目录是 dataset-registry.json，其中记录来源 URL、论文 URL、
可用时的不可变 revision、source-group 字段、反馈字段、许可证风险以及
该条目是否属于人工反馈。下载或训练前，服务器运行必须解析所有为 null
的 revision，并完成所有标记为 terms-review-required 的条目审查。

## 本地空运行

空运行会验证每种方法是否声明了相同的任务身份、split 顺序、初始检查点、
随机种子、预算字段和 final-test 锁：

~~~powershell
python -m local_experiments.aladdin.external_baselines
~~~

默认配置是 server-configs.local.json，它有意使用本地玩具任务，并使用
local-protocol-only 作为 commit 标记。输出只是执行计划，不会调用 LLM，
也不会训练模型。

## 服务器配置

复制 server-configs.json 的结构，并将所有占位符替换为：

- 真实数据集 manifest 和不可变数据集 revision；
- 官方仓库 commit；
- 初始任务模型 checkpoint；
- evaluator 和模型标识；
- 匹配的 seed、反馈、训练、评估器和 wall-clock 预算。

在所有方法和超参数冻结前，保持 final_test_unlocked 为 false。解锁
final-test 属于单独的只报告运行。

## 适配器边界

每个适配器都会返回包含共享核心所需审计字段的已验证计划。服务器实现必须
在这个边界之后补充官方仓库调用、模型运行器、数据转换和构件登记。
外部方法必须保留在本目录中，不能嵌入 core_system。
