# Aladdin 核心系统

本目录包含 Aladdin 使用的轻量级动作系统层，提供任务记录、构件、预算、
反馈记录、来源追踪以及本地演示和任务模型接口。Aladdin 的研究策略位于
local_experiments/aladdin，而不在这个底层包中。

从仓库根目录运行依赖较少的演示：

~~~bash
export PYTHONPATH="$PWD"
python -m core_system.demo
~~~

生成的演示构件应保存在 Git 之外。Aladdin 实验协议请参阅仓库根目录的
README。
