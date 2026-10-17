# Aladdin 集成管线服务器交接说明

本文档只说明 Aladdin 集成管线，不是 LEGO 运行器。

## 前置条件

- 已安装 requirements.txt 的 Python 环境；
- 用于模型生成和质量判断的 OpenAI-compatible vLLM endpoint；
- 包含 seed、development 以及可选 held_in 记录的规范化 JSONL 输入；
- 包含需求假设的 requirements JSON 文件；
- file 模式可选的、由独立流程准备的反馈文件。

将 final-test 记录保存在单独的封存文件中。offline replay 中已有的标签
不属于新的人工反馈。

## Preflight

~~~bash
export ALADDIN_ROOT="$PWD"
export PYTHONPATH="$ALADDIN_ROOT"
export TRAIN_PY=python
export DATA_PATH=/path/to/seed-development-held-in.jsonl
export REQUIREMENTS_PATH=/path/to/requirements.json
export ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export FEEDBACK_MODE=offline-replay
bash local_experiments/aladdin/integrated_pipeline/server_preflight.sh
~~~

## 单轮运行

~~~bash
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_DIR=/path/to/external/aladdin-runs/round-001
export RUN_ID=aladdin-round-001
export ROUNDS=1
export CANDIDATE_COUNT=4
bash local_experiments/aladdin/integrated_pipeline/server_run_round.sh   | tee /path/to/external/aladdin-runs/round-001.log
~~~

只有在确实进行人工审查时，才使用 FEEDBACK_MODE=interactive。只有在
反馈文件由独立流程准备时，才使用 FEEDBACK_MODE=file。只有当这些构件
属于冻结实验计划时，才设置 CALIBRATION_PATH、QUALITY_AUDIT_PATH 或
RESUME_STATE。

## 多轮运行

设置 ROUNDS=3，从 recursive-state.json 恢复时使用同一个仓库外输出目录。
在运行报告中记录 commit、输入哈希、endpoint 模型、反馈模式、查询预算以及
final_test_accessed 标志。

这条管线是工程协议。任何关于数据效用的结论，都需要在 held-out 输入上
进行单独的下游训练评估。
