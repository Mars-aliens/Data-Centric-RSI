# 数据经验持续治理实验

按[协议 v1](protocol-v1.md)执行四臂对照：无经验、文字反思、无治理可执行经验、生命周期治理经验。先跑每臂 4 批 smoke；通过后每臂用种子 17/23/29 跑 10 批 pilot。所有臂共享数据流、Task Model、训练参数、反馈延迟和预算。

服务器环境和数据按 [`docs/server-quickstart.md`](../../docs/server-quickstart.md) 准备。Smoke 配置示例：[`task_model.qwen.smoke.example.json`](../../code/technologies/technology_two/task_model.qwen.smoke.example.json)；Pilot 配置示例：[`task_model.qwen.example.json`](../../code/technologies/technology_two/task_model.qwen.example.json)。复制到服务器私有目录后填写绝对权重路径和不可变 revision。Decision LLM、Kubernetes 和 Docker 不是技术二运行前置条件。

```bash
cd "$LEGO_ROOT"
export TRAIN_JSONL="$LEGO_ROOT/data/helpsteer2-train.jsonl"
export VALIDATION_JSONL="$LEGO_ROOT/data/helpsteer2-validation.jsonl"
export TASK_CONFIG="$LEGO_ROOT/server-private/t2-task-model-smoke.json"
export OUTPUT_ROOT="$LEGO_ROOT/runs/technology-two-smoke"

for ARM in no_experience reflection_only executable_unmanaged lifecycle_governed; do
  "$TRAIN_PY" scripts/run_technology_two_qwen.py \
    --train-jsonl "$TRAIN_JSONL" \
    --validation-jsonl "$VALIDATION_JSONL" \
    --task-config-json "$TASK_CONFIG" \
    --output-dir "$OUTPUT_ROOT/$ARM/seed-17" \
    --run-id "t2-smoke-$ARM-seed-17" \
    --arm "$ARM" --seed 17 --stream-seed 17 \
    --groups-per-batch 50 --feedback-delay 2 --max-batches 4
done
```

smoke 全通过后，在新输出根目录使用 pilot 配置，逐臂运行种子 `17 23 29`，每次 `--max-batches 10`：

```bash
export TASK_CONFIG="$LEGO_ROOT/server-private/t2-task-model-pilot.json"
export OUTPUT_ROOT="$LEGO_ROOT/runs/technology-two-pilot"

for REGIME in randomized interleaved abrupt alternating; do
  REGIME_ROOT="$OUTPUT_ROOT/$REGIME"
  for SEED in 17 23 29; do
    for ARM in no_experience reflection_only executable_unmanaged lifecycle_governed; do
      "$TRAIN_PY" scripts/run_technology_two_qwen.py \
        --train-jsonl "$TRAIN_JSONL" \
        --validation-jsonl "$VALIDATION_JSONL" \
        --task-config-json "$TASK_CONFIG" \
        --output-dir "$REGIME_ROOT/$ARM/seed-$SEED" \
        --run-id "t2-pilot-$REGIME-$ARM-seed-$SEED" \
        --arm "$ARM" --seed "$SEED" --stream-seed "$SEED" \
        --stream-regime "$REGIME" \
        --groups-per-batch 50 --feedback-delay 2 --max-batches 20
    done
  done
done
```

不可复用 smoke 输出目录或在同一 GPU 并行运行多臂。出现失败时保留该 run 目录；修复后使用新的 run ID 与输出目录。

Pilot 完成后验证配对配置并汇总：

```bash
for REGIME in randomized interleaved abrupt alternating; do
  "$TRAIN_PY" scripts/summarize_technology_two.py \
    --root "$OUTPUT_ROOT/$REGIME" --seeds 17 23 29 --expected-batches 20 \
    --output "$OUTPUT_ROOT/$REGIME/technology-two-summary.json"
done
```
