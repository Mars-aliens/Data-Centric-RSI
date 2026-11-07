# MLE-LEGO 服务器执行步骤

按顺序执行。所有 `<...>` 都要替换成当前服务器的实际值。模型、数据、密钥和运行结果不要上传 GitHub。

**执行入口只有本文件。** 实验方案文档和其他协议只用于说明研究问题、对照设计和代码边界，不是额外的操作步骤；除本文件明确列出的命令外，不需要跳转或补执行命令。

不是所有实验都能无前置地立即运行。依赖关系如下：

| 实验 | 开始前必须具备 | 是否依赖前一项实验结果 |
| --- | --- | --- |
| 技术一 smoke | 环境、数据、Task Model 配置、固定流程 smoke、Decision LLM API smoke | 是，先完成本文件第 1～5 节 |
| 技术二 smoke | 环境、同一份数据、单独的 T2 Task Model smoke 配置 | 不依赖技术一结果；为统一流程，安排在技术一 smoke 之后 |
| 技术三回放 | 已完成的技术一 stable/candidate source summary | 依赖技术一；完整的 3 批决定窗口 + 3 批未来窗口至少需要 6 个配对来源批次 |
| 8 组合 contract smoke | 已克隆代码和项目 Python 依赖 | 不依赖数据、模型或前面实验结果；使用 fake executor |
| 真实 8 组合实验 | 三个机制接入同一个真实任务执行器，并冻结 8 臂配置 | 当前没有可直接执行的真实组合 runner，不能仅靠本文件启动 |

## 1. 获取代码和配置环境

```bash
git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/mle-lego
git checkout <要运行的commit>

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

按服务器驱动从 PyTorch 官网选择 CUDA wheel，然后安装：

```bash
python -m pip install torch==2.7.1 --index-url <PyTorch选择器给出的CUDA源>
python -m pip install -r requirements-research.txt
python -m pip check
python -m pytest -q
```

设置项目路径：

```bash
export LEGO_ROOT="$PWD"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export LEGO_TASK_MODEL_DEVICE=cuda
mkdir -p "$LEGO_ROOT/server-private" "$LEGO_ROOT/data" "$LEGO_ROOT/runs"
bash scripts/check_lego_install.sh
nvidia-smi
```

## 2. 准备数据

下载固定 revision，并导出 runner 要求的字段：

```bash
"$TRAIN_PY" - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

dataset_id = "nvidia/HelpSteer2"
revision = "990b2711a36180dd19d9c94b8627844866f8982a"
fields = (
    "prompt", "response", "helpfulness", "correctness", "coherence",
    "complexity", "verbosity",
)
outputs = {
    "train": Path("data/helpsteer2-train.jsonl"),
    "validation": Path("data/helpsteer2-validation.jsonl"),
}

for split, path in outputs.items():
    dataset = load_dataset(dataset_id, revision=revision, split=split)
    missing = set(fields) - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split} is missing columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in dataset:
            item = {field: row[field] for field in fields}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"{split} is empty")
    print(f"{split}: {count} rows -> {path}")
PY
```

`validation` 是锁定的 final-test split；只供可信 runner 做离线汇总，不能提供给 Decision LLM 或候选。生成完整流 manifest：

```bash
"$TRAIN_PY" scripts/prepare_long_horizon_task.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --output-dir "$LEGO_ROOT/runs/data-audit" \
  --stream-seed 17 --groups-per-batch 50 --feedback-delay 2
```

确认 development 数据可用且没有重复 prompt group，保存 manifest SHA256。数据审计通过后继续配置 Task Model 并运行固定流水线 smoke；T1 候选隔离 smoke 按本文件第 6 节运行。

## 3. 配置 Task Model

```bash
cp code/technologies/technology_one/task_model.qwen.example.json \
  "$LEGO_ROOT/server-private/task-model-smoke.json"
```

编辑 JSON：填写本机 Qwen3-8B 权重绝对路径和不可变 revision；smoke 的 `max_steps` 设为 `1`。GPU 编号按 `nvidia-smi` 实际空闲情况填写：

```bash
export CUDA_VISIBLE_DEVICES=<stable使用的GPU编号>
```

## 4. 跑固定流水线 smoke

这一步只验证真实数据、Task Model 预测、延迟反馈释放和 parent-checkpoint 续训，不包含 T1 候选分支验证。T1 stable/candidate 配对 smoke 在本文件第 6 节运行，默认使用 `local_shadow`，不要求 Kubernetes。

```bash
"$TRAIN_PY" scripts/run_long_horizon_qwen.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --config-json "$LEGO_ROOT/server-private/task-model-smoke.json" \
  --output-dir "$LEGO_ROOT/runs/baseline-smoke/b0-seed-17" \
  --run-id b0-seed-17 --seed 17 --stream-seed 17 \
  --groups-per-batch 50 --feedback-delay 2 --max-batches 4
```

通过条件：`status=complete`、`completed_batches=4`，第 4 批使用到期反馈训练，`final_test_used=false`。失败就停止排查。此步骤只验证固定流水线，不是 T1 候选隔离测试。

## 5. 配置 Decision LLM 并做 API smoke

复制 `code/technologies/technology_one/decision_model.qwen.example.json` 到 `server-private/decision-model.json`，填写服务商确认的 API model ID。然后设置真实 endpoint 和密钥：

```bash
export LEGO_POLICY_API_URL="<OpenAI兼容Chat Completions地址>"
export LEGO_POLICY_API_KEY="<服务器密钥管理器注入的密钥>"
export LEGO_POLICY_TIMEOUT_SECONDS=600

"$TRAIN_PY" scripts/smoke_decision_endpoint.py \
  --profile "$LEGO_ROOT/server-private/decision-model.json"
```

通过条件：`status=ok`、`schema_valid=true`。密钥不要写进 JSON 或提交 Git。

## 6. 技术一候选隔离 smoke

本节直接运行技术一 clean smoke。使用 `local_shadow`，只执行受限数据策略和固定可信 runner，不运行任意候选 Python/shell，也不提供容器、网络或操作系统资源隔离。无需 Docker、Kubernetes 或 kubectl。

生成 candidate 和本次运行目录：

```bash
mkdir -p "$LEGO_ROOT/server-private/t1-candidates"

T1_RUN_ID="t1-local-shadow-smoke-$(date +%Y%m%d-%H%M%S)"
T1_CANDIDATE_DIR="$LEGO_ROOT/server-private/t1-candidates/$T1_RUN_ID"
T1_CANDIDATE_CONFIG="$LEGO_ROOT/server-private/$T1_RUN_ID.json"
T1_OUTPUT="$LEGO_ROOT/runs/t1-smoke/$T1_RUN_ID"

"$TRAIN_PY" scripts/create_t1_smoke_candidate.py \
  --source-dir "$LEGO_ROOT/code/technologies/technology_one" \
  --output-dir "$T1_CANDIDATE_DIR" \
  --candidate-id "$T1_RUN_ID"

cp code/technologies/technology_one/candidate_shadow.example.json "$T1_CANDIDATE_CONFIG"
```

编辑 `$T1_CANDIDATE_CONFIG`，将 `candidate_id` 设为 `$T1_RUN_ID`，`sandbox_backend` 保持 `local_shadow`，`candidate_dir` 设为 `./t1-candidates/$T1_RUN_ID`。不要添加 `image` 或 `kubernetes` 字段。确认 `server-private/task-model-smoke.json` 中的模型路径、revision 和 `max_steps=1` 正确。

运行预检和 4 批 smoke：

```bash
T1_ARGS=(
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl"
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl"
  --task-config-json "$LEGO_ROOT/server-private/task-model-smoke.json"
  --candidate-config-json "$T1_CANDIDATE_CONFIG"
  --output-dir "$T1_OUTPUT"
  --run-id "$T1_RUN_ID"
  --seed 17
  --stream-seed 17
  --groups-per-batch 50
  --feedback-delay 2
  --max-batches 4
)

"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py "${T1_ARGS[@]}" --preflight-only && \
"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py "${T1_ARGS[@]}"
```

预检必须返回 `status=preflight_ok` 和 `sandbox_backend=local_shadow`。预检失败即停止并排查，不切换 backend。

运行结束后检查 stable summary：

```bash
"$TRAIN_PY" - "$T1_OUTPUT/stable/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
candidates = s.get("candidate_history", [])
batches = s.get("batch_history", [])
checks = {
    "stable_complete_4_of_4": s.get("status") == "complete" and s.get("completed_batches") == 4 and s.get("target_batches") == 4,
    "candidate_complete_4_of_4": [x.get("batch_index") for x in candidates] == [0, 1, 2, 3] and all(x.get("status") == "eligible_for_observation" for x in candidates),
    "candidate_isolated": len(candidates) == 4 and all(x.get("gate", {}).get("evidence", {}).get("sandbox_backend") == "local_shadow" and x.get("gate", {}).get("evidence", {}).get("candidate_output_isolated") is True and x.get("gate", {}).get("evidence", {}).get("stable_parent_unchanged") is True for x in candidates),
    "delayed_feedback_released": any(x.get("released_feedback_ids") for x in batches),
    "candidate_checkpoint_retained": candidates[-1].get("cost", {}).get("training_steps", 0) > 0 and candidates[-1].get("candidate_checkpoint_retained") is True,
    "held_out_not_used": s.get("held_out_used") is False,
    "final_test_not_used": s.get("final_test_used") is False,
}
for name, passed in checks.items(): print(f"{name}: {'PASS' if passed else 'FAIL'}")
if not all(checks.values()): raise SystemExit("T1 smoke failed; stop and inspect run artifacts")
print("T1 local-shadow smoke: PASS")
PY
```

全部为 `PASS` 后继续第 7 节。失败时保留本次 run 目录，根据 `candidate-shadow` 下的 `failure_code` 和 `gate.evidence` 排查；修复后使用新的 run ID 和输出目录。

## 7. 技术二 smoke

技术一 clean smoke 通过后，运行技术二四个经验臂。技术二不需要 Decision LLM、Docker 或 Kubernetes。执行参数固定为：`seed=17`、`stream_seed=17`、`groups_per_batch=50`、`feedback_delay=2`、`max_batches=4`，输出到 `runs/technology-two-smoke/<arm>/seed-17/`。

复制并填写技术二配置：

```bash
cp code/technologies/technology_two/task_model.qwen.smoke.example.json \
  "$LEGO_ROOT/server-private/t2-task-model-smoke.json"
```

编辑该 JSON，将 `base_model_path` 和 `base_model_revision` 改成服务器已有的不可变模型快照，并保持 smoke 的 `max_steps=1`。然后运行四臂：

```bash
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

四个实验臂为：`no_experience`、`reflection_only`、`executable_unmanaged`、`lifecycle_governed`。四个目录都必须是 `status=complete` 且 `completed_batches=4`，并确认 `held_out_used=false`、`final_test_used=false`。任何一个失败都停止，不进入 pilot。

技术二使用本手册第 2 节生成的 `helpsteer2-train.jsonl` 和 `helpsteer2-validation.jsonl`。pilot 需另复制 `task_model.qwen.example.json`，填入同一不可变模型快照，并使用正式训练步数配置；不要复用 `max_steps=1` 的 smoke 配置。

## 8. 技术三回放 smoke

技术三使用技术一的完整 source run，不重新加载 Task Model，也不需要 GPU。`T1_ROOT` 必须指向包含完整 `summary.json` 的技术一结果根目录：

```bash
export T1_ROOT="$LEGO_ROOT/runs/technology-one-pilot"
export T3_ROOT="$LEGO_ROOT/runs/technology-three-smoke"
mkdir -p "$T3_ROOT"
```

至少需要 6 个来源批次，才能评价 3 批决定窗口和 3 批未来结果窗口；只有 4 个来源批次时只能检查回放链路，不能报告技术三效果。确认 `$T1_ROOT/seed-17/summary.json` 是 `status=complete` 的技术一 stable/candidate source summary 后，执行：

```bash
"$TRAIN_PY" scripts/run_technology_three_replay.py \
  --t1-summary "$T1_ROOT/seed-17/summary.json" \
  --output-dir "$T3_ROOT/seed-17" \
  --window-batches 3 --outcome-window-batches 3 \
  --min-mean-gain 0.0 --max-regression 0.05
```

输出必须包含 `feedback-trace.json` 和 `summary.json`。该回放不训练模型、不晋升版本，也不证明在线回退能力。

## 9. 技术三多 seed pilot

技术一完成 seeds `17 23 29` 的长周期 source run 后，逐个 seed 回放并汇总。每个 seed 的输出目录必须独立，不能混用不同 source run。

```bash
export T1_ROOT="$LEGO_ROOT/runs/technology-one-pilot"
export T3_ROOT="$LEGO_ROOT/runs/technology-three-pilot"

for SEED in 17 23 29; do
  "$TRAIN_PY" scripts/run_technology_three_replay.py \
    --t1-summary "$T1_ROOT/seed-$SEED/summary.json" \
    --output-dir "$T3_ROOT/seed-$SEED" \
    --window-batches 3 --outcome-window-batches 3 \
    --min-mean-gain 0.0 --max-regression 0.05
done

"$TRAIN_PY" scripts/summarize_technology_three.py \
  --root "$T3_ROOT" --seeds 17 23 29 \
  --output "$T3_ROOT/technology-three-summary.json"
```

## 10. 跨技术组合 smoke

三个单项 smoke 通过后运行 8 组合 contract smoke：

```bash
"$TRAIN_PY" scripts/run_integration_smoke.py \
  --all-configs --batches 4 \
  --output-dir "$LEGO_ROOT/runs/integration-smoke"

"$TRAIN_PY" scripts/summarize_integration_matrix.py \
  --root "$LEGO_ROOT/runs/integration-smoke" \
  --output "$LEGO_ROOT/runs/integration-smoke/summary.json"
```

该 smoke 使用 fake executor，只验证 8 个配置、候选失败隔离、候选晋升、退化回退和汇总器，不是真实 HelpSteer2/Qwen 方法实验。真实 8 组合实验当前没有可运行的 Task Model 执行器，**不要启动真实组合 pilot**。需要先完成真实组合 runner 并发布新的执行命令。

## 11. 后续 pilot 顺序

Smoke 全部通过后，技术二 pilot 可直接运行；技术一的 10～20 批真实候选 pilot 目前还不能直接启动：当前没有定义可复现的候选观察输入与真实提案生成流程，因此不要把第 6 节的 smoke fixture 当作研究候选。真实 8 组合 pilot 也暂不可执行。

技术二 pilot 使用四个经验臂、四种数据流和 seeds `17 23 29`，每个 run 10 批。先创建正式 Task Model 配置：

```bash
cp code/technologies/technology_two/task_model.qwen.example.json \
  "$LEGO_ROOT/server-private/t2-task-model-pilot.json"
```

编辑配置中的模型路径与不可变 revision，确认训练参数已冻结，然后运行配对矩阵并汇总：

```bash
export TRAIN_JSONL="$LEGO_ROOT/data/helpsteer2-train.jsonl"
export VALIDATION_JSONL="$LEGO_ROOT/data/helpsteer2-validation.jsonl"
export TASK_CONFIG="$LEGO_ROOT/server-private/t2-task-model-pilot.json"
export T2_ROOT="$LEGO_ROOT/runs/technology-two-pilot"

for REGIME in randomized interleaved abrupt alternating; do
  for SEED in 17 23 29; do
    for ARM in no_experience reflection_only executable_unmanaged lifecycle_governed; do
      "$TRAIN_PY" scripts/run_technology_two_qwen.py \
        --train-jsonl "$TRAIN_JSONL" \
        --validation-jsonl "$VALIDATION_JSONL" \
        --task-config-json "$TASK_CONFIG" \
        --output-dir "$T2_ROOT/$REGIME/$ARM/seed-$SEED" \
        --run-id "t2-$REGIME-$ARM-seed-$SEED" \
        --arm "$ARM" --seed "$SEED" --stream-seed "$SEED" \
        --stream-regime "$REGIME" --groups-per-batch 50 \
        --feedback-delay 2 --max-batches 10
    done
  done
done

for REGIME in randomized interleaved abrupt alternating; do
  "$TRAIN_PY" scripts/summarize_technology_two.py \
    --root "$T2_ROOT/$REGIME" --seeds 17 23 29 --expected-batches 10 \
    --output "$T2_ROOT/$REGIME/summary.json"
done
```

确认所有 48 个 run 均为 `status=complete`、`completed_batches=10`，且 `held_out_used=false`、`final_test_used=false`。每种数据流单独汇总，不跨数据流合并。技术三按本文件第 9 节对至少 6 批的技术一来源运行离线回放；该结果只验证归因和版本选择逻辑，不代表在线部署验证。

每个 run 使用新的输出目录和唯一 `run_id`；失败目录保留，不覆盖重跑。

运行日志、manifest、summary、状态文件和配置哈希保存在 `runs/`；不要把数据、模型、checkpoint、密钥或私有配置提交到 Git。
