# 技术一服务器执行步骤

沿用服务器现有 Python 环境、数据和模型路径。此流程使用 `local_shadow`：Decision LLM 提出的候选只能改受限的数据策略 JSON，由固定可信 runner 执行；候选在独立输出目录和 parent-checkpoint 副本上运行。此机制检查候选分支是否污染 stable，不运行任意候选 Python/shell 代码，也不提供操作系统级资源或网络隔离。

完成 `4/4` 后检查结果并停止。

## 1. 生成 clean smoke candidate

```bash
cd "$LEGO_ROOT"
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

编辑 `$T1_CANDIDATE_CONFIG`，设置以下字段。`candidate_dir` 是相对于该 JSON 文件的路径；其余内容沿用 example：

```json
{
  "candidate_id": "替换为本次 T1_RUN_ID",
  "sandbox_backend": "local_shadow",
  "candidate_dir": "./t1-candidates/替换为本次 T1_RUN_ID"
}
```

配置中的 `candidate_id` 必须与 `create_t1_smoke_candidate.py` 使用的 ID 相同。不要添加 `image` 或 `kubernetes` 字段。确认 Task Model smoke 配置仍指向现有模型路径和固定 revision。

## 2. 预检并运行 4/4

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

"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py \
  "${T1_ARGS[@]}" --preflight-only &&
"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py \
  "${T1_ARGS[@]}"
```

预检输出必须包含 `status=preflight_ok` 和 `sandbox_backend=local_shadow`。预检失败就停止并处理报错，不改用另一 backend。

## 3. 检查结果并停止

```bash
"$TRAIN_PY" - "$T1_OUTPUT/stable/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)

candidate = summary.get("candidate_history", [])
batches = summary.get("batch_history", [])
last_candidate = next(
    (item for item in candidate if item.get("batch_index") == 3), {}
)
checks = {
    "stable_complete_4_of_4": (
        summary.get("status") == "complete"
        and summary.get("completed_batches") == 4
        and summary.get("target_batches") == 4
    ),
    "candidate_complete_4_of_4": (
        [item.get("batch_index") for item in candidate] == [0, 1, 2, 3]
        and all(item.get("status") == "eligible_for_observation" for item in candidate)
    ),
    "candidate_branch_and_stable_unchanged": len(candidate) == 4 and all(
        item.get("gate", {}).get("evidence", {}).get("sandbox_backend") == "local_shadow"
        and item.get("gate", {}).get("evidence", {}).get("candidate_output_isolated") is True
        and item.get("gate", {}).get("evidence", {}).get("stable_parent_unchanged") is True
        for item in candidate
    ),
    "delayed_feedback_released": any(
        batch.get("released_feedback_ids") for batch in batches
    ),
    "candidate_trained_and_checkpoint_retained": (
        last_candidate.get("cost", {}).get("training_steps", 0) > 0
        and last_candidate.get("candidate_checkpoint_retained") is True
    ),
    "held_out_not_used": summary.get("held_out_used") is False,
    "final_test_not_used": summary.get("final_test_used") is False,
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
if not all(checks.values()):
    raise SystemExit("T1 local-shadow smoke failed; stop and inspect the run artifacts.")
print("T1 local-shadow smoke: PASS")
PY
```

所有条件通过后，本阶段完成并停止。若失败，检查 `$T1_OUTPUT/stable/candidate-shadow/` 下对应批次的 `failure_code`、错误和 `gate.evidence`，根据实际错误修复后再运行新的 run ID。

`local_shadow` 的证据只支持“受限策略由可信 runner 在独立候选分支执行，stable parent 未被改写”。结果中的 `container_isolation=false` 是预期值；不得把本次结果描述为容器隔离、任意代码安全、网络隔离或 CPU/内存硬限制验证。Kubernetes backend 是可选增强，当前 T1 smoke 不要求 Kubernetes、kubectl 或 Docker。
