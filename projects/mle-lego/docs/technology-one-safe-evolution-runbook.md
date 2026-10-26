# 技术一执行手册

同步待验证代码后，沿用服务器现有 Python 环境、数据和模型路径，按以下步骤重跑 T1 smoke。候选 JSON 需兼容本次 K8s 后端，由预检核验。完成 `4/4` 并检查结果后停止。

## 1. 预检并运行 4/4

在已有服务器会话中执行。命令自动生成新的 run ID 和输出目录。

```bash
cd "$LEGO_ROOT"

T1_RUN_ID="t1-k8s-smoke-$(date +%Y%m%d-%H%M%S)"
T1_OUTPUT="$LEGO_ROOT/runs/t1-smoke/$T1_RUN_ID"

T1_ARGS=(
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl"
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl"
  --task-config-json "$LEGO_ROOT/server-private/task-model-smoke.json"
  --candidate-config-json "$LEGO_ROOT/server-private/t1-clean-smoke.json"
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

预检通过才自动启动 4/4。确认输出为 `preflight_ok`、`sandbox_backend=kubernetes`，且模型 revision、路径和 GPU 请求数量正确。预检失败就停止排查，不能回退 local。

命令完成后，在同一 shell 执行第 2 节检查。

## 2. 检查并停止

检查 `$T1_OUTPUT/stable/summary.json`。以下命令逐项核验 stable 完成 4 批、candidate 每批均通过隔离门、延迟反馈已释放、candidate 用反馈训练并保留自己的 checkpoint，且没有使用 held-out/final-test：

```bash
"$TRAIN_PY" - "$T1_OUTPUT/stable/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)

candidate = summary.get("candidate_history", [])
batches = summary.get("batch_history", [])
candidate_indexes = sorted(item.get("batch_index") for item in candidate)
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
        candidate_indexes == [0, 1, 2, 3]
        and all(item.get("status") == "eligible_for_observation" for item in candidate)
    ),
    "kubernetes_isolation_4_of_4": len(candidate) == 4 and all(
        item.get("gate", {}).get("evidence", {}).get("sandbox_backend") == "kubernetes"
        and item.get("gate", {}).get("evidence", {}).get("container_isolation") is True
        and item.get("gate", {}).get("evidence", {}).get("stable_parent_unchanged") is True
        and item.get("gate", {}).get("evidence", {}).get("isolation_evidence", {}).get("output_retrieved") is True
        and item.get("gate", {}).get("evidence", {}).get("isolation_evidence", {}).get("network_policy_enforcement_verified") is True
        and item.get("gate", {}).get("evidence", {}).get("isolation_evidence", {}).get("pod_uid")
        and item.get("gate", {}).get("evidence", {}).get("isolation_evidence", {}).get("cleanup_errors") == []
        for item in candidate
    ),
    "delayed_feedback_released": any(
        batch.get("released_feedback_ids") for batch in batches
    ),
    "candidate_trained_and_retained_checkpoint": (
        last_candidate.get("cost", {}).get("training_steps", 0) > 0
        and last_candidate.get("candidate_checkpoint_retained") is True
    ),
    "held_out_not_used": summary.get("held_out_used") is False,
    "final_test_not_used": summary.get("final_test_used") is False,
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
if not all(checks.values()):
    raise SystemExit("T1 clean smoke verification failed; stop and inspect the run artifacts.")
print("T1 clean smoke verification: PASS")
PY
```

以上条件全部满足后，本阶段完成。

失败时先读取 candidate-shadow/trials 下当前批次的 failure_code、error 和 gate.evidence，再检查对应 Job/Pod 事件。未被自动清理的资源名称保存在 isolation_evidence.cleanup_errors 中；请管理员处理后再运行。ConfigMap 请求上限为 700 KiB，超过会拒绝；本阶段不增加 batch size 或反馈载荷。
