# 技术一执行手册

先完成[服务器快速开始](server-quickstart.md)第 1～5 步。以下命令均在 `projects/mle-lego` 目录执行。本手册只执行到 T1 clean smoke 的 `4/4`，完成后停止。

## 1. 构建 clean fixture

```bash
export PYTORCH_BASE_IMAGE="<CUDA PyTorch镜像>@sha256:<64位小写digest>"
export T1_CANDIDATE_IMAGE_TAG="mle-lego-t1-candidate:locked"
bash scripts/build_t1_candidate_image.sh

"$TRAIN_PY" scripts/create_t1_smoke_candidate.py \
  --source-dir "$LEGO_ROOT/code" \
  --output-dir "$LEGO_ROOT/runs/t1-candidates/t1-clean-smoke" \
  --candidate-id t1-clean-smoke
```

复制 `code/technologies/technology_one/candidate_shadow.example.json`：

```bash
cp code/technologies/technology_one/candidate_shadow.example.json \
  "$LEGO_ROOT/server-private/t1-clean-smoke.json"
```

修改 `server-private/t1-clean-smoke.json`：

```text
candidate_id = t1-clean-smoke
candidate_dir = 填写该目录在服务器上的实际绝对路径（不要原样填写 $LEGO_ROOT）
image = 上一步输出的不可变 image_id
limits.gpu_devices = 与 stable 不同的空闲 GPU 或 MIG
require_proposal_manifest = true
```

确认 `server-private/task-model-smoke.json` 中的模型路径、不可变 revision 和 `max_steps=1` 正确。stable、candidate 必须使用不同 GPU/MIG。

## 2. 预检并运行 4/4

```bash
export CUDA_VISIBLE_DEVICES="<stable物理GPU编号>"
export LEGO_TASK_MODEL_DEVICE=cuda
export T1_RUN_ID="t1-clean-seed-17"
export T1_OUTPUT="$LEGO_ROOT/runs/t1-smoke/$T1_RUN_ID"

"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --task-config-json "$LEGO_ROOT/server-private/task-model-smoke.json" \
  --candidate-config-json "$LEGO_ROOT/server-private/t1-clean-smoke.json" \
  --output-dir "$T1_OUTPUT" --run-id "$T1_RUN_ID" \
  --seed 17 --stream-seed 17 --groups-per-batch 50 --feedback-delay 2 \
  --max-batches 4 --preflight-only
```

确认输出为 `preflight_ok`，且镜像、模型 revision、数据路径和 GPU 分配正确。

重新运行同一命令，删除最后的 `--preflight-only`：

```bash
"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --task-config-json "$LEGO_ROOT/server-private/task-model-smoke.json" \
  --candidate-config-json "$LEGO_ROOT/server-private/t1-clean-smoke.json" \
  --output-dir "$T1_OUTPUT" --run-id "$T1_RUN_ID" \
  --seed 17 --stream-seed 17 --groups-per-batch 50 --feedback-delay 2 \
  --max-batches 4
```

## 3. 检查并停止

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
