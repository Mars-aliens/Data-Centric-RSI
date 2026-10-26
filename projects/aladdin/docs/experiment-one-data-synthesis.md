# 实验一：补充运行指令

以下命令在已配置好的 Linux Aladdin 项目环境中执行。原 pilot 已完成 seed `17、23、29`；本次只补跑 seed `37、41、43、47、53`，共 15 个单元。完成后与 pilot 合并，得到 24 个单元的统一结果。

## 1. 设置结果路径

```bash
set -euo pipefail

export PILOT_RESULT="$ALADDIN_ROOT/runs/experiment-one-data-synthesis-prism-20261026-024147/result.json"
export RUN="$ALADDIN_ROOT/runs/experiment-one-data-synthesis-prism-extension-$(date +%Y%m%d-%H%M%S)"
export COMBINED_RUN="$ALADDIN_ROOT/runs/experiment-one-data-synthesis-prism-combined-$(date +%Y%m%d-%H%M%S)"

test -f "$PILOT_RESULT"
mkdir -p "$RUN" "$COMBINED_RUN"
```

如果 pilot 结果不在上述目录，只修改 `PILOT_RESULT`，指向原来的 `result.json`。

## 2. 补跑剩余 15 个单元

```bash
"$PYTHON_BIN" -u -m experiments.experiment_one_synthesis \
  --seed-data "$SEED_DATA" \
  --development-data "$DEV_DATA" \
  --development-labels "$DEV_LABELS" \
  --output "$RUN" \
  --base-model "$BASE_MODEL" \
  --model-revision "$MODEL_REVISION" \
  --endpoint "$ALADDIN_ENDPOINT" \
  --model "$ALADDIN_MODEL" \
  --strategies seed-only fixed-prompt self-instruct \
  --seeds 37 41 43 47 53 \
  --seed-group-limit 32 \
  --development-group-limit 64 \
  --development-selection-seed 101 \
  --generation-token-budget 24000 \
  --max-candidates-per-strategy 512 \
  --max-tokens 512 \
  --temperature 0.7 \
  --train-records 256 \
  --synthetic-train-records 128 \
  --train-token-budget 100000 \
  --max-steps 1000 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --eval-batch-size 8 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  2>&1 | tee "$RUN/run.log"
```

## 3. 合并并分析 24 个单元

```bash
"$PYTHON_BIN" - "$PILOT_RESULT" "$RUN/result.json" "$COMBINED_RUN/result.json" <<'PY'
import copy
import json
import sys
from datetime import datetime, timezone

pilot_path, extension_path, output_path = sys.argv[1:]
pilot = json.load(open(pilot_path, encoding="utf-8"))
extension = json.load(open(extension_path, encoding="utf-8"))

assert pilot["status"] == extension["status"] == "completed"
assert pilot["protocol"] == extension["protocol"]
assert pilot["data"]["final_test_accessed"] is False
assert extension["data"]["final_test_accessed"] is False

def comparable_config(result):
    config = copy.deepcopy(result["configuration"])
    config.pop("output", None)
    config.pop("seeds", None)
    return config

assert comparable_config(pilot) == comparable_config(extension), (
    "运行配置不一致，停止合并"
)
for key in (
    "seed_sha256",
    "split_manifest_sha256",
    "development_sha256",
    "development_labels_sha256",
    "dataset_revision",
    "development_groups_sha256",
):
    assert pilot["data"].get(key) == extension["data"].get(key), (
        f"数据不一致: {key}"
    )
for key in ("runner_sha256", "training_module_sha256", "synthesis_module_sha256"):
    assert pilot["runtime"].get(key) == extension["runtime"].get(key), (
        f"代码版本不一致: {key}"
    )

results = pilot["results"] + extension["results"]
keys = [(row["strategy"], row["seed"]) for row in results]
seeds = [17, 23, 29, 37, 41, 43, 47, 53]
strategies = ["seed-only", "fixed-prompt", "self-instruct"]
expected = {(strategy, seed) for strategy in strategies for seed in seeds}
assert len(results) == 24 and len(set(keys)) == 24 and set(keys) == expected

combined = copy.deepcopy(pilot)
combined["run_id"] = "pilot-plus-extension"
combined["configuration"]["seeds"] = seeds
combined["configuration"]["output"] = output_path.rsplit("/", 1)[0]
combined["data"]["seed_groups_selected_by_run_seed"].update(
    extension["data"]["seed_groups_selected_by_run_seed"]
)
combined["results"] = results
combined["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
    f.write("\n")
print("合并完成:", len(results), "/ 24")
PY

"$PYTHON_BIN" -m scripts.analyze_experiment_one_synthesis "$COMBINED_RUN/result.json" \
  | tee "$COMBINED_RUN/summary.txt"
```
