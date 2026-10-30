# Aladdin 实验运行命令

所有命令在 `projects/aladdin/` 目录执行。

## 环境、数据和模型服务

执行 [server-quickstart.md](server-quickstart.md) 第 1 至第 4 步。

## 小规模 pilot

```bash
.venv/bin/python -m scripts.check_model_services \
  --config configs/feedback-rsi.local.json

.venv/bin/python -m experiments.rsi \
  --config configs/feedback-rsi.local.json

.venv/bin/python -m scripts.analyze_feedback_rsi \
  runs/feedback-rsi-pilot/result.json \
  --output runs/feedback-rsi-pilot/analysis.json
```

配置文件 `configs/feedback-rsi-pilot.example.json` 的默认规模：1 个 seed、2 轮、每轮每组 4 个候选、2 条 Qwen3-32B 模拟反馈；`comparison` 保持 `overall`。

检查：

```bash
.venv/bin/python - "runs/feedback-rsi-pilot/result.json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("feedback_model:", data.get("configuration", {}).get("feedback_model"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    print(name, "rounds:", len(arm.get("rounds", [])))
PY
```

## 中断恢复

```bash
.venv/bin/python -m experiments.rsi \
  --config configs/feedback-rsi.local.json \
  --resume
```

## pilot 结束

确认 `result.json` 的 `status` 为 `completed` 后，查看 `analysis.json` 和日志，再决定是否扩大实验。
