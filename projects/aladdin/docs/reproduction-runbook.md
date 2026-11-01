# Aladdin 实验运行命令

所有命令在 `projects/aladdin/` 目录执行。

## 环境、数据和模型服务

执行 [server-quickstart.md](server-quickstart.md) 第 1 至第 4 步。

## 实验一：单 seed 完整配置 rehearsal

```bash
.venv/bin/python -m scripts.check_model_services \
  --config configs/experiment-one-single-seed.local.json

.venv/bin/python -m experiments.rsi \
  --config configs/experiment-one-single-seed.local.json

.venv/bin/python -m scripts.analyze_feedback_rsi \
  runs/experiment-one-single-seed/result.json \
  --output runs/experiment-one-single-seed/analysis.json
```

配置文件 `configs/experiment-one-single-seed.example.json` 使用实验一的 `overall` 对照，仅设置一个 seed（17）。它包含 `fixed_policy` 和 `recursive_policy` 两个实验组、3 轮、每轮每组 32 个候选和 16 条 Qwen3-32B 模拟反馈；每个 adapter 的训练 token 预算为 100000，`max_steps=1000` 是安全上限。此运行验证实验一的完整配置能否执行，不是多 seed 确认性结果，也不运行 final-test。

检查：

```bash
.venv/bin/python - "runs/experiment-one-single-seed/result.json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("status:", data.get("status"))
print("comparison:", data.get("comparison"))
print("feedback_model:", data.get("configuration", {}).get("feedback_model"))
print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
for name, arm in data.get("arms", {}).items():
    rounds = arm.get("rounds", [])
    print(name, "rounds:", len(rounds), "training:", [row["training"].get("status") for row in rounds])
PY
```

完成条件：`status` 为 `completed`、`comparison` 为 `overall`、结果含 2 个 arm 且每个 arm 完成 3 轮、所有训练状态为 `completed`、每轮每组完成 16 条模型反馈，并且 `final_test_accessed` 为 `false`。

## 中断恢复

```bash
.venv/bin/python -m experiments.rsi \
  --config configs/experiment-one-single-seed.local.json \
  --resume
```

## rehearsal 结束

确认两个组各 3 轮全部完成后，检查训练预算、反馈尝试日志和递归组策略版本变化。实验一 rehearsal 通过后，再运行实验二消融和实验三反馈选择；正式实验沿用冻结配置，只把 seeds 替换成预先确定的多 seed 列表。final-test 另行执行。
