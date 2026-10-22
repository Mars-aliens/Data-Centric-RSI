# 技术一服务器执行步骤

按顺序执行。所有 `<...>` 都要替换成当前服务器的实际值。模型、数据、密钥和运行结果不要上传 GitHub。

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
python -m pip install -r requirements-server.txt
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

下载 HelpSteer2 `nvidia/HelpSteer2` revision `990b2711a36180dd19d9c94b8627844866f8982a`，导出字段：`prompt`、`response`、`helpfulness`、`correctness`、`coherence`、`complexity`、`verbosity`。保存为：

```text
$LEGO_ROOT/data/helpsteer2-train.jsonl
$LEGO_ROOT/data/helpsteer2-validation.jsonl
```

`validation` 是锁定的 final-test，不要提供给模型或候选。生成完整流 manifest：

```bash
"$TRAIN_PY" scripts/prepare_long_horizon_task.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --output-dir "$LEGO_ROOT/runs/data-audit" \
  --stream-seed 17 --groups-per-batch 50 --feedback-delay 2
```

确认 development 数据可用且没有重复 prompt group，保存 manifest SHA256。数据审计完成后按执行手册先做 4 周期 smoke。

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

```bash
"$TRAIN_PY" scripts/run_long_horizon_qwen.py \
  --train-jsonl "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
  --validation-jsonl "$LEGO_ROOT/data/helpsteer2-validation.jsonl" \
  --config-json "$LEGO_ROOT/server-private/task-model-smoke.json" \
  --output-dir "$LEGO_ROOT/runs/t1-pilot/b0-seed-17" \
  --run-id b0-seed-17 --seed 17 --stream-seed 17 \
  --groups-per-batch 50 --feedback-delay 2 --max-batches 4
```

通过条件：`status=complete`、`completed_batches=4`，第 4 批使用到期反馈训练，`final_test_used=false`。失败就停止排查。此步骤是部署 smoke，之后再运行 10～20 周期 pilot。

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

## 6. 跑 T1 隔离 pilot

按[技术一安全自进化执行手册](technology-one-safe-evolution-runbook.md)配置 Docker digest、候选 bundle、资源限制和另一张 GPU/MIG。依次完成：

1. `--preflight-only` 预检；
2. 4 批无故障 `T1-shadow-clean` smoke；
3. 独立 smoke run，在 batch 3 注入 `invalid_protocol`；
4. 批次边界中断并恢复。

以上 smoke 全部通过后，按协议配置 10～20 周期 pilot。pilot 结果通过审查后，再决定是否进入正式规模实验。

通过条件：clean 两个分支均完成 4/4；故障候选停止而 stable 完成 4/4 且 checkpoint 未改变；恢复没有重复批次或反馈。每项使用新的 run ID 和输出目录。

## 7. Pilot

4 周期 smoke 全部通过后运行 10～20 周期 pilot。记录稳定分支完成率、候选隔离、恢复结果、模型/数据版本和资源成本。审查 pilot 结果后再确定正式实验设计与运行规模；不要提前启动长周期正式实验。

运行日志、manifest、summary、状态文件和配置哈希保存在 `runs/`；不要把数据、模型、checkpoint、密钥或私有配置提交到 Git。
