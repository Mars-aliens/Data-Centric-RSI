# Aladdin 服务器配置步骤

在 Linux 服务器上按顺序执行。将 `<...>` 替换成服务器实际值。不要上传数据、模型、密钥和运行结果。

## 1. 获取代码并安装环境

```bash
set -euo pipefail

git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/aladdin

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

按服务器 CUDA 版本从 PyTorch 官方选择器安装对应 wheel：

```bash
python -m pip install torch==<PyTorch版本> --index-url <PyTorch CUDA wheel源>
python -m pip install -r requirements-qwen.txt
python -m pip check
python -m pytest -q
python -m compileall -q experiments scripts tests
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
nvidia-smi
```

设置项目路径：

```bash
export ALADDIN_ROOT="$PWD"
export PYTHON_BIN="$ALADDIN_ROOT/.venv/bin/python"
export PYTHONPATH="$ALADDIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ALADDIN_ROOT/data" "$ALADDIN_ROOT/runs" "$ALADDIN_ROOT/server-private"
```

需要核对论文 baseline 的上游代码时执行以下命令。它们固定到指定 commit；实验一数据合成实际使用项目内适配器，不依赖这些 checkout。

```bash
bash scripts/fetch_synthesis_baselines.sh
test "$(git -C baselines/synthesis/self-instruct-upstream rev-parse HEAD)" = "0b26ccaa415992100fa32df62d41b994cf928e23"
test "$(git -C baselines/synthesis/wizardlm-upstream rev-parse HEAD)" = "cd4b4cc022b6426aee62d51f7ae99a4a6ec9aaef"
```

## 2. 下载并准备数据集

实验一数据合成使用 PRISM。固定版本为 `HannahRoseKirk/prism-alignment@18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`，下载 `conversations.jsonl`、`utterances.jsonl`、`metadata.jsonl`，不下载 `survey.jsonl`。数据包含参与者文本和模型回答，许可分别为 CC-BY-4.0 与 CC-BY-NC-4.0；只用于符合条款的研究，不尝试识别参与者，不把数据文件上传到 GitHub。下载前确认你接受这些限制。

在项目根目录运行以下命令。数据下载到 `data/raw/prism/`，处理结果写入 `data/processed/prism/`：

```bash
"$PYTHON_BIN" -m scripts.download_datasets \
  --dataset prism --confirm-research-terms

"$PYTHON_BIN" -m scripts.prepare_prism --split-seed 17

"$PYTHON_BIN" -m scripts.audit_datasets --dataset prism \
  | tee "$ALADDIN_ROOT/runs/prism-audit.json"
```

命令全部成功后，确认 prepare 状态为 `prepared`、审计命令退出码为 `0`，各 split 的 user overlap 均为 `0`。处理结果包含 `seed.jsonl`、`feedback_pool.jsonl`、`development.jsonl`、`final_test.jsonl` 和 `labels/`；数据合成环节使用 seed，development 及其标签只用于离线评价，不读取 final test。

HelpSteer2 不是当前数据合成实验的运行依赖，仅用于后续受控反馈回放。需要下载时，固定版本为 `nvidia/HelpSteer2@990b2711a36180dd19d9c94b8627844866f8982a`，许可为 CC-BY-4.0：

```bash
"$PYTHON_BIN" -m scripts.download_datasets --dataset helpsteer2
"$PYTHON_BIN" -m scripts.prepare_helpsteer2 --split-seed 17
"$PYTHON_BIN" -m scripts.audit_datasets --dataset helpsteer2 \
  | tee "$ALADDIN_ROOT/runs/helpsteer2-audit.json"
```

处理后写入 `data/processed/helpsteer2/`；保留 helpfulness、correctness、coherence、complexity、verbosity 五个评分维度及并列关系。官方 validation 留作 `final_test`，不得用于策略或阈值选择。

## 3. 启动生成模型 API

如果已有此模型 revision 的 OpenAI-compatible API 服务，跳过启动命令。否则在 vLLM 0.8.5 或更新版本的模型服务环境中启动：

```bash
vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --served-model-name Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000
```

在实验终端设置 API 地址。需要鉴权时由密钥管理器注入 `OPENAI_API_KEY`：

```bash
export ALADDIN_ENDPOINT="<OpenAI兼容Chat Completions地址>"
export ALADDIN_MODEL="Qwen/Qwen3-8B"
export MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
# export OPENAI_API_KEY="由密钥管理器注入"
```

API 服务需逐请求返回 token usage。生成模型与下游训练模型应使用同一 revision；建议将 API 服务与训练任务放在不同 GPU。继续执行 [`实验一数据合成运行步骤`](experiment-one-data-synthesis.md)。

## 4. 运行实验一：数据合成

按 [`实验一数据合成运行步骤`](experiment-one-data-synthesis.md) 完成 API 预检、正式运行和结果汇总。
