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

## 2. 下载并准备 PRISM

先阅读 `docs/datasets.md` 和 `configs/datasets.json`，确认数据使用条件后执行：

```bash
"$PYTHON_BIN" -m scripts.download_datasets \
  --dataset prism --confirm-research-terms

"$PYTHON_BIN" -m scripts.prepare_prism --split-seed 17

"$PYTHON_BIN" -m scripts.audit_datasets --dataset prism \
  | tee "$ALADDIN_ROOT/runs/prism-audit.json"
```

确认 prepare 状态为 `prepared`，审计无错误且 split 重叠数为 `0`。

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

API 服务需逐请求返回 token usage。生成模型与下游训练模型应使用同一 revision；建议将 API 服务与训练任务放在不同 GPU。继续执行 [`实验一运行步骤`](experiment-one-synthesis-baselines.md)。

## 4. 运行实验一 1A

按 [`实验一运行步骤`](experiment-one-synthesis-baselines.md) 完成 API 预检、正式运行和结果汇总。
