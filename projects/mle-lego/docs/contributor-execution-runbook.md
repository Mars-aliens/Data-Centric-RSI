# 贡献者实验执行手册

本手册说明贡献者如何从 GitHub 仓库执行技术一实验并提交可审计证据。除非 Issue 明确授权修订协议，否则不得更改协议、随机种子、模型版本、接受规则或数据切分。

## 角色

- **维护者：** 负责研究问题、协议、GitHub Issue 和结果解释。
- **贡献者：** 准备服务器、运行指定命令、保存日志与输出，并准确报告结果。
- **审查者：** 核对 commit、配置哈希、数据清单和审计结果，并确认运行是否符合协议、可进行比较。

## 1. 从 GitHub Issue 开始

每次运行都必须对应一个 Issue，包含：

- experiment name and purpose;
- Git commit or tag to execute;
- profile template and private profile revision;
- mode, seeds, batch count, GPU assignment, and output root;
- required smoke test and formal command;
- expected result files and acceptance criteria.

贡献者须在运行报告中记录 Issue 编号。不得从未提交的工作树开展正式实验。

## 2. 准备服务器

克隆或更新仓库，然后记录代码版本：

```bash
git clone <repository-url>
cd <repository-directory>
git checkout <commit-or-tag>
git rev-parse HEAD
```

在 Git 仓库之外准备私有运行环境：

```bash
export LEGO_ROOT="$PWD"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"
export LEGO_POLICY_API_URL="http://127.0.0.1:8000/v1/chat/completions"
export LEGO_POLICY_TIMEOUT_SECONDS=600
export CUDA_VISIBLE_DEVICES=<gpu-id>
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=40
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

将示例配置复制到私有路径，填写本地模型和命令路径，并确保私有配置不进入 Git。启动运行前，校验归一化数据、配置、Decision LLM 端点和 Task Model。

归一化文件属于外部输入；当前安装检查脚本要求它们位于检出目录中以下已忽略路径：

```text
$LEGO_ROOT/data/helpsteer2-train.jsonl
$LEGO_ROOT/data/helpsteer2-validation.jsonl
```

不要提交这些文件。执行前在 Issue 中核验其来源版本和 SHA256。

运行预检：

```bash
mkdir -p "$LEGO_ROOT/server"
"$TRAIN_PY" -m local_experiments.lego.server_preflight \
  --profile "$LEGO_ROOT/server_profile.evolution.qwen.json" \
  --root "$LEGO_ROOT" \
  --report "$LEGO_ROOT/server/preflight.json"

bash "$LEGO_ROOT/scripts/check_lego_install.sh"
```

Decision LLM 服务必须单独测试：

```bash
POLICY_API_ROOT="${LEGO_POLICY_API_URL%/chat/completions}"
curl -fsS --max-time 30 "$POLICY_API_ROOT/models"
"$TRAIN_PY" -m local_experiments.lego.policy_command <<'JSON'
{
  "protocol": "lego-harness-edit-v1",
  "model": {"id": "<pinned-decision-model>", "revision": "<revision>"},
  "prompt": {
    "system": "Return JSON only.",
    "user": "Return a minimal valid proposal with base_harness_version v1, empty edits, and rationale smoke test."
  }
}
JSON
```

仅当服务器上的 vLLM 环境为已配置的 `vllm-serve` 环境时，才使用仓库的 `scripts/check_decision_model.sh`；否则在 Issue 中记录等效的 tokenizer、Transformers、vLLM 和 CUDA 检查结果。

## 3. 冒烟测试

先运行指定的冒烟命令。冒烟运行只检查链路，不构成论文实验依据。确认：

- `/v1/models` exposes the pinned Decision LLM model;
- one policy command returns valid JSON;
- the Task Model and evaluator can load;
- `final_test_used` is `false`;
- the run writes `progress.json`, `summary.json`, and an audit trail.

如果冒烟测试失败，应停止并报告命令、退出状态、相关日志末尾、GPU 分配及配置/模型版本。不得悄悄修改协议以使运行通过。

冒烟命令：

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/<experiment-id>-smoke" \
GPU_ID=<gpu-id> BATCHES=2 SEEDS="17" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

## 4. 正式实验

使用 Issue 指定的命令。技术一正式比较的模式为 `independent` 和 `dependency-aware-joint`：

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/<experiment-id>" \
GPU_ID=<gpu-id> \
BATCHES=<batches> \
SEEDS="17 23 29 31 37 41 43 47 53 59" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

只有在 GPU 和显存相互隔离时才并行运行不同模式，否则串行运行。配置、代码版本、模型版本、随机种子集合或预算发生变化时，必须使用新的输出目录。

运行期间监控 `progress.json` 和日志。拒绝率高本身不代表实验失败；应按 `held_in_regression`、`duplicate_configuration`、提案无效和执行失败分类报告。

## 5. 必需的证据包

实验完成后，在源码仓库之外提交压缩证据包，包含：

- `summary.json` for every completed run;
- `progress.json` and `endurance_state.json`;
- run logs and the final audit output;
- code commit, profile hash, protocol hash, data manifest/hash;
- model IDs and revisions for Task Model and Decision LLM;
- GPU, seed, batch, training, evaluator, and retry settings;
- a short `run-report.md` using the template below.

不要将原始数据集、模型权重、检查点、凭据、日志或 `runs/` 提交到公开仓库。大型证据应上传至约定的产物存储，并只在 Issue 中记录链接、校验和及摘要。

## 6. 运行报告模板

完整报告字段请使用[运行报告模板](run-report-template.md)。

贡献者只报告事实，由维护者判断结果是否支持研究结论。

## 7. 完成与审查

只有在所有指定摘要均已生成、审计通过、`final_test_used` 符合协议且 Issue 已链接证据包后，运行才算完成。随后由维护者通过独立文档提交更新 `docs/current-status.md` 和实验摘要。

如果运行崩溃，保留失败输出和日志。若支持恢复，只能在代码、配置、数据、模型、随机种子和输出目录完全相同的条件下恢复；否则使用新输出目录，并在 Issue 中说明原因。
