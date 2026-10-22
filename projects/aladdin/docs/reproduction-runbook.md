# Aladdin 实验复现手册

本文将当前可执行命令与仍处于设计阶段的实验区分开来，供 Aladdin 协作者复现环境和交接工作。研究决策以 [`feasibility-pilot-protocol.md`](feasibility-pilot-protocol.md) 为准。

面向 Linux 模型服务器的按步骤操作版见 [`server-quickstart.md`](server-quickstart.md)。实验 1A 正式运行包含 Qwen LoRA 下游训练，需要 GPU；后续 1B、清洗、反馈选择和完整 RSI 仍需完成实现。

## GitHub 路径

公开 monorepo 中，本项目位于 `projects/aladdin/`。克隆仓库后进入该目录执行本文命令。

```text
projects/aladdin/
  README.md
  CONTRIBUTING.md
  pyproject.toml
  requirements*.txt
  configs/                 固定数据集来源和使用限制
  docs/                    方案、对照、实现与复现文档
  experiments/synthesis/   合成 baseline 适配器
  experiments/evaluation/  Qwen SFT 和 development 评价
  experiments/experiment_1a.py  实验 1A 正式矩阵 runner
  scripts/                 数据集下载、准备与审计
  tests/                   确定性 CPU 测试
  baselines/synthesis/     上游版本说明；实际 checkout 被忽略
  data/                    下载/处理后的数据；Git 忽略
  runs/                    模型运行输出；Git 忽略
```

不要提交 `.venv*`、`data/`、模型权重、凭据或 `runs/`。

## 1. 环境

需要 Python 3.10 或更高版本。数据处理和 baseline 测试使用 CPU 即可，不需要 GPU。

Windows PowerShell，在 Aladdin 项目目录中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q experiments scripts tests
```

Linux，在 `projects/aladdin/` 中运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q experiments scripts tests
```

`requirements-dev.txt` 会以 editable 模式安装项目，并安装测试、数据下载和分析依赖。`requirements-data.txt` 用于只准备数据的环境；`requirements.txt` 是较小的运行时环境。GPU 训练环境需先按服务器驱动/CUDA 选择 PyTorch wheel，再安装 `requirements-qwen.txt`。运行 manifest 中要记录最终解析出的依赖版本和 CUDA 详情；项目不会猜测所有服务器都适用的 PyTorch wheel。

## 2. 下载、准备并审计数据

下载前阅读 `docs/datasets.md`。数据集内容留在本地，不上传。Linux 命令如下：

```bash
PY=.venv/bin/python
$PY -m scripts.download_datasets --dataset all --confirm-research-terms
$PY -m scripts.prepare_prism --split-seed 17
$PY -m scripts.prepare_helpsteer2 --split-seed 17
$PY -m scripts.audit_datasets --dataset all
```

PowerShell 命令如下：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m scripts.download_datasets --dataset all --confirm-research-terms
& $PY -m scripts.prepare_prism --split-seed 17
& $PY -m scripts.prepare_helpsteer2 --split-seed 17
& $PY -m scripts.audit_datasets --dataset all
```

若数据版本/hash 校验失败、出现未预期的格式错误、某个 split 为空或任意组间重叠数不为零，立即停止。PRISM 用于主要的用户关联反馈研究；HelpSteer2 用于受控评分回放。历史标签不是新收集的人工意见，不能用于声称节省了真实人工时间。

## 3. 模型 API 预检

下面命令只检查模型 endpoint 的连通性和响应格式，不属于正式实验。实验 1A 已实现：正式 runner 比较 `seed-only`、`fixed-prompt` 和改造版 `Self-Instruct`，按共同 API token 预算生成候选，训练 Qwen LoRA，并在 PRISM development 上计算 chosen/rejected 排序准确率与参考回答 NLL。Linux 服务器完整命令、预算参数、输出检查和指标限制见 [`server-quickstart.md`](server-quickstart.md)。

Linux 示例：

```bash
PY=.venv/bin/python
SEEDS=data/processed/prism/seed.jsonl
ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
MODEL=your-served-model-name
OUT=runs/synthesis-api-preflight
mkdir -p "$OUT"

$PY -m experiments.synthesis --data "$SEEDS" --output "$OUT/seed-only.jsonl" \
  --strategy seed-only --count 2 --seed 17
$PY -m experiments.synthesis --data "$SEEDS" --output "$OUT/fixed-prompt.jsonl" \
  --strategy fixed-prompt --count 2 --seed 17 --endpoint "$ENDPOINT" --model "$MODEL"
$PY -m experiments.synthesis --data "$SEEDS" --output "$OUT/self-instruct.jsonl" \
  --strategy self-instruct --count 2 --seed 17 --endpoint "$ENDPOINT" --model "$MODEL"
```

只有确认任务适合 instruction/task-input 演化后才运行 Evol-Instruct：

```bash
$PY -m experiments.synthesis --data "$SEEDS" --output "$OUT/evol-instruct.jsonl" \
  --strategy evol-instruct --evol-task-compatible --count 2 --seed 17 \
  --endpoint "$ENDPOINT" --model "$MODEL"
```

以上两候选命令仅用于 API 预检。任务适合 instruction/task-input 演化且已完成兼容性确认后，可在正式命令中加入 `evol-instruct` 和 `--evol-task-compatible`。

扩大实验前，须固定 seed group、任务与模型版本、解码参数、生成 token/API 成本预算和下游训练 token 预算。以上候选命令只用于 API 预检。正式 1A 命令见 `server-quickstart.md`；检查每个 JSONL 及对应 `.manifest.json`，确认模型/endpoint、请求数、token 记录、记录格式和生成成本。不得根据 API 预检宣称某方法更好。

实验 1A 是静态数据合成组件比较，不检验 RSI；实验 1B 才加入真实反馈和递归更新。方法和适配器说明见 `experiment-one-synthesis-baselines.md`。

## 4. 实验 1B：合成 RSI

**方案已定，当前不可运行。** 实验 1A 选定并冻结一种合成方法后，至少运行两轮，设置三组：

| 组别 | 反馈选择 | 是否修改下一轮合成策略 |
|---|---|---|
| `R0 Fixed` | 随机 | 否 |
| `R1 Random-update` | 随机 | 是 |
| `R2 Aladdin-update` | 单位评审成本影响力 | 是 |

三组使用相同的新候选数、人工分钟预算、清洗器、任务模型、训练 token 预算和 development 评价集。R0 仍接收同样的反馈作为下游监督，只是不修改合成策略；R1 用于区分“反馈更新”与“主动选反馈”；R2 检验拟议选择方法。若要声称人工反馈改进了合成过程，必须对新合成的候选做真实人工评审。

开始 1B 前，须实现标注界面/事件记录、按预算运行的查询器、带版本的策略更新器、轮次调度器、等预算训练和评价，以及分析脚本。具体 rubric 和更新规则见 `feasibility-pilot-protocol.md` 第 4-5 节；不能用历史标签回放替代新的人类评审。

## 5. 实验二：清洗方法比较

**方案已定，当前不可运行。** 对同一冻结的合成候选池比较：

1. 不做清洗，只做解析。
2. 确定性清洗：schema 检查、文本规范化、精确去重，以及字符 5-gram TF-IDF 相似度阈值 `0.92` 的近重复过滤。
3. 固定质量评分过滤：在确定性清洗后运行固定 rubric judge；仅在人类校准审计达到预设门槛后启用。

使用相同下游任务模型、训练 token 预算和候选预算。根据 development 表现、保留率、保留数据中的低质量残留率、误删率和 judge 成本选择清洗器。最终阈值在进入反馈选择实验前冻结。当前活动代码中还没有清洗器、judge client、审计抽样器或分析代码。

## 6. 实验三：反馈选择

**方案已定，当前活动代码不可运行。** 用已冻结的合成和清洗方法构造同一个反馈池，比较随机、不确定性偏好选择和单位评审成本影响力选择；分歧选择作为诊断方法。历史标签回放用于检查选择机制；另外对新候选开展 live 标注，测量真实评审时间。所有方法使用同一累计人工分钟预算，报告每分钟有效反馈数、评审耗时分布、不确定性和 development 下游表现。选择器不得在查询释放前读取隐藏标签。

## 7. 实验四：整体 RSI 对照

**方案已定，当前不可运行。** 只有实验一至三的组件选择完成并冻结后，才比较以下四组：

| 组别 | 数据流程 | 反馈 | 是否递归更新 |
|---|---|---|---|
| A | 仅 seed | 无 | 否 |
| B | 固定合成 + 固定清洗 | 随机 | 否 |
| C | 与 B 相同的初始流程 | 随机 | 是 |
| D | 与 B 相同的初始流程 | 单位评审成本影响力 | 是 |

各组的数据分组、候选数、任务模型版本、训练 token、质量 judge 预算和累计人工分钟必须一致。主要比较为 B-A（固定数据流程的价值）、C-B（递归更新的价值）和 D-C（RSI 中拟议反馈选择的额外价值）。若要分别声称合成与清洗受益，还须增加仅更新合成、仅更新清洗两种消融。

逐轮报告效果、策略版本、各类成本和 final-test 人类偏好及不确定性。final test 应保持封存，直到协议和阈值全部冻结。

## 当前就绪状态

| 环节 | 状态 |
|---|---|
| CPU/数据环境和安装说明 | 已就绪；干净的本地虚拟环境验证通过 |
| 固定版本数据下载、准备和审计 | 已就绪；已在本地数据上运行，split 交叉重叠为零 |
| 实验 1A 合成、Qwen LoRA 训练和 development 评价 | 已实现；自动化测试通过，待目标 GPU/API 服务器运行 |
| 在线模型 endpoint API 预检 | 运行前需协作者提供可访问的服务器 endpoint |
| 清洗和质量 judge 实验 | 已有方案；活动代码尚未实现 |
| 人工反馈采集及真实耗时测量 | 已有方案；界面/事件记录和真人试验尚未实现 |
| 选择策略回放与 live 评价 | 已有方案；活动代码尚未实现 |
| 多轮整体 RSI 和确认性结果 | 已有方案；活动代码尚未实现 |

当前仓库可以安装环境、准备数据、运行测试并执行实验 1A；实验 1B、清洗、反馈选择和整体 RSI 仍需完成实现，当前还不是覆盖完整方案的端到端系统。
