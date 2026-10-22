# MLE-Aladdin

Aladdin 研究面向生成式任务的数据中心化递归改进流程，重点关注可用样本有限、用户要求会随着具体反馈逐步明确的场景。

## 当前研究方向

```text
少量种子样本
  -> 合成候选数据
  -> 清洗并检查质量
  -> 选择有代表性的样本，收集人工反馈
  -> 根据反馈调整下一轮的数据合成与清洗策略
  -> 进入下一轮并评估下游任务表现
```

三个数据操作由同一个外层 RSI 循环连接。

当前方向已有可行性试验方案，但还没有确认性结果，也尚未实现为完整的端到端系统。方案初步确定了数据集、对照方法、数据划分和预算控制；正式实验前仍须通过数据访问、隐私、模型版本和数据泄漏检查。

三阶段候选对照与系统对照见 `docs/baseline-shortlist.md`。这些是可行性研究建议，不是已经验证的结论。完整试验方案见 `docs/feasibility-pilot-protocol.md`。

实验一 1A 已有静态合成对照与 Qwen LoRA 下游评价 runner，默认比较 `seed-only`、固定提示词和改造版 Self-Instruct；任务适用并经确认后可加入改造版 Evol-Instruct。生成模型和下游任务模型统一固定为 `Qwen/Qwen3-8B`，revision 为 `b968826d9c46dd6066d109eabc6255188de91218`，并关闭 thinking 模式。服务器正式运行命令见 `docs/server-quickstart.md`，方法和评价定义见 `docs/experiment-one-synthesis-baselines.md`。确定性测试不调用模型，也不能作为合成质量证据。

协作者的环境配置、数据准备、API 预检、实验执行顺序和当前实现状态见 `docs/reproduction-runbook.md`。各实验阶段的目标模块、数据格式、算法接口、开发顺序和测试门槛见 `docs/implementation-design.md`。实验 1B、清洗、反馈选择及完整多轮 RSI 仍处于待实现阶段。

Linux 服务器按步骤执行的快速说明见 `docs/server-quickstart.md`。

人工反馈成本、主动偏好学习和标注分歧相关文献见 `docs/human-feedback-literature.md`。

## 项目结构

- `README.md`：研究范围和快速入口。
- `CONTRIBUTING.md`：协作约定。
- `docs/baseline-shortlist.md`：三阶段对照及系统对照建议。
- `docs/feasibility-pilot-protocol.md`：当前可行性试验方案、数据集和预算决策。
- `docs/experiment-one-synthesis-baselines.md`：实验一合成对照的实现状态。
- `docs/reproduction-runbook.md`：环境、数据、实验执行顺序及 GitHub 路径说明。
- `docs/server-quickstart.md`：Linux 服务器安装、数据准备、API 预检和正式实验流程说明。
- `docs/implementation-design.md`：目标代码结构、数据契约、算法和测试门槛。
- `baselines/synthesis/UPSTREAMS.md`：上游仓库版本、许可和改造说明。
- `configs/datasets.json`：固定的数据集版本、文件和使用限制。
- `scripts/`：数据下载、准备和审计脚本。
- `experiments/synthesis/`：当前已实现的数据合成方法和命令行程序。
- `tests/`：不依赖 GPU 或在线模型的确定性测试。

数据集、模型权重、凭据、服务器配置和运行产物不得放入待提交的项目文件中。

## 环境安装

已在 Windows、Python 3.12.3 的 CPU 环境验证。数据准备和单元测试不需要 GPU。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS 将 Python 路径改为 `.venv/bin/python`。仅需运行时依赖可安装 `requirements.txt`；开发依赖文件还包含测试、数据下载和分析依赖。GPU 训练依赖单独安装，因为 PyTorch wheel 需要匹配服务器驱动和 CUDA 环境。

## 数据集准备

下载脚本使用固定的数据集版本，数据写入 Git 忽略的 `data/raw/`。下载前先阅读 `docs/datasets.md`。PRISM 包含不同许可条件下的参与者文本和模型回复，需要明确确认使用限制。只下载对话、utterance 和 PII 元数据，不下载 survey 文件。

```powershell
.\.venv\Scripts\python.exe -m scripts.download_datasets --dataset all --confirm-research-terms
.\.venv\Scripts\python.exe -m scripts.prepare_prism --split-seed 17
.\.venv\Scripts\python.exe -m scripts.prepare_helpsteer2 --split-seed 17
.\.venv\Scripts\python.exe -m scripts.audit_datasets --dataset all
```

PRISM 处理后的数据位于 `data/processed/prism/`，包含 `seed`、`feedback_pool`、`development` 和 `final_test`。历史评分保存在独立的 `labels/` 文件中；对话级 `open_feedback` 是单独的反馈查询，不会复制到每条 response 上。HelpSteer2 将同一 prompt 的回答放在同一 split，训练集被确定性拆分，官方 validation 留作 final test；五个维度的评分另存于 `data/processed/helpsteer2/labels/`。seed 中保留 helpfulness 最高的回答及并列项。

这些标签只用于公开数据集回放，不是新收集的人工反馈。不要提交原始或处理后的数据集文件。
