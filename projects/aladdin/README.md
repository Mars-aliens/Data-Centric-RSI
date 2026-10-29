# MLE-Aladdin

Aladdin 研究面向生成式任务的数据中心化递归改进流程，重点关注可用样本有限、用户要求会随着具体反馈逐步明确的场景。

## 应用场景示例：生成式中文输入法

一个有现实依据的目标应用是生成式中文输入法：系统需要把用户的原始按键序列转成中文，也可能根据上下文续写，或按用户要求改写文字。输入序列可能是完整拼音、简拼、混合输入或含错键的拼音；同一输入的可接受结果也可能因上下文和个人习惯而不同。用户通常通过选词、改字或修改生成结果表达偏好，因此输入日志、用户选择和修改记录可能成为有限且不完整的反馈来源。

这一场景与 Aladdin 关注的问题相符：可用的高质量训练案例和明确偏好有限，人工逐条评审成本高，部分个性化要求要从实际使用和修改中逐渐显现。Aladdin 的三阶段流程可用于研究如何扩充候选训练数据、自动清理候选，并在相同评审预算下挑出更值得用户反馈的样本，再用反馈改进后续数据构建策略。

任务例子和技术背景参考 Ding 等人的 GeneInput 论文（[arXiv:2311.01166](https://arxiv.org/abs/2311.01166)）：论文研究完整按键序列到汉字（FK2C）、智能联想和对话式改写，并讨论利用输入历史、用户信息及反馈优化系统。论文使用其自建 XF 数据集及特定输入法数据和模型；这不等于这些数据可直接获取，也不代表 Aladdin 已在输入法场景完成验证。

**当前实验边界：** 现有实验一仍在 PRISM 对话数据上比较数据合成方法，是 Aladdin 数据构建流程的可行性验证，不是中文输入法实验。迁移到输入法前，需要另行确认可合法使用的数据集、按键序列及目标文本配对、输入噪声覆盖、适配的合成基线、任务模型和 top-k/人工偏好评测方案。不得把当前 PRISM 结果表述为输入法场景的效果证据。

## 当前研究方向

```text
少量固定种子样本
  -> 阶段一：合成候选数据（不请求人工反馈）
  -> 阶段二：自动清洗候选（不请求人工反馈）
  -> 阶段三：选择值得评审的候选并收集人工反馈
  -> 用反馈调整下一轮的合成和清洗策略
  -> 下一轮
```

人工反馈只在阶段三进入系统。阶段一、二不让人逐条判断候选是否“好”；阶段三挑选适合优先评审、能帮助澄清要求或影响后续决策的候选。这里的“好”指反馈价值高，不表示系统已判定内容质量合格。公开数据集已有的标签只用于固定的离线评价或查询策略回放，不等于本轮新收集的人工反馈，也不参与候选生成过程。三个数据操作由同一个外层 RSI 循环连接。

当前方向已有可行性试验方案，但还没有确认性结果，也尚未实现为完整的端到端系统。方案初步确定了数据集、对照方法、数据划分和预算控制；正式实验前仍须通过数据访问、隐私、模型版本和数据泄漏检查。

三阶段候选对照与系统对照见 `docs/baseline-shortlist.md`。这些是可行性研究建议，不是已经验证的结论。完整试验方案见 `docs/feasibility-pilot-protocol.md`。

三阶段分工和递归关系的简明说明见 `docs/three-stage-research-flow.md`。

实验一的数据合成环节已有静态对照与 Qwen LoRA 下游评价 runner，默认比较 `seed-only`、固定提示词和改造版 Self-Instruct；任务适用并经确认后可加入改造版 Evol-Instruct。本环节不请求新人工反馈、不按人工意见筛候选；development 中已有的 PRISM 标签只用于统一的离线结果评价。生成模型和下游任务模型统一固定为 `Qwen/Qwen3-8B`，revision 为 `b968826d9c46dd6066d109eabc6255188de91218`，并关闭 thinking 模式。服务器正式运行命令见 `docs/server-quickstart.md`，方法和评价定义见 `docs/experiment-one-data-synthesis.md`。确定性测试不调用模型，也不能作为合成质量证据。

协作者的环境配置、数据准备、API 预检、实验执行顺序和当前实现状态见 `docs/reproduction-runbook.md`。各实验阶段的目标模块、数据格式、算法接口、开发顺序和测试门槛见 `docs/implementation-design.md`。数据清洗、反馈选择、反馈驱动的递归更新及完整多轮 RSI 仍处于待实现阶段。

Linux 服务器按步骤执行的快速说明见 `docs/server-quickstart.md`。

人工反馈成本、主动偏好学习和标注分歧相关文献见 `docs/human-feedback-literature.md`。

## 项目结构

- `README.md`：研究范围和快速入口。
- `CONTRIBUTING.md`：协作约定。
- `docs/baseline-shortlist.md`：三阶段对照及系统对照建议。
- `docs/three-stage-research-flow.md`：Aladdin 三阶段研究流程与当前实验对应关系。
- `docs/feasibility-pilot-protocol.md`：当前可行性试验方案、数据集和预算决策。
- `docs/experiment-one-data-synthesis.md`：实验一数据合成对照及运行步骤。
- `docs/reproduction-runbook.md`：环境、数据、实验执行顺序及 GitHub 路径说明。
- `docs/server-quickstart.md`：Linux 服务器安装、数据准备、API 预检和正式实验流程说明。
- `docs/implementation-design.md`：目标代码结构、数据契约、算法和测试门槛。
- `baselines/synthesis/UPSTREAMS.md`：上游仓库版本、许可和改造说明。
- `configs/datasets.json`：固定的数据集版本、文件和使用限制。
- `scripts/`：数据下载、准备、审计和固定版本 baseline 获取脚本。
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
