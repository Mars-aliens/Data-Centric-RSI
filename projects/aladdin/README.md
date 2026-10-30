# MLE-Aladdin

Aladdin 研究面向生成式任务的数据中心化递归改进流程，重点关注可用样本有限、用户要求会随着具体反馈逐步明确的场景。

## 当前研究方向

```text
少量固定种子样本
  -> 阶段一：合成候选数据（不请求人工反馈）
  -> 阶段二：自动清洗候选（不请求人工反馈）
  -> 阶段三：选择值得评审的候选并收集 Qwen3-32B 模拟反馈
  -> 用反馈调整下一轮的合成、清洗和反馈选择策略
  -> 下一轮
```

当前实验在第三阶段选择候选后，由独立的 Qwen3-32B 服务生成模拟反馈。阶段一、二不请求反馈；模型反馈不代表真人意见或人工耗时。公开数据集已有的标签只用于固定的离线评价或查询策略校准，不参与候选生成。三个数据操作由同一个外层 RSI 循环连接。

PRISM 主场景的多轮实验 runner 已在本地实现，覆盖自动合成、自动清洗、隔离池抽查、第三阶段模型模拟反馈或真人反馈、下一轮策略更新、Qwen LoRA development 评价和final-test 自动盲评入口及可选真人入口。CPU 单元测试通过不等于已完成服务器 GPU/API 试跑；确认性实验的样本量与预算仍须预检后冻结。

第二任务场景研究“基于 LCCC 构造的中文语境续写（智能联想代理任务）”，数据使用、样本转换、分组划分、真人反馈及递归对照设计见 `docs/chinese-intelligent-association-protocol.md`。LCCC 仍是候选数据源，下载前须人工审查数据条款；准备脚本会排除命中常见直接标识符规则的整段对话，但这不能替代完整隐私审查。该场景与 PRISM 实验独立，尚无实验结果。

当前主实验设计按三项比较组织：整体递归收益、更新消融、反馈选择对照。详细流程见 `docs/feedback-driven-update-experiment.md`；已选方法与基线见 `docs/baseline-shortlist.md`。工程试跑配置位于 `configs/feedback-rsi-pilot.example.json`，它用于检查整条链路，不是确认性实验规模。

协作者的环境配置、数据准备和主实验运行步骤见 `docs/reproduction-runbook.md`；Linux 服务器按步骤执行见 `docs/server-quickstart.md`。各模块的数据格式、接口和实现边界见 `docs/implementation-design.md`。

人工反馈成本、主动偏好学习和标注分歧相关文献见 `docs/human-feedback-literature.md`。

## 项目结构

- `README.md`：研究范围和快速入口。
- `CONTRIBUTING.md`：协作约定。
- `docs/baseline-shortlist.md`：三阶段对照及系统对照建议。
- `docs/feedback-driven-update-experiment.md`：当前主实验设计。
- `docs/feasibility-pilot-protocol.md`：早期可行性方案及数据、反馈协议参考；实验比较以当前主实验设计为准。
- `docs/chinese-intelligent-association-protocol.md`：第二任务场景的 LCCC 数据构造与实验协议设计。
- `docs/reproduction-runbook.md`：环境、数据、实验执行顺序及 GitHub 路径说明。
- `docs/server-quickstart.md`：Linux 服务器安装、数据准备和多轮 pilot 运行步骤。
- `docs/implementation-design.md`：当前实现结构、数据边界、算法和测试方式。
- `baselines/synthesis/UPSTREAMS.md`：上游仓库版本、许可和改造说明。
- `configs/datasets.json`：固定的数据集版本、文件和使用限制。
- `scripts/`：数据下载、准备、审计和固定版本 baseline 获取脚本。
- `experiments/synthesis/`：当前已实现的数据合成方法和命令行程序。
- `experiments/cleaning/`、`experiments/feedback/`、`experiments/rsi/`：清洗、查询选择、真人反馈、策略更新和多轮实验控制器。
- `experiments/evaluation/`：Qwen LoRA development 评价与 final-test 盲评工具。
- `tests/`：不依赖 GPU 或在线模型的确定性测试。

PRISM 主 runner 支持 `overall`、`ablation`、`selection` 和 `all` 四种比较入口。每种比较中所有组的首轮共用候选池。长任务可用 `--resume` 从写入的检查点继续；只有配置、输入文件、split manifest 和 runner 代码 hash 均未变化时才允许恢复。当前配置和执行结果应以本地 pilot 验证为准，确认性实验参数尚未冻结。

LCCC 中文续写场景目前有数据准备和协议文件，但尚未接入上述 PRISM RSI runner；不能将 PRISM 的执行结果外推到该场景。

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

Linux/macOS 将 Python 路径改为 `.venv/bin/python`。仅需运行时依赖可安装 `requirements.txt`；开发依赖文件还包含测试、数据下载和分析依赖。Qwen LoRA 训练环境安装 `requirements-qwen.txt`；vLLM 模型服务使用独立环境安装 `requirements-vllm.txt`，避免和训练环境混用。PyTorch wheel 需要匹配服务器驱动和 CUDA 环境。

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
