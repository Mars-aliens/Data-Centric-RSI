# 技术一：安全自进化与故障隔离

本文说明当前新路线中技术一的代码入口、运行步骤和结论边界。旧 `independent`、`dependency-aware-joint` Harness 实验仍保留为历史证据，不是本流程的输入，也不能替代这里的长周期候选隔离实验。

4 批工程 pilot 的步骤以[技术一实验协议 v1](../experiments/technology_one/protocol-v1.md)为准；正式 100-cycle 矩阵以[正式长周期协议 v1](../experiments/technology_one/formal-protocol-v1.md)和机器可读计划为准。本手册负责服务器部署和命令。正式研究规模已冻结，但服务器模型、数据哈希、Docker digest、资源预算仍须预检登记。

## 1. 要验证什么

技术一允许 Decision LLM 对 Data System 运行代码提出有界修改，但不能让提案直接覆盖稳定系统。流程是：

```text
运行观察 + 已释放反馈
  → Decision LLM 提出源码修改
  → 本地 schema/path/size 检查并生成带哈希 manifest 的候选 bundle
  → 候选在网络隔离、只读父 checkpoint 的 Docker 中运行
  → 检查输出协议、标签、预算、版本来源和稳定 checkpoint 哈希
  → 合格候选仅进入后续观察；失败候选停止，稳定流程继续
```

当前实现验证的是**一份候选 bundle 在一个长周期任务流中的隔离和运行安全**。候选 Task Model 可以依次使用同一条流中已释放的反馈续训；候选源码在该 run 期间固定。此 T1 runner 不会逐批自动生成新源码提案，也不会晋升候选为稳定版本。正式协议要求每 10 个周期有一次新候选提案机会，共 10 次；该调度和多候选恢复当前尚未实现，代码完成并通过集成测试前不得启动正式递归实验。候选是否经过多批次观察后保留属于后续 T3 的研究问题。

## 2. 环境与模型配置

新服务器建议使用 Linux、Python 3.11、Git、NVIDIA 驱动、CUDA-enabled PyTorch 2.7.1、Docker Engine 和 NVIDIA Container Toolkit。T1 要求稳定分支和候选分支有不同物理 GPU 或 MIG 设备；Task Model smoke 只需分配一张卡。先克隆仓库，再在仓库目录创建独立环境：

```bash
git clone https://github.com/Mars-aliens/Data-Centric-RSI.git
cd Data-Centric-RSI/projects/mle-lego
git rev-parse HEAD

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

按新服务器的 NVIDIA 驱动/平台，在 PyTorch 官方安装选择器选出 PyTorch 2.7.1 对应的 CUDA wheel；不要照抄旧服务器 CUDA 配置。先安装该选择器生成的 PyTorch wheel，再安装仓库运行依赖并执行单元测试：

```bash
python -m pip install torch==2.7.1 --index-url "<PyTorch 官方选择器给出的 CUDA wheel index>"
python -m pip install -r requirements-server.txt
python -m pip check
python -m pytest -q
```

占位符必须替换成选择器为该服务器生成的 index URL。仓库依赖文件不指定 CUDA 变体，避免把一台服务器的 GPU wheel 错装到另一台服务器；若 PyTorch 2.7.1 不兼容目标服务器，先冻结兼容版本组合再开展实验。`requirements-server.txt` 安装 numpy、scikit-learn、pytest、Transformers、PEFT 和 Accelerate；`requirements-research.txt` 还安装 Hugging Face 数据下载库。Task Model 权重、原始数据、服务器私有配置和运行输出均不随仓库上传。

在项目目录设置运行变量。Decision API endpoint 和 API key 由这台服务器提供，不预设旧服务器地址：

```bash
export LEGO_ROOT="$PWD"
cd "$LEGO_ROOT"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export LEGO_TASK_MODEL_DEVICE=cuda
bash "$LEGO_ROOT/scripts/check_lego_install.sh"
nvidia-smi
```

新路线首轮 pilot 的 Decision LLM 候选为百炼 Qwen3.8-Flash；若实际候选质量达不到门槛，再对 Qwen3.8-Max 单独做 pilot。模型卡版本标识不一定等于 API `model_id`；私有 profile 必须填写 API 文档/控制台确认的调用 ID，不能把模型市场链接或未验证的版本标识当作 endpoint/model ID。确认账号、地域、OpenAI-compatible Chat Completions 和 JSON 支持后再 smoke。密钥不得写入 profile、候选代码、候选容器环境或 Git。

先将示例复制到仓库外的私有目录，填写服务商确认的 API model ID/provider，并由服务器密钥管理注入 endpoint/key：

```bash
mkdir -p "$LEGO_ROOT/server-private"
cp "$LEGO_ROOT/code/technologies/technology_one/decision_model.qwen.example.json" \
   "$LEGO_ROOT/server-private/decision-model.json"
export LEGO_POLICY_API_URL="<服务商 OpenAI-compatible Chat Completions endpoint>"
export LEGO_POLICY_API_KEY="<由服务器密钥管理器注入的密钥>"
export LEGO_POLICY_TIMEOUT_SECONDS=600
```

先运行接口 smoke；它会实际调用配置模型、校验候选 JSON 结构，只输出模型 ID 和服务端返回的调用/Token 计数，不会写候选 bundle：

```bash
"$TRAIN_PY" scripts/smoke_decision_endpoint.py \
  --profile /data/configs/decision-model-profile.json
```

输出 `status: ok` 且 `schema_valid: true` 后再做提案筛选。对固定的 10 条合规输入，要求 API/JSON/schema 10/10 成功、无越界修改，至少 8/10 提案通过本地 schema/path/size/bundle 校验且改动后 bundle 源文件哈希确有变化。该机械检查不等价于语义有效；另需记录提案是否改动可执行逻辑、理由是否对应观察、是否有可验证预期，注释或格式变化不能算作有意义候选。记录 token、费用和延迟，pilot 前设定费用上限。此筛选不按候选是否提升任务指标来选模型，也不计入正式实验。Qwen3.8-Flash 截图标价为输入/输出 0.8/2.7 元每百万 tokens；Qwen3.8-Max 为 12/36 元。先测 Flash；通过门槛就冻结 Flash，不必再测 Max。未通过时先修 prompt/协议并重测一次，仍失败才评估 Max；两者都失败则暂停 T1 修协议。主实验不需要同时运行两者、百炼 `Decision Model`、Kimi K3 或所有 Qwen 尺寸。新服务器的模型清单和 GPU 资源以执行者实际盘点为准。

Task Model 是另一个角色：首轮基准建议使用服务器本地 `Qwen/Qwen3-8B` Hugging Face causal-LM 权重，由 stable/candidate 分支共享同一个冻结底座。配置示例位于 `code/technologies/technology_one/task_model.qwen.example.json`。执行者需填实际本地快照路径和不可变 revision，并检查 tokenizer 标签编码、LoRA target modules、显存和短步 parent-checkpoint 续训。Qwen3-8B 是首轮固定基准，不代表任务指标已证明最好。纯托管 API 不提供当前代码所需的本地 LoRA parent-checkpoint 续训能力，不能直接充当 Task Model。更换 Task Model 型号或 revision 时必须使用独立 run，不能并入基准结果。示例不预设显存额度；按服务器资源填写，Docker 本身也不提供 GPU 显存硬隔离。

## 3. 准备固定任务输入

原始数据、模型权重、服务器 profile 和运行结果均保存在仓库外。安装下载依赖后，从任务定义固定的 HelpSteer2 revision 导出新 runner 所需的原始评分字段：

```bash
python -m pip install -r "$LEGO_ROOT/requirements-research.txt"
mkdir -p "$LEGO_ROOT/data"
"$TRAIN_PY" - "$LEGO_ROOT/data" <<'PY'
import json
from pathlib import Path
import sys
from datasets import load_dataset

output = Path(sys.argv[1])
revision = "990b2711a36180dd19d9c94b8627844866f8982a"
fields = ("prompt", "response", "helpfulness", "correctness", "coherence", "complexity", "verbosity")
for split in ("train", "validation"):
    rows = load_dataset("nvidia/HelpSteer2", split=split, revision=revision)
    path = output / f"helpsteer2-{split}.jsonl"
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps({field: row[field] for field in fields}, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} rows -> {path}")
PY

sha256sum "$LEGO_ROOT/data/helpsteer2-train.jsonl" \
          "$LEGO_ROOT/data/helpsteer2-validation.jsonl"
```

若数据集要求访问授权，先接受数据集条款并在服务器自行登录；不要把访问 token 写入脚本、profile 或日志。`validation` 是锁定的 final-test split，不提供给 Decision LLM 或候选代码。先检查两个 JSONL 每行均包含 `prompt`、`response` 和五项评分，再生成信息边界预检 bundle：

```bash
"$TRAIN_PY" scripts/prepare_long_horizon_task.py \
  --train-jsonl /data/helpsteer2-train.jsonl \
  --validation-jsonl /data/helpsteer2-validation.jsonl \
  --output-dir /data/lego-preflight/seed-17 \
  --stream-seed 17 --groups-per-batch 50 --feedback-delay 2
```

预检 bundle 用于检查切分、批次和反馈释放时间；runner 仍从上述原始 train/validation JSONL 重建同一个任务。正式实验固定并记录数据哈希、dataset revision、`stream_seed`、`groups_per_batch` 和 `feedback_delay`。smoke 可以另建小规模数据副本或使用较大的 `groups_per_batch` 减少批次数，但 smoke 不能计入正式结果。

数据审计先按完整 development 流生成 manifest，不设置 `--max-batches`。短 smoke 使用相同数据、seed 和流顺序，只排程前四批；manifest 会记录流前缀限制，短流结果不得与完整流合并：

```bash
"$TRAIN_PY" scripts/prepare_long_horizon_task.py \
  --train-jsonl /data/helpsteer2-train.jsonl \
  --validation-jsonl /data/helpsteer2-validation.jsonl \
  --output-dir /data/lego-preflight/seed-17-smoke \
  --stream-seed 17 --groups-per-batch 50 --feedback-delay 2 \
  --max-batches 4
```

创建仓库外的 Task Model 配置，例如 `/data/configs/task-model.json`。以下模型路径和 ID 是占位字段，按新服务器上的实际可用模型填写；示例超参数仅供 smoke：

```json
{
  "base_model_id": "<selected-local-model-id>",
  "base_model_path": "/models/<selected-model>/snapshots/<immutable-snapshot>",
  "base_model_revision": "<immutable-snapshot-id>",
  "max_steps": 1,
  "learning_rate": 0.0002,
  "per_device_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_length": 1024,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "deterministic": true
}
```

`max_steps: 1` 只用于首轮四批工程 smoke，确保在有限成本下实际发生一次 parent-checkpoint 续训；它不是正式训练配方。记录实际模型权重 revision、Tokenizer、target modules、显存限制和该服务器分配的 GPU。示例不预设显存额度；按服务器情况决定是否设置 `max_gpu_memory_gib`，它只是进程软限制。所有实验臂必须使用一致配置；GPU 编号不可照抄其他服务器的示例。

先运行不含候选的固定流水线 `B0-fixed`，建立同流控制并确认延迟反馈后确实触发 checkpoint 更新。批次数限定为 4：

```bash
"$TRAIN_PY" scripts/run_long_horizon_qwen.py \
  --train-jsonl /data/helpsteer2-train.jsonl \
  --validation-jsonl /data/helpsteer2-validation.jsonl \
  --config-json /data/configs/task-model-smoke.json \
  --output-dir /data/lego-runs/t1-pilot/b0-fixed-seed-17 \
  --run-id b0-fixed-seed-17 --seed 17 --stream-seed 17 \
  --groups-per-batch 50 --feedback-delay 2 --max-batches 4
```

必须看到 `completed_batches=4`、`target_batches=4`、`status=complete`，且 batch 3 开始处理 batch 0 释放的反馈并发生训练。否则先修固定流水线，不进入 T1。

## 4. 生成候选提案

准备决策时已可见的运行观察 JSON，例如 `/data/lego-observations/seed-17.json`。观察可以包括已释放反馈和由这些反馈计算的结果；当前批次标签、未释放反馈、held-out 和 final-test 信息不可提供。带标签字段只允许出现在 `released_feedback` 下。不要把完整数据集、权重、私有 profile 或凭据放进 observation。

候选源目录只使用项目的 `code/`，不要使用整个仓库根目录。输出路径必须在 `code/` 之外：

```bash
"$TRAIN_PY" scripts/propose_t1_candidate.py \
  --profile /data/configs/decision-model-profile.json \
  --source-dir "$LEGO_ROOT/code" \
  --output-dir "$LEGO_ROOT/runs/t1-candidates/seed-17-candidate-001" \
  --observation-json /data/lego-observations/seed-17.json \
  --candidate-id t1-seed-17-candidate-001 \
  --base-version-id runtime-stable
```

命令会：

- 从 profile 的 `decision_model` 读取 provider、API 模型 ID、采样设置和 endpoint/key 环境变量名；
- 只向 Decision LLM 提供允许的源码上下文及 observation，不传 API key；
- 对 JSON、路径逃逸、文件类型、单文件/总 patch 大小及文件级内容是否实际变化作本地检查；这不判断代码的语义效果。schema 不合法时最多追加一次纠正请求；
- 排除数据、权重、运行目录、环境变量文件和常见密钥文件；拒绝源码树中的有效符号链接；
- 生成 `candidate_proposal.json`，记录 candidate ID、模型、提示词哈希、改动文件哈希、调用成本和源码哈希。

提案命令以临时目录构建并在成功后安装 bundle；输出目录已存在时会拒绝覆盖。manifest 中的 `runtime-stable` 会在候选首次运行时绑定到该 run 的真实稳定版本 ID；若显式给出的父版本不匹配，候选不会启动。

## 5. 配置候选与显存隔离

候选容器镜像使用匹配 Task Model 环境的 CUDA-enabled PyTorch runtime，并且基础镜像必须通过 digest 固定。仓库提供构建配方；先从批准的镜像仓库取得实际基础镜像 digest，再在服务器构建：

```bash
export PYTORCH_BASE_IMAGE="<registry>/<pytorch-runtime>@sha256:<64位小写digest>"
export T1_CANDIDATE_IMAGE_TAG="mle-lego-t1-candidate:locked"
bash scripts/build_t1_candidate_image.sh
```

脚本输出的 `image_id=sha256:<64位>` 是本机不可变镜像 ID。将该值填入候选配置的 `image` 字段。runner 会拒绝可变 tag，只接受本机 image ID 或 `repository@sha256:<digest>`；预检还会确认镜像已存在。正式实验记录基础镜像 digest、候选镜像 ID 和 `docker image inspect` 输出。

复制 `code/technologies/technology_one/candidate_shadow.example.json` 到仓库外的私有配置目录，至少设置：

- `candidate_id`：必须与提案命令一致；
- `candidate_dir`：提案命令生成的 bundle 绝对路径；
- `require_proposal_manifest`：正式运行保持 `true`；
- `image`：已在服务器本地构建并用 digest 固定的候选镜像；
- `limits`：容器时间、CPU、内存、PID、临时盘和候选 GPU 限制；
- `max_training_steps_per_batch`：候选每批训练上限。

先用确定性、无行为变化的 fixture 验证 Docker 和候选 checkpoint 链路，不调用 LLM：

```bash
"$TRAIN_PY" scripts/create_t1_smoke_candidate.py \
  --source-dir "$LEGO_ROOT/code" \
  --output-dir /data/lego-candidates/t1-clean-smoke \
  --candidate-id t1-clean-smoke
```

在私有候选配置中将 `candidate_id` 设为 `t1-clean-smoke`、`candidate_dir` 设为上面生成的绝对路径，并填写已固定的 image digest 和候选 GPU。该 fixture 只用于基础设施 smoke，不是 RSI 提案或研究结果。Decision LLM 接口 smoke 已在本手册第 2 节执行；基础设施 pilot 通过后，再用完成内容审查的 LLM 候选开展独立 pilot。

容器使用 `--network=none`、只读根文件系统、只读候选代码、只读父 checkpoint 和只读 Task Model 权重；候选唯一可写位置 `/candidate-output` 是容量受 `tmpfs_mib` 限制的临时文件系统，容器退出后由可信宿主进程复制工件并删除容器。宿主仅保留候选当前有效 checkpoint；新 checkpoint 写入状态后，旧候选 checkpoint 和失败尝试输出会被清理。候选 stdout 缓存最多 4 MiB、请求体最多 32 MiB，stderr 只保留末尾 4 KiB。Decision API key 不进入候选容器。Docker 无 GPU 显存硬配额，稳定分支和候选分支必须使用不同物理 GPU 或 MIG；不能只设置显存软上限来声称隔离。

## 6. 预检与运行

先按新服务器实际分配情况，将稳定 runner 使用的物理卡写入 `CUDA_VISIBLE_DEVICES`；候选配置中的 `gpu_devices` 使用另一张空闲物理卡或独立 MIG 设备。不要假设 GPU 0/1 在该服务器可用。然后执行不启动训练的预检：

```bash
export CUDA_VISIBLE_DEVICES="<stable-physical-gpu-id>"
export LEGO_TASK_MODEL_DEVICE=cuda

"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py \
  --train-jsonl /data/helpsteer2-train.jsonl \
  --validation-jsonl /data/helpsteer2-validation.jsonl \
  --task-config-json /data/configs/task-model.json \
  --candidate-config-json /data/configs/t1-candidate-001.json \
  --output-dir /data/lego-runs/t1-seed-17-candidate-001 \
  --run-id t1-seed-17-candidate-001 \
  --seed 17 --stream-seed 17 --groups-per-batch 50 --feedback-delay 2 \
  --max-batches 4 \
  --preflight-only
```

预检应显示 `preflight_ok`、候选 manifest 哈希和 `final_test_used: false`。确认 Docker daemon、镜像 digest、Task Model revision、数据路径以及 GPU 不重叠后，用相同参数去掉 `--preflight-only` 运行四批 pilot。`T1-shadow-clean` 和故障实验必须使用各自的新 output directory、run ID 和 candidate ID。正式运行长度是 100 cycles，不传 `--max-batches`；但周期性候选调度未实现前，当前 runner 仅可用于 pilot/工程验证，不可作为正式 RSI 实验入口。

先运行无故障候选：

```bash
"$TRAIN_PY" scripts/run_long_horizon_t1_qwen.py \
  --train-jsonl /data/helpsteer2-train.jsonl \
  --validation-jsonl /data/helpsteer2-validation.jsonl \
  --task-config-json /data/configs/task-model-smoke.json \
  --candidate-config-json /data/configs/t1-clean-smoke.json \
  --output-dir /data/lego-runs/t1-pilot/t1-clean-seed-17 \
  --run-id t1-clean-seed-17 --seed 17 --stream-seed 17 \
  --groups-per-batch 50 --feedback-delay 2 --max-batches 4
```

确认 stable 和 candidate 都完成四批，candidate 在 batch 3 收到已释放反馈，并保留的是候选自己的 checkpoint。随后为独立的候选配置设置故障映射 `"fault_injection": {"3": "invalid_protocol"}`，使用全新的 candidate、run ID 和 output directory 重跑。预期候选在 batch 3 被拒绝并停止，stable 仍完成 `4/4`，stable checkpoint 前后哈希一致。通过 pilot 后，才逐类执行下一节的完整故障覆盖；正式效能 run 不注入故障。

## 7. 故障注入与判读

故障注入通过候选配置中的 `fault_injection` 设置 batch index 和故障类型。至少分别测试超时、非零退出、非法 JSON/协议、错误父版本、漏预测、非法标签、报告预算超限和写只读父 checkpoint。每种故障使用独立 run/output dir，避免故障类型互相混淆；正式效能比较不启用故障注入。

检查 `/data/lego-runs/<run>/stable/summary.json`，重点记录：

- `status`、`completed_batches/target_batches`、稳定分支预序列指标和成本；
- `candidate_shadow.gate_status`、`terminal_failure`、候选 manifest 哈希和实际父版本绑定；
- `candidate_history` 中每批候选状态、失败代码、稳定父 checkpoint 前后哈希和反馈归属；
- `candidate_prequential_metrics` 与稳定指标的分开结果；
- `held_out_used` 和 `final_test_used` 必须为 `false`。

T1 的关键成功条件是候选故障被限制在候选分支、稳定分支按目标批次继续运行、父 checkpoint 哈希不变、失败原因可审计且资源/时间限制有效。候选安全门通过只表示可以观察，不表示性能提升或可以成为稳定版本。任务质量是独立结果，不能用“候选通过安全检查”代替。

## 8. 结论边界与尚需服务器验证

本地代码现在包含候选提案生成、bundle provenance、Docker 隔离、运行时父版本绑定、候选输出校验、故障注入和稳定分支配对 runner。还没有在真实服务器上验证百炼/Qwen/Kimi endpoint、真实 Docker 镜像、GPU 双卡隔离、真实 HelpSteer2 文件和长周期故障压力结果。因此当前只能称为“本地实现与单元测试完成”，不能称技术一实验或稳定性结论已经完成。

第一轮服务器验证应先用小数据和少量批次逐项通过：Task Model smoke、Decision LLM 提案 smoke、Docker 预检、无故障候选运行、单一故障注入下稳定分支完成。全部通过后冻结 commit、模型 ID/revision、profile 哈希、数据哈希、容器 digest、GPU 分配、种子、批次数、反馈延迟和资源预算，再运行多种子与长周期压力实验。
