# Aladdin Experiment Parameters

This document freezes the parameters used by the current Aladdin experiments.
The four sections below are separate protocols and must not be pooled into a
single result.

## 1. HelpSteer2 Technique One Replay

### Purpose

Test whether selecting feedback requests with disagreement, expected
downstream selection impact, and a random exploration reserve changes the
quality of selected training data under a fixed simulated feedback budget.
This is a historical replay of existing HelpSteer2 annotations, not a new
human-feedback study.

### Dataset and split protocol

| Parameter | Frozen value |
|---|---|
| Dataset | `nvidia/HelpSteer2` |
| Dataset revision | `990b2711a36180dd19d9c94b8627844866f8982a` |
| Input artifact | normalized `helpsteer2-train.jsonl` |
| Recorded local SHA-256 | `5bd88e03cb93527131d2a75b30940ea8a969a23096f8f529eb0e9bb114f4d6e9` |
| Group key | `source_group_id` |
| Used splits | `seed`, `held_in`, `held_out` |
| Final test | not loaded; `final_test_accessed=false` |
| Target profiles | `helpfulness-only`, `correctness-only`, `coherence-only` |
| Profile weights | one-hot over the corresponding HelpSteer2 attribute |
| Split seeds | `17, 23, 29, 37` |
| Policy seeds | `17, 18, 19` |
| Total conditions | `4 x 3 x 3 x 5 = 180` |

The target profile is used only for the replay's simulated target, seed
supervision, posterior labels, and held-out metric. It is not a user-provided
requirement and must not be described as a discovered user preference.

### Data and feedback budgets

| Parameter | Frozen value |
|---|---:|
| Seed group limit | 64 |
| Query-pool group limit | 256 |
| Selection-pool group limit | 512 |
| Held-out group limit | 512 |
| Feedback-time budget | 1,800 seconds |
| Maximum feedback queries | 48 |
| Selected training-group budget | 64 |
| Random reserve fraction | 0.25 |
| Tie margin | 0.0 |
| Review-cost proxy | 5 seconds + visible words / 220 words per minute |

The review-cost formula is only an uncalibrated proxy. It is not measured
human annotation time. Query, selection, and held-out groups are disjoint.
Unqueried selection-pool labels remain hidden.

### Compared strategies

| Strategy | Definition |
|---|---|
| `seed-only` | seed groups only; no feedback queries and no extra selected groups |
| `random` | random feedback requests under the common time/query budget |
| `disagreement-only` | requirement disagreement divided by estimated review cost |
| `impact-aware` | disagreement x expected selected-set change / review cost |
| `impact-aware-random` | impact-aware selection with a 25% random reserve; primary method |

### Metric and inference

- Primary metric: group-level pairwise accuracy on decisive held-out pairs.
- Ties are excluded from the accuracy denominator and reported separately.
- Primary comparisons: `impact-aware-random` minus `random`,
  `disagreement-only`, and `seed-only`.
- Pairing unit: split seed. Target profiles and policy seeds are averaged
  within each split before the paired comparison.
- Confidence interval: two-sided 95% paired t interval over the four split
  differences.
- Required audits: hidden query labels, hidden selection-pool labels, zero
  query/selection/held-out overlap, equal selection and feedback budgets, and
  zero final-test rows loaded.

### Reproduction command

Run from `projects/aladdin/` or the standalone Aladdin root. Keep data and
outputs outside Git.

```bash
export PYTHONPATH="$PWD"
export TRAIN_PY=python
export DATA_PATH=/external/data/helpsteer2-train.jsonl
export OUTPUT=/external/runs/aladdin-helpsteer2-technique-one.json

"$TRAIN_PY" -u -m local_experiments.aladdin.active_feedback_helpsteer2 \
  --data "$DATA_PATH" \
  --output "$OUTPUT" \
  --split-seeds 17 23 29 37 \
  --policy-seeds 17 18 19 \
  --target-profiles helpfulness-only correctness-only coherence-only \
  --seed-group-limit 64 \
  --query-group-limit 256 \
  --selection-group-limit 512 \
  --heldout-group-limit 512 \
  --feedback-time-budget-seconds 1800 \
  --max-queries 48 \
  --selection-budget 64 \
  --random-reserve-fraction 0.25 \
  --tie-margin 0.0
```

Analyze the completed report with the project analyzer specified in the
repository:

```bash
export ANALYSIS=/external/runs/aladdin-helpsteer2-technique-one-analysis.json
"$TRAIN_PY" -u -m local_experiments.aladdin.analyze_active_feedback_helpsteer2 \
  "$OUTPUT" \
  --output "$ANALYSIS"
```

Do not tune these values after inspecting held-out scores and then call the
same run confirmatory.

## 2. Qwen LoRA Downstream Validation

This protocol tests whether the selected groups from a frozen simulated-
feedback matrix improve a downstream preference classifier. It does not add
new feedback and does not validate an independent dataset.

| Parameter | Frozen value |
|---|---|
| Base/task model | `Qwen/Qwen2.5-7B-Instruct` |
| Target profile | `helpfulness-only` |
| Split seeds | `17, 23, 29, 37` |
| Policy seeds | `17, 18, 19` |
| Strategies | `seed-only`, `random`, `disagreement-only`, `impact-aware-random` |
| Conditions | `4 x 3 x 4 = 48` |
| Training steps | 150 |
| Per-device batch size | 2 |
| Gradient accumulation | 1 |
| Maximum sequence length | 1,024 |
| Learning rate | `2e-4` |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0.05 |
| Evaluation batch size | 16 |
| GPU memory cap used in the server run | 60 GiB |
| Evaluation groups | 512 held-out groups from the frozen matrix |
| Final test | not accessed |

All strategies use the same base model, training budget, evaluation groups,
and optimizer configuration. The output must record the matrix SHA-256,
dataset SHA-256, model revision, adapter hash, and `final_test_accessed`.

The matrix itself is an input artifact generated by the frozen feedback
protocol. It must be supplied with the experiment issue; do not regenerate it
with changed seeds or budgets during a confirmation run.

Example command:

```bash
export PYTHONPATH="$PWD"
export TRAIN_PY=/path/to/qwen-environment/bin/python
export DATA_PATH=/external/data/helpsteer2-seed-development-clean.jsonl
export MATRIX_PATH=/external/runs/technology-one-multiround-matrix.json
export BASE_MODEL=/external/models/Qwen2.5-7B-Instruct
export OUTPUT=/external/runs/aladdin-helpsteer2-qwen-lora-matrix.json
export CUDA_VISIBLE_DEVICES=0

"$TRAIN_PY" -u -m local_experiments.aladdin.helpsteer2_qwen_lora_matrix \
  --data "$DATA_PATH" \
  --matrix "$MATRIX_PATH" \
  --output "$OUTPUT" \
  --base-model "$BASE_MODEL" \
  --split-seeds 17 23 29 37 \
  --policy-seeds 17 18 19 \
  --target-profile helpfulness-only \
  --max-steps 150 \
  --batch-size 2 \
  --eval-batch-size 16 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --max-gpu-memory-gib 60
```

## 3. HH-RLHF Qwen Diagnostic

This is a separate downstream diagnostic for selector labels. It is not the
HelpSteer2 matrix and must be reported separately.

| Parameter | Formal/default value |
|---|---|
| Dataset | Anthropic HH-RLHF train artifact supplied externally |
| Split unit | prompt |
| Split seed | 17 |
| Strategy seeds | 17 through 26 |
| Strategies | `seed-only`, `random`, `aladdin-v2` by default; Qwen selector variants are explicit diagnostics |
| Candidate feedback budget | 32 groups |
| Seed/pool/evaluation limits | 64 / 256 / 512 groups |
| Training budget | 100 steps, `equal-epochs` mode |
| Batch / max length | 2 / 1,024 |
| Learning rate | `2e-4` |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Evaluation | held-out pairwise accuracy, batch size 8 by default |
| Feedback endpoint | OpenAI-compatible Qwen endpoint |
| Official test | sealed |

The diagnostic may compare LLM-simulated labels with dataset-oracle labels,
but oracle labels are analysis-only and are not an available feedback method.
The result supports a selector-label diagnostic, not a human-feedback claim.

## 4. Integrated Pipeline Engineering Validation

The integrated pipeline is an engineering validation of the broader Aladdin
loop, not a replacement for the frozen technique-one matrix.

| Parameter | Default |
|---|---:|
| Feedback mode in direct Python entry point | `interactive` |
| Feedback mode in server wrapper when unset | `offline-replay` |
| Strategies | `coverage`, `revision`, `contrastive` |
| Candidates per round | 4 |
| Feedback budget | 4 requests |
| Feedback time budget | 120 seconds |
| Generation token budget | 1,536 |
| Seed examples in generation context | 16 examples, 700 chars each |
| Selected-data limit | 8 |
| Random feedback fraction | 0.25 |
| Model seed | 17 |
| Rounds | 1 |
| Quality ensemble size | 1 |
| Endpoint timeout | 120 seconds |

Offline replay requires `seed`, `development`, and `held_in` records and uses
existing labels only after a selected query. Interactive and file modes are
the modes that can represent new human feedback. Final-test records are not
accepted by the configured round runner.

The pipeline's local TF-IDF trainer is a plumbing baseline. Its smoke metrics
must not be reported as evidence of generation-model improvement.

## Required report fields

Every server report must include:

- repository commit and protocol version;
- dataset path placeholder, SHA-256, and immutable dataset revision;
- model identifier/revision and GPU environment;
- split seeds, policy seeds, strategies, and all budget parameters;
- completed condition count and output path outside the repository;
- primary metric, paired differences, and confidence interval where applicable;
- leakage audit and explicit `final_test_accessed` value;
- whether feedback was historical replay, model-simulated, file-provided, or
  genuinely collected from a human.
