# Contributor Execution Runbook

This runbook defines how a contributor executes a Technology One experiment
from the GitHub repository and returns auditable evidence. The contributor
must not change the protocol, seeds, model revisions, acceptance rule, or data
splits unless the issue explicitly authorizes a protocol revision.

## Roles

- **Maintainer:** owns the research question, protocol, GitHub issue, and
  interpretation of results.
- **Contributor:** prepares the server, runs the assigned commands, preserves
  logs and outputs, and reports the exact result.
- **Reviewer:** checks the commit, profile hash, data manifest, audit output,
  and whether the run is comparable with the protocol.

## 1. Start From A GitHub Issue

Every run must have one issue containing:

- experiment name and purpose;
- Git commit or tag to execute;
- profile template and private profile revision;
- mode, seeds, batch count, GPU assignment, and output root;
- required smoke test and formal command;
- expected result files and acceptance criteria.

The contributor records the issue number in the run report. Do not run formal
experiments from an uncommitted working tree.

## 2. Prepare The Server

Clone or update the repository, then record the code revision:

```bash
git clone <repository-url>
cd <repository-directory>
git checkout <commit-or-tag>
git rev-parse HEAD
```

Prepare the private environment outside Git:

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

Copy the example profile to a private path, fill in local model and command
paths, and keep the private profile out of Git. Validate the normalized data,
profile, Decision LLM endpoint, and Task Model before starting a run.

The normalized files are external inputs, but the current install checker
expects them at these ignored paths inside the checkout:

```text
$LEGO_ROOT/data/helpsteer2-train.jsonl
$LEGO_ROOT/data/helpsteer2-validation.jsonl
```

Do not commit these files. Verify their source revision and SHA256 in the
issue before execution.

Run the preflight checks:

```bash
mkdir -p "$LEGO_ROOT/server"
"$TRAIN_PY" -m local_experiments.lego.server_preflight \
  --profile "$LEGO_ROOT/server_profile.evolution.qwen.json" \
  --root "$LEGO_ROOT" \
  --report "$LEGO_ROOT/server/preflight.json"

bash "$LEGO_ROOT/scripts/check_lego_install.sh"
```

The Decision LLM service must be tested independently:

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

Use the repository's `scripts/check_decision_model.sh` only when the server's
vLLM environment is the configured `vllm-serve` environment; otherwise record
the equivalent tokenizer, Transformers, vLLM, and CUDA checks in the issue.

## 3. Smoke Test

Run the assigned smoke command first. A smoke run is only a wiring check; it
does not provide paper evidence. Confirm that:

- `/v1/models` exposes the pinned Decision LLM model;
- one policy command returns valid JSON;
- the Task Model and evaluator can load;
- `final_test_used` is `false`;
- the run writes `progress.json`, `summary.json`, and an audit trail.

If the smoke test fails, stop and report the command, exit status, relevant log
tail, GPU assignment, and profile/model revisions. Do not silently edit the
protocol to make the run pass.

The smoke command is:

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/<experiment-id>-smoke" \
GPU_ID=<gpu-id> BATCHES=2 SEEDS="17" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

## 4. Formal Execution

Use the command specified by the issue. For the Technology One comparison,
the formal modes are `independent` and `dependency-aware-joint`:

```bash
OUTPUT_ROOT="$LEGO_ROOT/runs/<experiment-id>" \
GPU_ID=<gpu-id> \
BATCHES=<batches> \
SEEDS="17 23 29 31 37 41 43 47 53 59" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

Run modes in parallel only when the assigned GPUs and memory are isolated.
Otherwise run them serially. Never reuse an output root for a changed profile,
code revision, model revision, seed set, or budget.

During execution, monitor `progress.json` and the log. A high rejection rate
is not by itself a failure; report its breakdown by `held_in_regression`,
`duplicate_configuration`, invalid proposal, and execution failure.

## 5. Required Evidence Package

After completion, return a compressed evidence package outside the source
repository containing:

- `summary.json` for every completed run;
- `progress.json` and `endurance_state.json`;
- run logs and the final audit output;
- code commit, profile hash, protocol hash, data manifest/hash;
- model IDs and revisions for Task Model and Decision LLM;
- GPU, seed, batch, training, evaluator, and retry settings;
- a short `run-report.md` using the template below.

Do not commit raw datasets, model weights, checkpoints, credentials, logs, or
`runs/` into the public repository. Upload large evidence to the agreed
artifact store and place only its URL, checksum, and summary in the issue.

## 6. Run Report Template

Use [docs/run-report-template.md](run-report-template.md) for the complete
report fields.

The contributor should report facts only. The maintainer decides whether the
result supports a research claim.

## 7. Completion And Review

The run is complete only when all requested summaries exist, the audit passes,
`final_test_used` matches the protocol, and the evidence package is linked in
the issue. The maintainer then updates `docs/current-status.md` and the
experiment summary in a separate documentation commit.

If a run crashes, preserve the failed output and logs. If resume is supported,
resume only with the same code, profile, data, model, seed, and output root;
otherwise start a new output root and explain why in the issue.
