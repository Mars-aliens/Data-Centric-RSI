# MLE-Aladdin

Aladdin is the data-centric RSI research project for generative ML tasks
whose requirements and quality judgments are only partially known at the
start. This repository contains only the Aladdin implementation. MLE-LEGO is
a separate project and is intentionally not included here.

## Research scope

Aladdin studies how a data pipeline can turn a small seed set and limited
feedback into better training data. The repository covers:

- requirement hypotheses and requirement-state updates;
- calibrated human/model feedback protocols;
- active feedback selection under a fixed query budget;
- HelpSteer2 and HH-RLHF adapters with leakage checks;
- data synthesis, selection, and downstream utility evaluation;
- frozen baselines and paired strategy comparisons;
- an integrated multi-round pipeline that can use a vLLM-compatible endpoint.

The current experiments are evidence for individual mechanisms. They are not
an official reproduction of any external paper and must not be reported as
one. Dataset files, model weights, checkpoints, server profiles, credentials,
and run outputs stay outside Git.

## Repository layout

```text
Aladdin/
  core_system/                 shared action-system primitives used by Aladdin
  local_experiments/aladdin/   Aladdin adapters, protocols, runners, analyzers
  local_experiments/tests/     unit and protocol tests
  docs/                        experiment and contributor documents
  scripts/                     repository checks and packaging utilities
  requirements.txt             CPU test and baseline dependencies
  requirements-qwen.txt        optional Qwen/LoRA dependencies
```

## Quick verification

From the repository root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python -m pytest local_experiments/tests core_system/tests -q
python -m compileall -q core_system local_experiments
python scripts/check_aladdin_install.py
```

The tests use synthetic fixtures and do not need a GPU, dataset download, or
model endpoint.

## Integrated pipeline smoke test

The integrated pipeline can run offline against a prepared JSONL replay. The
data and requirements files are supplied by the operator:

```bash
export PYTHONPATH="$PWD"
export TRAIN_PY=python
export DATA_PATH=/path/to/seed-development-held-in.jsonl
export REQUIREMENTS_PATH=/path/to/requirements.json
export FEEDBACK_MODE=offline-replay
bash local_experiments/aladdin/integrated_pipeline/server_preflight.sh
```

For a real endpoint-backed round:

```bash
export ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_DIR=/path/to/runs/aladdin-round-001
export RUN_ID=aladdin-round-001
bash local_experiments/aladdin/integrated_pipeline/server_run_round.sh
```

Use `FEEDBACK_MODE=interactive` only when a human is actually reviewing the
shown candidates. Use `FEEDBACK_MODE=file` only for an independently prepared
feedback file. Existing dataset labels in `offline-replay` are not new human
feedback.

The complete server handoff is in
`local_experiments/aladdin/integrated_pipeline/SERVER_HANDOFF.md`.

The frozen values for the HelpSteer2 replay, Qwen downstream validation,
HH-RLHF diagnostic, and integrated pipeline are listed in
`docs/experiment-parameters.md`. Do not combine their metrics into one
experiment result.

## Active feedback and downstream validation

The HelpSteer2 and HH-RLHF runners accept operator-provided dataset paths and
write results to an external output directory. The protocol keeps seed,
development, held-in, held-out, and final-test data separated. Final-test
labels must remain sealed until the pre-registered final evaluation.

For Qwen LoRA validation, use an environment with
`requirements-qwen.txt`, set `ALADDIN_TASK_MODEL_BASE_PATH`, and point the
runner at a local model snapshot. The Qwen adapter is implemented inside
`local_experiments/aladdin/qwen_preference.py`; it has no dependency on the
separate LEGO package namespace.

## Reproducibility rules

Every reported run should record the repository commit, dataset SHA-256,
model identifier/revision, split seeds, policy seeds, feedback budget,
training budget, endpoint model name, and whether final-test was accessed.
Never commit raw user feedback, private server profiles, model weights, or
large run artifacts.

## Collaboration

The GitHub repository is the source of truth for protocol and code changes.
An execution contributor should:

1. checkout the exact commit named in the issue;
2. run the listed preflight and smoke commands;
3. run only the assigned experiment with external data/model paths;
4. return the commit, commands, logs, output manifest, hashes, and summary;
5. open a pull request containing no data, checkpoints, secrets, or generated
   run directories.

See `CONTRIBUTING.md`, `docs/contributor-execution-runbook.md`, and
`docs/run-report-template.md`.
