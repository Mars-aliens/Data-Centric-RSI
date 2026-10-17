# Aladdin Integrated Pipeline Server Handoff

This handoff describes the Aladdin pipeline only. It is not a LEGO runner.

## Prerequisites

- Python with `requirements.txt` installed;
- an OpenAI-compatible vLLM endpoint for model-backed generation and quality
  judgments;
- a normalized JSONL input containing `seed`, `development`, and optionally
  `held_in` records;
- a requirements JSON file with requirement hypotheses;
- optional independently prepared feedback for `file` mode.

Keep final-test records in a separate sealed file. Existing labels in an
offline replay are not new human feedback.

## Preflight

```bash
export ALADDIN_ROOT="$PWD"
export PYTHONPATH="$ALADDIN_ROOT"
export TRAIN_PY=python
export DATA_PATH=/path/to/seed-development-held-in.jsonl
export REQUIREMENTS_PATH=/path/to/requirements.json
export ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export FEEDBACK_MODE=offline-replay
bash local_experiments/aladdin/integrated_pipeline/server_preflight.sh
```

## One round

```bash
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export OUTPUT_DIR=/path/to/external/aladdin-runs/round-001
export RUN_ID=aladdin-round-001
export ROUNDS=1
export CANDIDATE_COUNT=4
bash local_experiments/aladdin/integrated_pipeline/server_run_round.sh \
  | tee /path/to/external/aladdin-runs/round-001.log
```

Use `FEEDBACK_MODE=interactive` only for actual human review. Use
`FEEDBACK_MODE=file` only for an independently prepared feedback file. Set
`CALIBRATION_PATH`, `QUALITY_AUDIT_PATH`, or `RESUME_STATE` only when those
artifacts are part of the frozen experiment plan.

## Multi-round runs

Set `ROUNDS=3` and use the same external output directory when resuming from
`recursive-state.json`. Record the commit, input hashes, endpoint model,
feedback mode, query budget, and `final_test_accessed` flag in the run report.

This pipeline is an engineering protocol. A claim of data utility requires a
separate downstream training evaluation on held-out inputs.
