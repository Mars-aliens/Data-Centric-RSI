# LEGO Server Handoff

## Local freeze

The local fixed baseline is frozen as a reproducible deterministic scaffold, not
as the final paper ReAct baseline.

External comparison candidates and their paper/code/license verification are
tracked in `baseline_matrix.json` and `baseline_research.md`. Do not schedule
external-method runs until their reproduction gates are resolved.

Server profile and Action System contract:

- `server_profile.example.json` is the no-secret configuration template.
- `server_action_contract.json` defines the Action System model adapter boundaries.
- The canonical profile block is `action_system`; older private profiles using
  `execution` remain readable as a compatibility alias and emit a deprecation warning.
- `server_preflight.py` validates data, hashes, split locks, methods, and
  unresolved model placeholders without running a model.
- `server_adapters.py` provides the JSON stdin/stdout bridge for the real
  decision policy, task-model trainer, and evaluator.
- `server_runner.py` executes one registered method through that bridge after
  preflight; it refuses unresolved placeholders and shell-string commands.

- Freeze manifest: `runs/real-data/lego-fixed-baseline-freeze.json`
- Reproduction runs: `runs/real-data/reproducibility/`
- Dataset: `data/helpsteer2-train.jsonl`
- Final-test data: `data/helpsteer2-validation.jsonl`
- Protocol: `data_protocol.json`
- Dataset source revision: `nvidia/HelpSteer2@990b2711a36180dd19d9c94b8627844866f8982a`
- Dataset license: `CC-BY-4.0`

The frozen local scaffold reports held-in accuracy `0.36393196598` and held-out
accuracy `0.36502463054`. These numbers are pipeline checks only.

## Server order

The first server deployment target is the frozen technology-one fixed control.
It is deliberately narrower than the later decision-LLM baseline matrix and
must be run before any final-test unlock:

1. Copy the source, normalized JSONL files, protocol, requirements, freeze
   manifest and the baseline profile. Do not copy credentials or local virtual
   environments.
2. Fill or review the private Qwen baseline profile. The profile scope must be
   `technology_one_baseline`, `final_test_unlocked` must remain `false`, and
   the Task Model path must point to the pinned local Qwen3-4B directory.
3. Run the baseline preflight and import checks.
4. Run one fixed-baseline pilot seed with
   `server_technical_one_baseline`.
5. Inspect `summary.json`, especially the train-group hash, selector
   truncation audit, held-in/held-out metrics, model provenance and budget.
6. Only after the pilot is accepted, run the fixed baseline for the paired
   seed set using `run_technology_one_baseline_batch.sh`.
7. Run the full deployable baseline/control/adaptation matrix with
   `run_baseline_matrix.sh` or `run_baseline_matrix_batch.sh`. External-paper
   rows must be reported as task adaptations unless a separate official
   reproduction is completed.
8. Run the final natural-support candidate matrix with exactly the same profile
   recipe, seeds, group budget and evaluation limits.
9. Keep `final_test` locked until both the fixed control and the candidate
   matrix are complete and the selection/statistical analysis is frozen.

The command for one baseline seed is:

```bash
export LEGO_ROOT="$(pwd)"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=5
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_BASE_PATH="$LEGO_ROOT/models/Qwen3-4B"
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=32
export LEGO_EVAL_BATCH_SIZE=16
export LEGO_DETERMINISTIC=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

"$TRAIN_PY" -m local_experiments.lego.server_technical_one_baseline \
  --profile "$LEGO_ROOT/server_profile.baseline.qwen.json" \
  --root "$LEGO_ROOT" \
  --output-dir "$LEGO_ROOT/runs/qwen-technology-one-fixed-baseline" \
  --run-id "qwen-technology-one-fixed-baseline-seed-17-$(date +%Y%m%d-%H%M%S)" \
  --seed 17 \
  --max-train-groups 3000 \
  --max-eval-groups 1000 \
  --train-group-budget 1000
```

The later full Action System comparison remains a separate stage:

1. Copy the source, normalized JSONL files, protocol, requirements, and freeze
   manifest. Do not copy credentials or local virtual environments.
2. Fill a private copy of `server_profile.example.json`; keep credentials in
   environment variables or the server secret manager.
3. Run the preflight profile check before loading models:

   ```powershell
   python -m local_experiments.lego.server_preflight `
     --profile server_profile.json `
     --root . `
     --report server/preflight.json
   ```

4. Create a clean environment and install the pinned base/research requirements.
5. Run import, CUDA, model-loading, and dataset-validation smoke checks.
6. Configure the real decision model and Task Model separately. The decision
   model remains Qwen/Qwen2.5-7B-Instruct through vLLM. The formal Task Model
   adapter is Qwen2.5-3B-Instruct with a pinned LoRA recipe and the
   `qwen_task_model_command` / `qwen_evaluator_command` pair. TF-IDF remains
   only as a lightweight mechanism check. Record model paths, revisions,
   training steps, LoRA configuration, GPU assignment, and checkpoint format
   in a server config revision.
   For the private profile, the relevant command and model fields are:

   ```json
   "task_model": {
     "model_id": "Qwen3-4B-preference-lora",
     "revision": "qwen-lora-preference-v1",
     "initial_checkpoint": "none",
     "training_config": {
       "base_model_path": "models/Qwen3-4B",
       "max_steps": 20,
       "max_length": 1024,
       "lora_r": 8,
       "lora_alpha": 16,
       "learning_rate": 0.0002,
       "gradient_accumulation_steps": 1
     }
   },
   "evaluator": {
     "name": "qwen_lora_preference_logits",
     "revision": "qwen-lora-preference-v1"
   },
   "commands": {
     "trainer": ["python3", "-m", "local_experiments.lego.qwen_task_model_command"],
     "evaluator": ["python3", "-m", "local_experiments.lego.qwen_evaluator_command"]
   }
   ```

   The path above is an example based on a known server model location; verify
   it with `test -f <path>/config.json` before running. Use a small
   `max_steps` only for the first smoke test, then freeze the chosen recipe
   before formal comparisons.
7. Run a small fixed Qwen Task Model smoke test first. Verify that the data split,
   batch order,
   seed, budget accounting, and audit-log schema match the local protocol.
8. Run the formal fixed baseline with the Qwen Task Model, then run ReAct + Reflexion,
   General Self-Harness, and the CurateEvo adaptation
   under the same model, data, and budget. Report each as an adaptation unless
   its original protocol is reproduced exactly.
9. Run independent edit, random joint edit, one-batch/three-batch selection,
   same-memory/no-process-update, and dependency-aware joint edit under the
   same model, data, and budget.
10. Unlock `final_test` only after every candidate-selection run is complete.
11. Run optional DataEvolver and HarnessEvolve adaptations only after the core
    matrix is complete and their code/license gates are resolved.

## Server acceptance checks

- `held_in` is the only candidate-selection split.
- `held_out` is report-only during candidate selection.
- `final_test` is report-only and explicitly unlocked.
- Every candidate has a parent harness version and an atomic accept/reject log.
- Rejected candidates do not change the active harness or checkpoint.
- Delayed feedback retains its source batch and release batch.
- All runs preserve dataset, harness, model, code, and checkpoint identifiers.
- The real server baseline is labeled separately from
  `deterministic-scaffold` until a real LLM-backed policy is implemented.
- The Action System is the server-side layer that exposes permitted actions,
  model/trainer/evaluator adapters, budgets, and audit requirements. It does
  not imply that every model execution is itself a Harness edit.
