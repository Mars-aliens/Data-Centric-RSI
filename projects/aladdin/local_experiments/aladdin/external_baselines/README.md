# External Baseline Handoff

This directory defines the server handoff contract for external methods named in
`idea/MLE-Aladdin.md`. It does not claim to implement the papers locally.

The official-data catalog is `dataset-registry.json`. It records the source URL,
paper URL, immutable revision when available, source-group fields, feedback
fields, license risk, and whether an entry is human feedback. A server run must
resolve every `null` revision and finish any `terms-review-required` entry
before downloading or training.

## Local dry-run

The dry-run validates that each method declares the same task identity, split
order, initial checkpoint, seeds, budget fields, and final-test lock:

```powershell
python -m local_experiments.aladdin.external_baselines
```

The default config is `server-configs.local.json`, which intentionally uses the
local toy task and `local-protocol-only` as its commit marker. The output is an
execution plan only; it does not call an LLM or train a model.

## Server configuration

Copy the shape of `server-configs.json` and replace every placeholder with:

- the real dataset manifest and immutable dataset revision;
- the official repository commit;
- the initial task-model checkpoint;
- the evaluator and model identifiers;
- matched seed, feedback, training, evaluator, and wall-clock budgets.

Keep `final_test_unlocked` false until every method and hyperparameter is frozen.
Unlocking final test belongs to a separate report-only run.

## Adapter boundary

Each adapter returns a validated plan with the audit fields required by the
shared core. The server implementation must add the official repository call,
model runner, data conversion, and artifact registration behind that boundary.
The external method must remain in this directory; it must not be embedded in
`core_system`.
