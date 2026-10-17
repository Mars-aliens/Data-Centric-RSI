# Integrated Aladdin Pipeline Prototype

This new path is deliberately separate from the archived data-selection study.
It composes small, replaceable components to smoke-test the intended loop:

```text
seed cases
  -> validate and exact-deduplicate
  -> generate candidate data
  -> quality gate (accept / defer / reject)
  -> ask feedback where requirement hypotheses disagree and could change the
     downstream selected set, divided by review effort; reserve random queries
  -> update hypothesis weights from feedback and retain direct human edits
  -> optionally compare two candidates and convert the preference into
     record-level training decisions
  -> select accepted labeled data with TF-IDF maximal marginal relevance
  -> compare seed-only and augmented training under the same trainer
  -> evaluate on development and allocate future synthesis using decayed UCB1
  -> persist requirement, evaluator, calibration, and allocation state
```

## Method choices and limits

- Cleaning uses deterministic validation and exact deduplication. High TF-IDF
  cosine pairs are flagged for review, not deleted automatically.
- The query score is a project-specific policy based on requirement disagreement,
  counterfactual symmetric difference in the downstream top-k set, and estimated
  human effort. It is a testable hypothesis, not an established winning method.
- Quality decisions require a score and confidence threshold. This prototype
  additionally certifies automatic accept and reject separately for each task
  slice using an independent quality-audit channel and a Wilson agreement lower
  bound. Training acceptance feedback is never reused as quality evidence.
  Evidence below the configured count or agreement threshold is routed to
  human review (`defer`). The fixture evidence in `demo.py` only exercises this
  control path; real thresholds still require held-back human judgments.
- Candidate generation and requirement/quality scoring are injected interfaces.
  `demo.py` uses deterministic fixtures that do not interpret natural-language
  requirements or generate real examples.
- The synthesis library has distinct `coverage`, `revision`, and `contrastive`
  objectives. The LLM adapter places the selected objective and required
  evidence in the generation prompt; it does not treat the strategy name alone
  as evidence of a different method.
- Pairwise feedback is represented separately from binary acceptance. The
  pairwise selector and expansion helper preserve the preferred record and
  review cost; the default smoke path remains single-record feedback so the
  two protocols are not silently mixed.
- Multiple quality scorers can be combined into a conservative ensemble.
  Score disagreement lowers confidence, and independent audit disagreements are
  retained as versioned evaluator prompt notes for later rounds.
- TF-IDF MMR uses the established maximal marginal relevance recipe
  (Carbonell and Goldstein, 1998, <https://doi.org/10.1145/290941.291025>).
- Synthesis strategy allocation uses a bounded, cost-normalized UCB1 reward
  (development accuracy gain is in `[-1, 1]`, generation cost is at least one
  token/call/example), following UCB1 (Auer et al., 2002,
  <https://www.jmlr.org/papers/v3/auer02a.html>) on development gain per
  measured generation cost. Actual human review seconds are checked against the
  declared query-time budget. A smoke run cannot validate regret or benefit.
- Downstream utility is measured on `development`, not held-out/final test.
  The local TF-IDF classifier is a plumbing baseline, not the target model.

## Run locally

From the workspace root with the project environment active:

```powershell
python -m local_experiments.aladdin.integrated_pipeline.demo `
  --output-dir tmp\aladdin-integrated-smoke
python -m pytest local_experiments\tests\test_integrated_aladdin_pipeline.py -q
```

The report marks itself as a smoke test. Do not report its metrics as evidence
that the method improves a real downstream task. The next integration stage
uses the configured adapters below; it still needs real independent human
judgments before its automatic quality actions can be certified.

## Configured vLLM/OpenAI-compatible round

`adapters.py` provides a real JSONL loader, deterministic JSON parsing for an
OpenAI-compatible endpoint, candidate generation with seed-only lineage,
requirement and quality scorers, calibration-evidence loading, and explicit
file or console human feedback. Model-scored quality is not treated as human
feedback, and training feedback is not treated as quality-audit evidence.
Generated records retain their seed parent and source group.

Run preflight first. In the default `offline-replay` protocol the input must
contain `seed`, `development`, and `held_in` rows; the final test is
deliberately unavailable to this command. The `held_in` labels are hidden
until a selected feedback query releases them.

Minimal input and configuration examples are under `examples/`. The empty
feedback and audit files are schema examples only; they do not provide human
evidence and should not be treated as calibration.

```powershell
python -m local_experiments.aladdin.integrated_pipeline.preflight `
  --data path\train-development.jsonl `
  --requirements path\requirements.json `
  --calibration path\calibration.json `
  --endpoint http://127.0.0.1:8000/v1/chat/completions
```

Then run one development round with a new empty output directory. Human review
must be supplied explicitly, either interactively or from an independently
prepared feedback file.

```powershell
python -m local_experiments.aladdin.integrated_pipeline.run_configured `
  --data path\train-development.jsonl `
  --requirements path\requirements.json `
  --calibration path\calibration.json `
  --interactive-feedback `
  --output-dir runs\integrated-round-001 `
  --run-id integrated-round-001 `
  --endpoint http://127.0.0.1:8000/v1/chat/completions `
  --model Qwen/Qwen2.5-7B-Instruct
```

Use `--quality-ensemble-size N` to run N independently prompted quality
scorers. Use `--rounds N` with `--interactive-feedback` for multiple rounds;
the runner writes `recursive-state.json` and `recursive-manifest.json` after
each completed round. `recursive.py` also provides a validated state loader for
resumption. A final-test protocol must still be frozen separately before
claiming recursive improvement.
