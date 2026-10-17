# MLE-LEGO

MLE-LEGO is a modular research project for Data-Centric Recursive Self-Improvement
(RSI). It studies how a data system can improve its data-processing Harness
through execution feedback, reusable experience, and long-running evaluation.
The current implementation is concentrated in Technology One; Technologies Two
and Three and the integrated workflow are reserved in the project structure and
are not yet implemented.

The repository is intentionally limited to the LEGO contribution. It does not
contain Aladdin experiments, model weights, private datasets, GPU environments,
credentials, or previous run outputs.

## Project Architecture

```text
code/
├── core_system/mle_core/          shared interfaces, runtime, data and budget
├── technologies/
│   ├── technology_one/             implemented Harness evolution
│   ├── technology_two/             executable experience reuse (planned)
│   └── technology_three/           continuous feedback selection (planned)
├── integration/                    cross-technology orchestration (planned)
└── local_experiments/lego/         compatibility import for existing commands
experiments/                        protocols, manifests and result indexes
docs/                               research, server and contributor guidance
scripts/                            launch, audit and reporting entry points
data/                               instructions only; datasets stay external
```

The Python import `local_experiments.lego` remains supported for existing
profiles and server commands. New code should import from
`technologies.technology_one`.

## Research Scope

The Harness organizes data generation, selection, filtering, training, tool
calls, feedback, and reusable experience. The core loop is:

```text
execute -> observe feedback -> propose Harness edit -> evaluate candidate
-> accept on held-in improvement or roll back -> continue
```

Technology One compares independent component edits with dependency-closed
joint edits. The current formal joint mode is named `dependency-aware-joint`:
it requires a valid multi-component proposal such as `generator + selector`.
The decision model is held fixed; the executable Harness and its accepted
versions are the objects that may change.

## Current implementation

- bounded JSON proposal protocol with strict validation;
- `independent` and `dependency-aware-joint` evolution modes;
- candidate branches, held-in acceptance, held-out reporting, rollback;
- Harness versions, provenance, cost accounting, checkpoints, and resume;
- fixed ReAct and ReAct+Reflexion control runners;
- deterministic component-interaction matrix;
- batch runners, audits, summaries, and local integration tests.

## Evidence boundary

The latest joint smoke run completed successfully at the execution level:

- `independent`, seed 17, 2 batches: generator proposal rejected for
  `held_in_regression`;
- `dependency-aware-joint`, seed 17, 2 batches: generator+selector proposal
  rejected for `held_in_regression`;
- both runs completed with status `0` and final test remained locked.

This confirms that the joint proposal, candidate evaluation, and rollback path
are executable. It is not evidence that the method improves average model
quality. Formal multi-seed comparisons and long-horizon joint validation remain
separate experiments.

Earlier evidence recorded in `docs/current-status.md` includes the completed
component matrix and the 60-batch endurance run. Those results support the
corresponding claims only; endurance is execution-stability evidence, not
proof of sustained performance improvement.

## Where To Start

The project is organized around these entry documents:

- **README.md:** installation and experiment entry point.
- **Server Runbook:** [docs/server-handoff.md](docs/server-handoff.md) contains
  server preparation, profiles, model adapters, and execution order.
- **Collaboration Guide:**
  [docs/contributor-execution-runbook.md](docs/contributor-execution-runbook.md)
  defines the contributor workflow, evidence package, and review handoff.
- **Issue Template:**
  [.github/ISSUE_TEMPLATE/experiment-execution.yml](.github/ISSUE_TEMPLATE/experiment-execution.yml)
  assigns a reproducible experiment to a contributor.
- **Report Template:**
  [docs/run-report-template.md](docs/run-report-template.md) records the
  execution result and evidence links.
- **Research Protocol:**
  [docs/technical-one-protocol.md](docs/technical-one-protocol.md) defines
  the research question, experiment arms, controls, metrics, and claim
  boundaries.

## Local verification

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The local tests exercise protocol validation, proposal constraints, reporting,
rollback, and fake-command integration. They do not require a GPU or a live
decision-LLM endpoint.

## Server setup

The server uses an existing Python environment for the Task Model and an
OpenAI-compatible vLLM endpoint for the Decision LLM. Keep private profiles,
datasets, model paths, logs, and results outside Git.

```bash
export LEGO_ROOT="$(pwd)"
export TRAIN_PY="$LEGO_ROOT/.venv/bin/python"
export PYTHONPATH="$LEGO_ROOT/code:${PYTHONPATH:-}"

cp "$LEGO_ROOT/code/technologies/technology_one/server_profile.evolution.qwen.example.json" \
   "$LEGO_ROOT/server_profile.evolution.qwen.json"

export LEGO_POLICY_API_URL="http://127.0.0.1:8000/v1/chat/completions"
bash "$LEGO_ROOT/scripts/check_lego_install.sh"
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
  bash "$LEGO_ROOT/scripts/check_decision_model.sh"
```

Run a two-batch smoke test before formal experiments:

```bash
export CUDA_VISIBLE_DEVICES=0
export LEGO_TASK_MODEL_DEVICE=cuda
export LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB=40

OUTPUT_ROOT="$LEGO_ROOT/runs/technology-one-joint-smoke" \
GPU_ID=0 BATCHES=2 SEEDS="17" \
MODES="independent dependency-aware-joint" \
bash "$LEGO_ROOT/scripts/run_technology_one_evolution_comparison_batch.sh"
```

The formal experiment protocol and the required controls are documented in
`docs/technical-one-protocol.md`. Do not unlock `final_test` before the
protocol, statistics plan, and evidence audit are frozen.

## Collaboration

Read `CONTRIBUTING.md` before changing the protocol or experiment runners.
Changes to proposal validation, acceptance rules, data splits, budget limits,
or model revisions require a protocol note and a new smoke test. Do not commit
private profiles, credentials, datasets, model checkpoints, `runs/`, `logs/`,
temporary files, or generated archives.

For server-based collaboration, use
`docs/contributor-execution-runbook.md`. It defines the GitHub issue handoff,
fixed commit/profile requirements, smoke test, formal execution, evidence
package, and result-report format. The repository also includes a GitHub Issue
template under `.github/ISSUE_TEMPLATE/experiment-execution.yml` for assigning
reproducible runs to contributors.

## License

No license is asserted in this package. Choose and add a license before public
redistribution.
