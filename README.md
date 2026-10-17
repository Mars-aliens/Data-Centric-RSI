# Data-Centric RSI

Data-Centric RSI is the umbrella research effort on systems that improve data
construction and model development over repeated execution. It contains two
independent contributions, MLE-Aladdin and MLE-LEGO, with integration treated
as a later, explicit engineering step rather than assumed code sharing.

## Projects

| Project | Focus | Status in this repository |
|---|---|---|
| MLE-Aladdin | Data construction for generative ML under partially known requirements, including calibrated feedback and active feedback selection | Independent local Git repository; commit `6571f7d`; 102 tests pass; separate GitHub publication remains pending |
| MLE-LEGO | Self-improving data Harnesses: dependency-aware edits, experience reuse, and continuous feedback | Snapshot published under `projects/mle-lego/`; Technology One implementation; 56 tests pass; Technologies Two and Three remain planned |

These projects have separate source trees, protocols, dependencies, and
research claims. The Aladdin source is intentionally not copied into this
umbrella checkout; its independent repository remains the source of truth.

## Repository Structure

```text
Data-Centric-RSI/
├── projects/
│   └── mle-lego/                 # MLE-LEGO project snapshot
├── docs/
│   ├── architecture.md           # Project boundaries and integration rules
│   └── collaboration.md          # Two-person development and execution flow
├── .github/ISSUE_TEMPLATE/       # Root-level GitHub task templates
├── CONTRIBUTING.md               # Cross-project contribution rules
└── README.md
```

The Aladdin project directory is intentionally absent because its code is
maintained in an independent Git repository. Do not copy its source here as a
second source of truth. No root-level shared code package exists until both
projects have a concrete common interface.

## Start Here

- Read [Project Architecture](docs/architecture.md) before changing project
  boundaries or shared interfaces.
- Read [Collaboration Workflow](docs/collaboration.md) before assigning work
  between implementation and server execution.
- For MLE-LEGO setup, experiments, and current evidence, use
  [projects/mle-lego/README.md](projects/mle-lego/README.md).
- For the cross-project progress and pending publication/integration steps,
  see [Project Status](docs/project-status.md).
- For reproducible MLE-LEGO server runs, follow
  [projects/mle-lego/docs/contributor-execution-runbook.md](projects/mle-lego/docs/contributor-execution-runbook.md).

## Run MLE-LEGO Tests

```bash
cd projects/mle-lego
python -m pytest -q
```

Datasets, model weights, private profiles, checkpoints, and run outputs remain
outside Git. See the MLE-LEGO data and server handoff documentation for setup.

## Project Status

This umbrella checkout includes the MLE-LEGO snapshot. Aladdin is developed in
its separate local repository and is not copied here. The umbrella repository
is published at `Mars-aliens/Data-Centric-RSI`; current implementation and
experiment boundaries are tracked in
`docs/project-status.md`.
