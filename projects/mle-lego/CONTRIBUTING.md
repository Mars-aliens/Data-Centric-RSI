# Contributing

## Scope

This repository is the Technology One implementation of the larger
Data-Centric RSI project. Keep changes focused on the Harness, its evaluation
protocol, reproducibility, and the associated documentation.

## Before opening a change

1. Create a branch named `feature/<short-name>`, `fix/<short-name>`, or
   `experiment/<short-name>`.
2. Run `python -m pytest -q` from the repository root.
3. Run shell syntax checks for changed scripts with `bash -n <script>`.
4. Update the relevant protocol or status document when behavior or evidence
   changes.

## Experiment changes

Keep the following fixed unless the change is explicitly a protocol revision:

- dataset split and source revision;
- seed set and group budgets;
- Decision LLM model and revision;
- held-in acceptance rule;
- held-out report-only rule;
- final-test lock;
- candidate training and evaluator budgets.

If one of these changes, record the reason, update the protocol, and use a new
output root. Never overwrite an old formal run to make it appear comparable.

## Contributor execution

Server execution must be assigned through a GitHub issue and performed from a
recorded commit. Contributors should follow
`docs/contributor-execution-runbook.md`, return the required evidence package,
and report facts without interpreting method superiority. The maintainer owns
the protocol, result interpretation, and updates to `docs/current-status.md`.

## Files that must stay out of Git

Do not commit credentials, private server profiles, datasets, model weights,
checkpoints, `runs/`, `logs/`, `tmp/`, `.venv/`, generated zip files, or local
editor/LibreOffice profiles. The repository `.gitignore` covers the common
cases, but inspect `git status` before committing.

## Commit guidance

Use short imperative commit subjects, for example:

```text
fix: reject incomplete dependency closures
test: cover joint proposal validation
docs: record formal comparison protocol
```
