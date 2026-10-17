# Contributing

This repository coordinates independently owned research contributions.
Keep project-specific code, tests, protocols, and dependencies inside the
owning project directory. Promote code to a shared top-level package only
when at least two projects use the same stable interface and both maintainers
agree to the change.

## Workflow

1. Open a GitHub Issue using the closest project/task template.
2. State the project owner, implementer, executor, expected artifacts, and
   acceptance criteria.
3. Work on a focused branch and keep the change within its project boundary.
4. Run that project's tests and protocol checks.
5. Open a pull request linking the issue; update project status documents only
   with reviewed evidence.

For a MLE-LEGO experiment, follow its contributor runbook. The code revision,
profile hash, data manifest, model revisions, seeds, budgets, and evidence
location must be recorded. Execution output alone is not a research conclusion.

## Research Integrity

- Do not change data splits, seeds, model revisions, budgets, acceptance rules,
  or metrics without recording a protocol revision.
- Keep raw datasets, model weights, credentials, private profiles, checkpoints,
  and bulky run outputs out of Git.
- Do not report a planned project or experiment as implemented or completed.
- Preserve failed runs and explain resumptions; do not overwrite evidence.

## Commits

Use concise imperative subjects with an optional project scope, for example:

```text
docs: define cross-project integration boundary
lego: validate dependency-closed proposals
aladdin: add reviewed experiment protocol
```
