# Contributing to MLE-Aladdin

Aladdin and MLE-LEGO have separate repositories, namespaces, experiment
plans, and result directories. Do not add LEGO code or LEGO run artifacts to
this repository.

## Code changes

- Use relative paths or environment variables for data, models, and outputs.
- Keep final-test loading behind an explicit, audited protocol.
- Add or update a focused test for every protocol or selection change.
- Do not commit datasets, model weights, checkpoints, logs, server profiles,
  credentials, or generated runs.
- Preserve the existing dataset hash, seed, budget, and leakage metadata in
  experiment outputs.

## Execution contributor workflow

The experiment owner freezes a commit and gives the contributor an issue with
the dataset path, model path, endpoint, GPU assignment, command, and expected
output location. The contributor must not silently change those values.

```bash
git checkout <frozen-commit>
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python -m pytest local_experiments/tests core_system/tests -q
python scripts/check_aladdin_install.py
```

Run the assigned command, save logs outside the repository, and report:

- exact commit and environment information;
- exact command and external paths represented by placeholders;
- status and completed condition count;
- dataset/model hashes or immutable revisions;
- final-test access flag;
- output JSON path and a concise interpretation.

The report template is `docs/run-report-template.md`.
