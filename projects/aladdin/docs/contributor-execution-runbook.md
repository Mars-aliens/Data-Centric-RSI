# Contributor Execution Runbook

This document is for a server contributor executing a frozen Aladdin
experiment.

## Before execution

```bash
git status --short
git rev-parse HEAD
export ALADDIN_ROOT="$PWD"
export PYTHONPATH="$ALADDIN_ROOT"
python -m pytest local_experiments/tests core_system/tests -q
python scripts/check_aladdin_install.py
```

Confirm that the dataset and model are outside the checkout, that the endpoint
serves the expected model, and that the selected GPU is free. Do not put
private paths or credentials into a committed profile.

## During execution

Use the exact command from the experiment issue. Write logs and outputs under
an external directory such as `$ALADDIN_RUNS`. A resumed run must use the same
run identifier and immutable input files. Do not read final-test labels before
the pre-registered final evaluation step.

## After execution

Check the output status, number of completed conditions, data hash, budget,
and `final_test_accessed` flag. Run the assigned analyzer and attach only
small, non-sensitive summaries to the pull request or issue. Keep raw outputs
on the server unless the owner explicitly approves a sanitized artifact.
