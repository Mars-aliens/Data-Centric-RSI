# Aladdin Core System

This directory contains the small action-system layer used by Aladdin. It
provides task records, artifacts, budgets, feedback records, provenance, and
the local demo/task-model interfaces. The Aladdin research strategies live in
`local_experiments/aladdin`, not in this lower-level package.

Run the dependency-light demo from the repository root:

```bash
export PYTHONPATH="$PWD"
python -m core_system.demo
```

Generated demo artifacts should remain outside Git. See the repository root
README for the Aladdin experiment protocol.
