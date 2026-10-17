# Data-Centric RSI Project Status

Updated 2026-10-17. This document summarizes the two independent contributions
and the umbrella project's integration state. It does not combine their
datasets, environments, protocols, or source histories.

## MLE-Aladdin

- **Scope:** data construction for generative ML with partially known
  requirements; requirement-state updates, calibrated human/model feedback,
  active feedback selection, dataset adapters, data synthesis/selection, and
  downstream utility evaluation.
- **Repository:** independent local Git repository; its GitHub remote has not
  been configured yet.
- **Revision:** local commit `6571f7d`; working tree was clean when checked.
- **Implementation/package work:** Aladdin-only code and Qwen public training
  support in the Aladdin namespace; README, contribution/server guides, report
  template, dependencies, audit and packaging scripts are present.
- **Verification:** `102 passed` in the local test run; repository audit was
  reported as passing. This verifies package/tests, not a scientific
  superiority claim.
- **Publication:** no GitHub remote or push is configured yet.
- **Evidence boundary:** experiments support individual mechanisms and are not
  official reproductions of external papers. Final-test labels remain sealed
  until the registered final evaluation.

## MLE-LEGO

- **Scope:** improve the data-processing Harness itself through execution,
  feedback/reflection, bounded edits, candidate evaluation, acceptance or
  rollback, and continued execution.
- **Repository snapshot:** `projects/mle-lego/`; a standalone prepared project
  tree also exists locally. The two trees were byte-for-byte identical when
  checked, but the standalone tree has no `.git` history yet.
- **Implementation:** Technology One includes independent and strict
  dependency-aware joint edits, proposal validation, candidate evaluation,
  held-in acceptance, held-out reporting, rollback, checkpoint/resume, server
  runners, audit scripts, and tests. Technology Two (executable experience
  reuse) and Technology Three (continuous feedback selection) have documented
  module boundaries but are not implemented.
- **Verification:** `56 passed` in the local test run. The final package also
  verified both the new `technologies.technology_one` import and the legacy
  `local_experiments.lego` server import.
- **Completed evidence recorded by the project:** ten-seed component matrix;
  selector+filter held-out mean delta `+0.0404` with `8/10` paired wins; a
  60-batch endurance run; and two-batch joint smoke runs. These support only
  their stated component, execution-stability, and wiring claims.
- **Not established:** strict dependency-aware-joint formal ten-seed
  comparison, per-seed joint-proposal coverage, final method superiority, and
  long-horizon 100/300-batch joint validation.
- **Publication:** the prepared snapshot is published through the umbrella
  repository at `Mars-aliens/Data-Centric-RSI` on `main`; the standalone tree
  has no separate Git history yet.

## Umbrella Project

- The umbrella coordinates the two projects but does not merge their code or
  claim one project's test results for the other.
- MLE-Aladdin remains the source of truth in its independent Git repository;
  the umbrella checkout intentionally has no `projects/aladdin/` copy.
- The umbrella checkout contains an MLE-LEGO snapshot under
  `projects/mle-lego/`. It currently duplicates the standalone prepared tree;
  designate one canonical editable source before accepting changes in both.
- **Publication:** the umbrella `main` branch is published at
  `Mars-aliens/Data-Centric-RSI`; the initial repository commit is `f662d5c`.
- No shared code has been extracted. Define and test a common interface only
  after both projects have a concrete integration requirement.

## Immediate Coordination Tasks

1. Configure and verify the Aladdin GitHub remote, then push its existing
   history. For MLE-LEGO, choose whether the standalone tree or the published
   umbrella snapshot is canonical; the standalone tree currently has no Git
   history to push.
2. Decide whether the umbrella is documentation-only, uses Git submodules, or
   becomes a monorepo. Do not maintain duplicate editable source trees without
   an explicit source-of-truth rule.
3. Continue each project's experiments from its own frozen commit and
   protocol; link reviewed summaries from this status document.
4. Open a separate integration issue when concrete cross-project APIs and
   shared evaluation requirements are known.
