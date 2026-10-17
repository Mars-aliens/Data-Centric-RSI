# Current Status

Updated 2026-10-16.

## Implementation

The local Technology One implementation is complete for the current protocol:

- strict `dependency-aware-joint` proposal validation;
- `independent` versus dependency-closed joint editing;
- candidate evaluation, held-in acceptance, rejection, rollback, checkpoint,
  and resume;
- baseline runners, component matrix, batch execution, audits, and tests.

## Completed evidence

- The ten-seed component matrix completed under the frozen HelpSteer2/Qwen
  protocol. The selector+filter arm was the strongest observed component
  combination in that matrix, with held-out mean delta `+0.0404` and 8/10
  paired wins.
- A 60-batch endurance run completed with `60/60` batches, no command retries,
  and no final-test use. This is execution-stability evidence only.
- The latest joint smoke completed both modes for seed 17 and two batches. The
  joint mode produced a valid `generator + selector` proposal. Both candidates
  were rejected because held-in performance regressed.

## Not yet claimed

- The smoke run does not establish method superiority.
- The latest strict joint mode still needs the formal ten-seed comparison, with
  at least one joint proposal audited per seed.
- Long-horizon 100/300-batch joint validation is separate from the smoke and
  ten-seed comparison.
- Final test remains locked until the protocol and statistical analysis are
  frozen.

## Evidence locations

Run outputs are intentionally excluded from this source package. On the
execution server, preserve them under a versioned `runs/` root and record the
root, profile hash, code revision, model revision, seeds, and budgets in the
experiment report.
