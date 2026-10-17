# Technology Three: Continuous Feedback Selection

Status: planned; no method implementation or experimental result is claimed.

This package is reserved for methods that select and attribute delayed or
continuous feedback across long-running data-system executions.

Planned boundaries:

- feedback event identity, source-run and release-time tracking;
- attribution to the data, model, and Harness versions that caused the event;
- selection policy for incorporating feedback into later decisions;
- evaluation against fixed and immediate-feedback controls.

The module must reuse `core_system.mle_core` contracts and preserve the shared
held-in, held-out, and final-test separation.
