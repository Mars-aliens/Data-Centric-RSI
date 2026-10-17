# Integrated Data-Centric RSI Workflow

Status: planned; the end-to-end integration is not implemented yet.

This package is the stable integration boundary for composing the three
technology modules over the shared `core_system.mle_core` runtime. Technology
modules must remain independently testable and expose versioned contracts;
integration code owns cross-module orchestration, configuration, and audit
provenance rather than reimplementing each technology.
