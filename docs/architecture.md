# Project Architecture

## Research Scope

Data-Centric RSI is the umbrella research effort. Aladdin and MLE-LEGO are
separate contributions under that effort, not interchangeable names for the
same implementation. Each contribution owns its code, environment, data
protocol, experiments, and claims until a specific integration contract is
agreed.

## Current Repository Boundary

`projects/mle-lego/` contains the prepared MLE-LEGO snapshot. MLE-Aladdin is
maintained in an independent Git repository and is not vendored here. The
umbrella repository is for cross-project status, protocols, and coordination;
it must not silently become a second copy of Aladdin source.

MLE-LEGO internally has a shared runtime, three technology module boundaries,
and an integration boundary. Technology One is implemented; Technologies Two
and Three are planned. There is no root-level shared code package yet.

## Integration Rules

- Keep dependencies project-local by default.
- Share an interface only after both projects identify a concrete use case,
  owner, versioning rule, and compatibility tests.
- Integrate through explicit APIs and data contracts; do not import another
  project's private modules or experiment outputs.
- Preserve provenance across project boundaries: code revision, data
  manifest, model revision, configuration, and artifact identifiers.
- Keep each contribution independently installable and testable.

## Repository Growth

If the team later chooses to make this umbrella repository the canonical
monorepo, integrate Aladdin as a Git subtree or a reviewed source migration,
not as an untracked copy. Until then, link to its canonical remote once one is
configured. If common code emerges, extract it to a root `shared/` package only
through a reviewed change with migration notes and tests for both projects.
