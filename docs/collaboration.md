# Collaboration Workflow

The current workflow has two people: the research/code maintainer and the
server experiment executor. The projects remain independent: Aladdin is in its
own local Git repository, while MLE-LEGO is prepared as a separate project
snapshot. Either person may review a change, but code changes and experiment
execution must be tied to the same recorded Git revision.

## Code Work

The maintainer opens an Issue describing the project, research question,
protocol constraints, expected code change, tests, and completion criteria.
The implementer works on a branch, runs local tests, and opens a pull request.
The maintainer reviews and merges the change before formal execution.

## Experiment Execution

The maintainer assigns an experiment Issue with an immutable commit or tag,
profile revision/hash, data manifest/hash, model revisions, seeds, budgets,
GPU assignment, output root, smoke-test command, formal command, and audit
criteria. The executor prepares the server, runs the smoke test, then runs only
the assigned experiment. Any protocol or code change requires a new reviewed
revision before continuing.

The executor attaches or links summaries, progress/state files, logs, preflight
and audit output, plus a checksum for the evidence archive. Large artifacts
stay in the approved artifact store, not GitHub. Use the report template owned
by the project being executed: MLE-LEGO's template is under
`projects/mle-lego/docs/`; Aladdin's template is in its own repository.

## Review and Progress

The maintainer checks that the run matches the assigned commit and protocol,
that the audit passes, and that final-test restrictions were respected. Only
then should the project status and umbrella README be updated. The executor
reports observations; research interpretation remains a reviewed project
decision. Cross-project integration requires a separate issue and an agreed
versioned interface.
