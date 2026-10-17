# Impact-Aware Active Feedback: Local Protocol

This `active-feedback-impact-local-v2` prototype operationalizes the intended technique-one claim before adding
Qwen or human annotation infrastructure. It is a protocol test and debugging
benchmark, not evidence about HH-RLHF performance.

## Research question

Under the same feedback-cost budget, does selecting a query by its expected
effect on future data selection outperform random queries and disagreement-only
queries on a held-out downstream task?

## Operational definition

The local benchmark maintains several candidate requirement hypotheses. Each
hypothesis predicts a label and a utility score for every candidate. The
experiment has three disjoint pools: a query pool, a downstream selection
pool, and a held-out evaluation pool. For a query candidate `x`:

```text
disagreement(x) = label disagreement across requirement hypotheses
impact(x) = expected symmetric difference between current and counterfactual top-k sets
value(x) = disagreement(x) * impact(x) / query_cost(x)
```

The counterfactual sets are computed on the separate downstream selection pool,
using each possible feedback answer and the corresponding posterior over
requirement hypotheses. The hidden benchmark label is never used by the
selector. After a query, the observed feedback updates the posterior and the
next round recomputes the values. Queried items are not automatically added to
the downstream training set.

## Strategies

- `random`: random queries under the cost budget.
- `disagreement-only`: high disagreement per unit cost.
- `impact-aware`: disagreement multiplied by expected downstream selection change.
- `impact-aware-random`: 25% of the cost budget reserved for random queries, with
  the remainder allocated by impact-aware value.

All strategies share the same generated benchmark, seed, candidate pool,
rounds, per-round cost budget, and maximum selected candidates. A strategy may
select a different number of candidates because query costs differ, but no
strategy may exceed the same cost budget.

The report includes both total cost and cost utilization. A strategy that
spends less than the cap is not described as better merely because it used
fewer queries; its downstream metric is reported alongside realized cost.

## Local run

```powershell
python -m local_experiments.aladdin.active_feedback_impact `
  --output local_experiments\aladdin\runs\active-feedback-impact-local.json `
  --seeds 17 18 19 20 21 22 23 24 25 26 `
  --rounds 2 `
  --cost-budget-per-round 12 `
  --selection-budget 12 `
  --random-reserve-fraction 0.25
```

The report records per-round queried IDs, query costs, disagreement,
counterfactual affected-selection counts, the resulting downstream selected
IDs, posterior updates, and held-out nearest-neighbour accuracy. Each run also
stores the IDs of the seed, query, downstream-selection, and held-out pools,
plus pairwise overlap counts, so pool isolation can be audited from the report.
It records that held-out labels were not used for selection, paired accuracy
deltas against random, and realized cost utilization.

## Current local diagnostic

The v2 implementation passed the focused tests (6 tests) and the isolated
local test suite (62 tests). The 10-seed diagnostic used one round, a cost cap
of 4 per run, a maximum of 4 queries, and a 25% random reserve. Mean held-out
KNN accuracy was:

| Strategy | Mean accuracy | Mean cost | Mean queries | Mean cost utilization |
|---|---:|---:|---:|---:|
| `random` | 66.42% | 3.58 | 3.00 | 89.4% |
| `disagreement-only` | 67.75% | 4.00 | 4.00 | 100.0% |
| `impact-aware` | 66.04% | 4.00 | 4.00 | 100.0% |
| `impact-aware-random` | 66.87% | 3.58 | 3.40 | 89.4% |

Paired against random over the same 10 synthetic seeds, the mean accuracy
deltas were `+1.33 pp` for disagreement-only, `-0.38 pp` for impact-aware, and
`+0.46 pp` for impact-aware-random. These are mechanism diagnostics only, not
confirmatory estimates: the toy requirements, oracle-simulated feedback, and
KNN downstream task are synthetic. In particular, this run does not show an
advantage for impact-aware selection. Random uses less of the nominal budget
because query costs are indivisible and the remaining budget cannot always buy
another candidate; realized costs must therefore be shown with every
comparison.

The impact score estimates expected top-k selection-set change using the
synthetic downstream selector. It is not a validated estimate of human time,
human usefulness, or real model performance. The next meaningful local step is
to test the score on controlled synthetic cases where query impact and label
disagreement are independently varied, and verify that ranking responds to
each factor as specified. Do not tune the benchmark against held-out accuracy.

## Required interpretation

This local run can establish that the proposed selection score is implemented
and that the comparison is budget-matched. It cannot establish human usability,
LLM judgment quality, or real-task downstream gains. A later server experiment
must replace the synthetic requirement hypotheses with independently defined
requirement models and replace simulated feedback with human or explicitly
validated LLM feedback.
