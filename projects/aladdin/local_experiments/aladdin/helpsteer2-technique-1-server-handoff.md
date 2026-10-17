# Technology One: HelpSteer2 Historical Feedback Replay

## Purpose and scope

This experiment tests only the first technique in `idea/MLE-Aladdin.md`:
choose which feedback request to make next by combining requirement
disagreement, expected changes to the downstream selected-data set, review
cost, and a fixed random-exploration reserve.

It is a historical replay over existing HelpSteer2 human ratings. It does not
collect new human annotations, call an LLM, test techniques two or three, or
demonstrate the complete recursive RSI system.

## Frozen protocol

- Dataset: `nvidia/HelpSteer2`, revision
  `990b2711a36180dd19d9c94b8627844866f8982a`.
- Only the `seed`, `held_in`, and `held_out` splits from
  `helpsteer2-train.jsonl` are used. The validation/final-test file is not
  opened. Groups are kept intact using `source_group_id`.
- Candidate requirement interpretations are fixed before each run:
  helpfulness-only, correctness-only, and coherence-only. These are
  semi-synthetic preference profiles constructed from HelpSteer2's existing
  human attribute ratings, not newly elicited user requirements.
- Seed examples train five TF-IDF attribute models. Their outputs estimate
  each candidate response's attribute values. The profile posterior is
  initialized using only the seed groups.
- A query is one complete two-response prompt group. Its historical label is
  released only after the selection policy requests that group. The query pool
  is disjoint from the downstream data-selection pool.
- After feedback, the requirement posterior is updated. The downstream
  selection pool is reranked, and a fixed-size set is pseudo-labeled from the
  updated posterior. Unqueried selection-pool labels are never released.
- Query cost is an explicitly uncalibrated proxy:
  `5 seconds + visible input words / 220 words per minute`. Methods share the
  same time cap and maximum request count; realized cost and request count are
  both recorded.
- Impact score:
  `disagreement * expected_selection_change / estimated_review_seconds`.
  Expected selection change is the posterior-weighted symmetric difference
  between the current top-k downstream set and the counterfactual top-k set
  after each possible feedback answer.
- `impact-aware-random` reserves the configured fraction of the feedback-time
  budget for random queries, then uses the impact score.
- Downstream metric: group-level pairwise accuracy on held-out, decisive
  pairs, against the simulated target profile. Ties are excluded and counted.
- `final_test` is never used for selection, training, parameter choice, or
  evaluation. Unlock it only after the protocol and method are frozen.

## Strategies

- `seed-only`: train using the same seed groups, with no new feedback and no
  extra selected training groups.
- `random`: random feedback requests under the common cost cap.
- `disagreement-only`: maximize requirement disagreement per estimated review
  second.
- `impact-aware`: maximize disagreement times expected downstream selection
  change per estimated review second.
- `impact-aware-random`: the proposed method; fixed random exploration reserve
  plus impact-aware requests.

Every method within a `(split_seed, policy_seed, target_profile)` block uses
identical seed, query, downstream-selection, and held-out groups. The four
feedback-request methods share the same time cap, maximum request count, and
downstream selection size. Primary paired comparisons are
`impact-aware-random` against `random`, `disagreement-only`, and `seed-only`.
Confidence intervals use split-level paired differences; target profiles and
policy seeds are averaged within each split before inference.

## Local validation

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest local_experiments\tests\test_active_feedback_helpsteer2.py -q
```

Run a small real-data debug matrix:

```powershell
.\.venv\Scripts\python.exe -m local_experiments.aladdin.active_feedback_helpsteer2 `
  --data data\helpsteer2-train.jsonl `
  --output tmp\technology-one-helpsteer2-debug.json `
  --split-seeds 17 `
  --policy-seeds 17 18 `
  --seed-group-limit 32 `
  --query-group-limit 32 `
  --selection-group-limit 64 `
  --heldout-group-limit 64 `
  --feedback-time-budget-seconds 300 `
  --max-queries 8 `
  --selection-budget 16
```

After debug passes, run the frozen local matrix:

```powershell
.\.venv\Scripts\python.exe -m local_experiments.aladdin.active_feedback_helpsteer2 `
  --data data\helpsteer2-train.jsonl `
  --output tmp\technology-one-helpsteer2-matrix.json `
  --split-seeds 17 23 29 37 `
  --policy-seeds 17 18 19 `
  --seed-group-limit 64 `
  --query-group-limit 256 `
  --selection-group-limit 512 `
  --heldout-group-limit 512 `
  --feedback-time-budget-seconds 1800 `
  --max-queries 48 `
  --selection-budget 64 `
  --random-reserve-fraction 0.25
```

The report records the dataset checksum, configuration, per-request diagnostics,
posterior updates, selected group IDs, released-record counts, pool IDs and
overlap audit, pairwise metrics, paired differences, and the final-test access
flag.

## Local matrix result

The frozen local matrix completed all 180 runs (4 split seeds x 3 policy seeds
x 3 historical preference profiles x 5 strategies). Mean held-out pairwise
accuracy was:

Exact local train-file SHA-256:
`5bd88e03cb93527131d2a75b30940ea8a969a23096f8f529eb0e9bb114f4d6e9`.
For replication, transfer this exact normalized `helpsteer2-train.jsonl` as a
separate data artifact and verify its checksum. Do not silently substitute an
older or malformed server copy.

| Strategy | Mean pairwise accuracy |
|---|---:|
| `seed-only` | 49.44% |
| `random` | 52.53% |
| `disagreement-only` | 53.18% |
| `impact-aware` | 53.18% |
| `impact-aware-random` | 53.09% |

Split-level paired differences for `impact-aware-random` were `+0.56 pp`
versus random (95% t CI `[-0.48, +1.60]`, 4 split seeds), `-0.09 pp` versus
disagreement-only (95% t CI `[-0.79, +0.61]`), and `+3.65 pp` versus seed-only
(95% t CI `[+1.03, +6.27]`). These are local exploratory replay results, not
independent human validation. The impact-aware method did not improve over
disagreement-only in this matrix. Although query traces differed in some
conditions, impact-aware and disagreement-only selected the same downstream
data groups in all 36 paired conditions. The random-reserve variant selected
a different downstream set in 13/36 conditions versus disagreement-only.

The implementation passed its leakage and budget audits, but the mechanism
has not yet shown incremental value over disagreement-only. Preserve this
negative result; do not tune the profile definitions or budgets against these
held-out scores and then present the same matrix as confirmatory evidence.

## Server run

The experiment requires Python, NumPy, and scikit-learn only. It does not need a
GPU, vLLM, Transformers, or a model API. On the server, install the pinned
`requirements-integrated-aladdin.txt` dependencies in the chosen Python
environment, set `PYTHONPATH` to the extracted package root, and use the same
dataset revision and checksums as the local run. Keep the normalized dataset
outside the source archive and verify its checksum before starting.

Run the debug command first. Only then run the frozen matrix, writing output to
a new run path. Do not reuse a partially written output as evidence of a
completed run. Because the local full matrix is already complete, a server
rereun is for environment replication or a separately registered protocol,
not a prerequisite for interpreting the local matrix.

## Interpretation limits

This replay can test whether the policy implementation changes its requests
under a controlled, budgeted historical-feedback setting. It cannot establish
that the estimated time is actual human effort, that the three hand-specified
profiles represent a real user's evolving requirements, or that an LLM/human
interface would obtain the same feedback. TF-IDF is a lightweight downstream
protocol baseline, not the final generation-model result. Positive local
results are a reason to run an independently controlled human or model-backed
study, not a claim that the full Aladdin system is validated.
