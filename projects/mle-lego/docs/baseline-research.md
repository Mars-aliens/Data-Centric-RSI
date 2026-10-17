# LEGO Baseline Research and Reproduction Plan

## Evidence standard

An arXiv paper is evidence that a method was published as a preprint; it does
not establish an official benchmark requirement, official implementation, or
permission to reuse code. Each result must be labeled as one of:

- `official reproduction`: authors' implementation and original protocol,
- `author-code adaptation`: authors' code changed for the LEGO task,
- `independent reimplementation`: method implemented from the paper,
- `project control/ablation`: designed for LEGO, without an external method claim.

Do not report adapted runs as results from the original paper.

## Primary comparison set

| Method | Original source | Author implementation evidence | LEGO status and adaptation |
|---|---|---|---|
| Fixed Harness + ReAct-style controller | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629), ICLR 2023. Metadata verified against arXiv API on 2026-10-02. | [ysymyth/ReAct](https://github.com/ysymyth/ReAct), MIT, `master` at commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`, active. The paper code covers its original QA/interactive environments, not this HelpSteer2 stream. | Project control. Use the same decision LLM and action/tool interface, but persistently freeze Harness, program, and Skill versions. Do not call this an exact ReAct benchmark reproduction. Current local run is deterministic TF-IDF scaffold only.
| ReAct + Reflexion | Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366), NeurIPS 2023. Metadata verified against arXiv API on 2026-10-02. | [noahshinn/reflexion](https://github.com/noahshinn/reflexion), MIT, `main` at commit `218cf0ef1df84b05ce379dd4a8e47f17766733a0`. Original benchmark/task implementations are not the LEGO data stream. | Author-code adaptation or faithful independent reimplementation: freeze executable Harness; save verbal reflections from released feedback; expose only eligible prior reflections later. No persistent code/prompt/Skill rewrite. Account for reflection-token budget. Existing local implementation is toy/rule-based, not paper implementation.
| General Self-Harness | Zhang et al., *Self-Harness: Harnesses That Improve Themselves*, [arXiv:2606.09498](https://arxiv.org/abs/2606.09498), v3 dated 2026-08-20. Paper metadata and its repository URL were verified against arXiv API on 2026-10-02. | [qzzqzzb/Self-Harness](https://github.com/qzzqzzb/Self-Harness), `main` at commit `2720dbb3f52283684f4b85a1065d642df1779dd8`, not archived. GitHub metadata reports no declared license. Inspect repository license and exact release before reuse; absent license means do not assume permission to copy/modify. | Adapt the paper's weakness mining, minimal failure-tied proposal, and regression validation while removing LEGO dependency guidance. Same editable components, traces, candidate budget, and validation data. If code cannot legally/technically be used, independently reimplement from paper and label it accordingly. |
| CurateEvo | Wang et al., *CurateEvo: Data-Curation Evolving for Agentic Post-Training*, [arXiv:2607.06140v1](https://arxiv.org/abs/2607.06140), submitted 2026-07-07. Title, authors, version and abstract page verified on 2026-10-02. | The arXiv page exposes no author GitHub link; official implementation/license: **unverified**. Do not substitute a similarly named third-party repository. | Required by the current LEGO proposal as an adaptation. Confirm the original curation loop, mutation/evaluation protocol, license, dependencies, and dataset assumptions before implementation. CurateEvo evolves executable curation code from held-out failures and includes cost-aware pruning; preserve these defining ideas while adapting the interface to the same stream. Until code/protocol review is complete, do not claim a completed CurateEvo baseline.

## Additional literature methods

These are relevant prior work but are not explicitly required in the current
baseline table in `idea/MLE-LEGO.md`. They are extension candidates, not gates
for the initial core matrix.

| Method | Original source | Code status | Decision |
|---|---|---|---|
| DataEvolver | Deng et al., *DataEvolver: Automatic Data Preparation for Large Language Models through Multi-Level Self-Evolving*, [arXiv:2606.07001](https://arxiv.org/abs/2606.07001), v2 dated 2026-06-10. Title, seven benchmarks, code link and authors verified on 2026-10-02. | [ruc-datalab/DataEvolver](https://github.com/ruc-datalab/DataEvolver), MIT, checked at commit `2f02be1267ba97164ac6c5326d1e583af5532104`. | Optional. Its original protocol uses seed-guided raw-to-prepared SFT on Alpaca, ARC-Easy/Challenge, GSM8K/MATH and Spider/BIRD. Treat it as a data-preparation alignment baseline, not a native long-horizon LEGO reproduction.
| HarnessEvolve | Jiang et al., *HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution*, [arXiv:2609.00829](https://arxiv.org/abs/2609.00829), v1 dated 2026-09-01. Metadata and HTML experimental setup verified on 2026-10-02. | The checked arXiv record exposed no code URL. Official repository/license not verified. The paper reports SearchQA, OfficeQA, SpreadsheetBench, CloudCoreNetwork-QA, and Wireless-QA; the last two are in-house datasets. | Optional but highly relevant. Its reference-trajectory alignment and quality/performance gates differ from dependency-aware data-component grouping. Public adapter candidates are OfficeQA and SpreadsheetBench; do not claim reproduction on the in-house datasets or equivalence to LEGO dependency-aware editing.

## Project controls and ablations

These are not externally sourced baselines and should be labeled as controls or
ablations, not as published methods:

- `independent-edit`: one component at a time, testing whether joint edits help.
- `random-joint-edit`: same-size joint candidate without dependency guidance.
- `one-batch-selection` vs `three-batch-selection`: tests the value of future-batch validation.
- `same-memory-no-process-update`: tests persistent process change against access to the same initial experience.
- `dependency-aware-joint-edit`: proposed method, not a baseline.

## Uniform server protocol

Before running any comparison, lock and hash the following:

1. Decision LLM name/revision, decoding settings, system prompt, context limit,
   and all API/model versions.
2. Task model initial checkpoint and identical training recipe for every arm.
3. Source dataset revision, normalized-data hash, group-level split, ordered
   stream batches, feedback-release schedule, and label construction.
4. Tool/API surface and permissions. Baselines may differ in policy, not in
   access to data or tools.
5. Total budget, including decision/evolution LLM calls and tokens, candidate
   executions, failed candidates, task-model training, evaluation, and human
   feedback requests/time.
6. Candidate acceptance logic, held-in selection, held-out regression policy,
   and final-test lock. `final_test` must not be visible before the final report.
7. At least multiple fixed seeds/runs and a paired statistical analysis plan.

Fork candidate branches from the same initial model, data, and memory state.
Do not share branch-created checkpoints, reflections, or experiences. Keep
results for simulated feedback separate from newly collected human preference
results.

## Method acceptance tests

The following tests define whether an external-method adapter is faithful enough
to enter the server comparison. They are protocol tests, not claims that the
original papers evaluated HelpSteer2.

### ReAct + Reflexion

- ReAct action selection and tool results are logged as ordered reasoning/action
  steps under the fixed Harness interface.
- Reflection is generated only after an eligible feedback/result event and is
  stored as text in episodic memory.
- Later decisions can read eligible prior reflections, but the Harness code,
  component graph, and persistent Skills remain unchanged.
- Reflection tokens and calls are charged to the same decision budget as LEGO.

### General Self-Harness

- Weakness mining clusters execution traces or failure records before proposing
  edits; a raw score alone is not accepted as a failure diagnosis.
- Each proposal is minimal, tied to a diagnosed failure, and preserves a parent
  Harness version.
- Regression validation runs on held-in/held-out policy before acceptance; no
  dependency-aware grouping or LEGO-specific graph heuristic is exposed.
- Failed proposals leave the active Harness, task checkpoint, and memory
  unchanged.

### CurateEvo

- The evolving object is executable curation code, not only a natural-language
  reflection or a model checkpoint.
- Feedback comes from held-out development failures and drives augmentation,
  filtering, refinement, or pruning under a cost-aware objective.
- Re-curation and downstream post-training use the same fixed recipe as other
  arms; only the curation strategy is varied.
- The adapter records which parts are original CurateEvo behavior and which are
  required by the LEGO streaming protocol.

### DataEvolver (optional)

- Seed-guided data understanding is separated from operator-level and
  pipeline-level evolution.
- Operator and pipeline candidates have executable checks and held-out
  validation; the adapter does not collapse both levels into one arbitrary edit.
- Results are reported as an optional data-preparation comparison unless the
  task target and raw-to-prepared interface are demonstrably equivalent.

### HarnessEvolve (optional)

- Reference trajectories are distinct from ordinary failed trajectories and are
  available only under the declared protocol.
- Evaluation, optimization, and gating remain separable modules.
- Quality/data-leakage/prompt-bloat gates are checked before performance gates;
  recent-batch regression and held-out epoch-end selection are logged.
- The adapter must not be labeled equivalent to LEGO dependency-aware editing;
  the two methods use different signals and gate definitions.

## Pre-server completion gates

- [x] Record candidate paper identifiers and known author repository URLs.
- [x] Distinguish published-method adaptations from LEGO controls/ablations.
- [x] Review the public HTML method structure and capture exact versions,
      algorithms, and claims for Self-Harness, CurateEvo, DataEvolver, and
      HarnessEvolve. Full PDF/appendix review remains part of each adapter's
      implementation review.
- [ ] Verify official repository links, commit hashes, license files, and
      reproduction instructions for every external method to be run.
- [ ] Decide per method: author-code adaptation or independent reimplementation.
- [ ] Specify exact method-to-LEGO mapping and acceptance tests before coding.
- [ ] Freeze the real decision/task model, prompt, budget, stream protocol,
      repetitions, primary metric, and statistical analysis.
- [ ] Implement and test real-model adapters locally or in a server smoke job
      before scheduling expensive runs.

The repository and paper metadata checks in this note were performed on
2026-10-02. Recheck mutable GitHub branches and licenses before server execution.
