# HH-RLHF Qwen Protocol

This document is the canonical flow for the current Aladdin core-technology-one
experiment. The old `formal-hh-rlhf-qwen-v1` run is retained as a historical
baseline; the current main strategy is `aladdin-qwen-reliable` under
`custom-diagnostic-v1`.

```mermaid
flowchart TD
    A[Anthropic HH-RLHF train<br/>fixed dataset revision] --> B[Prompt-level split<br/>predeclared split seeds]
    B --> C{Split audit<br/>prompt overlap = 0}
    C -->|fail| X[Abort run]
    C -->|pass| D[Shared seed groups: 64<br/>held-in pool: 256<br/>held-out groups: 512]
    D --> E1[seed-only<br/>use 64 seed groups]
    D --> E2[random<br/>select 32 feedback groups]
    D --> E3[aladdin-qwen-reliable<br/>target rank=0.35 + 25% novelty]
    E2 --> F[Fixed Qwen A/B feedback<br/>order from seed + group_id]
    E3 --> F
    E1 --> G[Seed labels]
    F --> H[LLM-label records]
    E2 --> I[Oracle-label diagnostic]
    E3 --> I
    G --> J[Qwen LoRA preference classifier]
    H --> J
    I --> K[Qwen LoRA oracle diagnostic]
    J --> L[Held-out pairwise accuracy]
    K --> M[LLM-label vs oracle-label gap]
    L --> N[Paired analysis over seeds 17..26]
    M --> N
    N --> O[Report trend or stable evidence<br/>CI crossing zero means no claim of advantage]
```

## Frozen Controls

| Component | Formal setting |
|---|---|
| Strategies | `seed-only`, `random`, `aladdin-qwen-reliable` |
| Seeds | `17..26` per split |
| Split | `prompt`, predeclared split seed, abort on overlap |
| Feedback budget | 32 groups per feedback strategy |
| Training exposure | `equal-epochs`, 100 seed-only reference steps |
| Evaluation batch | 8 for split 17/23/29; 16 for split 37 |
| Feedback model | `Qwen/Qwen2.5-7B-Instruct`, pinned revision |
| Task model | Same Qwen snapshot with LoRA preference classifier |
| Selector | 75% target-centered uncertainty value + 25% prompt novelty; target rank `0.35` |
| Evaluation | 512 held-out groups, pairwise accuracy |
| Test set | Official test remains sealed |

## Split Schedule

The completed schedule contains prompt splits 17, 23, 29 and 37. Results are
reported within split using paired seeds; split-level averages are descriptive
and are not treated as a pooled test over independent observations.

Split 17 is exploratory because the reliable target was fixed after the earlier
diagnostic. Splits 23, 29 and 37 are confirmation runs under the frozen rule.
The split-37 Oracle-label result is lower than random and is retained as
heterogeneity evidence; it is not omitted in favor of the LLM-label result.

## Interpretation Boundary

This is a single-round historical replay. Qwen supplies simulated feedback from
existing HH-RLHF pairs. The run does not create new answers, collect new human
labels, or update the selector from downstream training gain. It tests whether
the selection rule improves this downstream preference-classification task. It
does not demonstrate the complete recursive RSI loop, technical two, technical
three, SFT, DPO, or official HH-RLHF test performance.
