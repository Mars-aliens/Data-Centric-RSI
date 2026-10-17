# Decision Model Selection

## Locked recommendation for the first real run

Use `Qwen/Qwen2.5-7B-Instruct` as the **decision LLM** for both
`fixed-harness-react` and `react-reflexion`.

This is a practical choice for the first server experiment because it is an
open instruct model with a stable Transformers/vLLM path, supports the English
HelpSteer2 records, and is large enough to select structured data operations
without making the controller more expensive than the task-model experiment.
It is not being claimed as an official ReAct or Reflexion model.

Run it behind a pinned vLLM OpenAI-compatible endpoint. Use:

```text
temperature = 0.0
top_p = 1.0
max_output_tokens = 512
structured JSON output = required
```

Pin the exact Hugging Face revision or local model-directory checksum before
the first result. The same revision, prompt hash, context limit, tool surface,
and total LLM call/token budget must be used by both baselines.

The repository includes `policy_command.py`, a small stdin/stdout client for
that endpoint. Configure `LEGO_POLICY_API_URL` and keep credentials in the
server secret manager. The command returns one proposal and reports its call
and token cost to the LEGO budget ledger.

## Baseline semantics

`fixed-harness-react` uses the fixed baseline pipeline, including data
processing, Task Model training, evaluation, and run logging. It lets the controller choose one current-batch operation. It never
persists a code, Prompt, Skill, or Memory-rule change.

`react-reflexion` uses exactly the same setup and additionally stores the
controller's bounded textual reflection in `reflections.jsonl`. A record is
visible starting with the next policy decision, carries its source batch and model/prompt
provenance, and is retrieved deterministically with a maximum of four records.
It remains text memory; it cannot directly edit the Harness.

The Task Model is a separate experimental choice. It must be selected and
pinned in the training profile independently of this decision-model choice.

The fixed baseline pipeline is the shared starting point. The three LEGO
technology names are: dependency-guided joint editing, executable experience
reuse, and continuous-run feedback selection. These names describe the
research mechanisms; they are not pipeline version labels.
