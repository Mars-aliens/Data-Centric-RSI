"""Task-adapted synthesis strategies; all outputs use the same record schema."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable
from typing import Any

from .generator import TextGenerator

Record = dict[str, Any]
Generate = Callable[..., str]

STRATEGIES = ("seed-only", "fixed-prompt", "self-instruct", "evol-instruct")

EVOLUTION_OPERATIONS = (
    "add a meaningful constraint while preserving the task's intent",
    "increase the depth of the task without making it needlessly verbose",
    "replace a broad concept with a concrete, realistic case",
    "request a multi-step answer only when the task genuinely needs it",
    "create a new task in the same domain with similar complexity",
)


def _field(record: Record, names: tuple[str, ...], *, required: bool = True) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if required:
        raise ValueError(f"record is missing one of fields: {', '.join(names)}")
    return ""


def normalize_seed(record: Record, index: int) -> Record:
    task_input = _field(record, ("input", "prompt", "instruction", "query"))
    output = _field(record, ("output", "response", "completion", "chosen"), required=False)
    record_id = str(record.get("record_id") or record.get("id") or f"seed-{index}")
    return {
        "record_id": record_id,
        "input": task_input,
        "output": output,
        "source_group_id": str(record.get("source_group_id") or record_id),
        "requirements": record.get("requirements", []),
        "context": record.get("context", []) if isinstance(record.get("context", []), list) else [],
    }


def _parse_candidate(text: str) -> tuple[str, str]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("model must return a JSON object with input and output strings") from exc
    if not isinstance(value, dict):
        raise ValueError("candidate JSON must be an object")
    task_input = _field(value, ("input", "prompt", "instruction", "query"))
    output = _field(value, ("output", "response", "completion"))
    return task_input, output


def _record(
    task_input: str,
    output: str,
    *,
    strategy: str,
    seed: Record,
    index: int,
    calls: int,
    operation: str | None = None,
) -> Record:
    result: Record = {
        "record_id": f"{strategy}-{index:05d}",
        "input": task_input,
        "output": output,
        "source_group_id": seed["source_group_id"],
        "parent_record_id": seed["record_id"],
        "context": seed.get("context", []),
        "synthesis_strategy": strategy,
        "generation_calls": calls,
    }
    if operation:
        result["evolution_operation"] = operation
    return result


def _seed_for(seeds: list[Record], index: int, rng: random.Random) -> Record:
    return seeds[rng.randrange(len(seeds))]


def _fixed_prompt(seed: Record) -> str:
    return """Create one new, realistic task example for the same task domain as the seed.
The new input must be meaningfully different, not a paraphrase. Write a high-quality
response that satisfies the task. Return only JSON with string fields \"input\" and
\"output\". Do not copy the seed response.

Seed input:
{input}

Seed response:
{output}

Prior conversation context (may be empty):
{context}

Current confirmed requirements for future examples:
{policy_guidance}""".format(
        **{
            **seed,
            "context": json.dumps(seed.get("context", []), ensure_ascii=False),
            "policy_guidance": _format_policy_guidance(seed.get("policy_guidance", [])),
        }
    )


def _format_policy_guidance(guidance: Any) -> str:
    if isinstance(guidance, str):
        return guidance.strip() or "None recorded."
    if isinstance(guidance, (list, tuple)):
        values = [str(item).strip() for item in guidance if str(item).strip()]
        return "\n".join(f"- {item}" for item in values) or "None recorded."
    return "None recorded."


def _self_instruct_input(seed: Record) -> str:
    return """Following the seed-driven Self-Instruct idea, propose one new task input
from the same broad task domain. Keep it answerable, realistic, and distinct from the
seed. Return only the task input, with no explanation.

Seed task input:
{input}

Seed response:
{output}

Prior conversation context (may be empty):
{context}

Current confirmed requirements for future examples:
{policy_guidance}""".format(
        **{
            **seed,
            "context": json.dumps(seed.get("context", []), ensure_ascii=False),
            "policy_guidance": _format_policy_guidance(seed.get("policy_guidance", [])),
        }
    )


def _answer_prompt(task_input: str, seed: Record) -> str:
    return """Answer the task input below. Match the quality and response style of the
demonstration, but do not copy its wording. Return only the response.

Task input:
{task_input}

Demonstration input:
{seed_input}

Demonstration response:
{seed_output}

Prior conversation context (may be empty):
{context}

Current confirmed requirements for future examples:
{policy_guidance}""".format(
        task_input=task_input,
        seed_input=seed["input"],
        seed_output=seed["output"],
        context=json.dumps(seed.get("context", []), ensure_ascii=False),
        policy_guidance=_format_policy_guidance(seed.get("policy_guidance", [])),
    )


def _evolve_input(seed: Record, operation: str) -> str:
    return """Apply this Evol-Instruct-inspired operation to the seed task input:
{operation}
Preserve its core intent and keep the result answerable. Return only the revised
task input, without labels or explanation.

Seed task input:
{input}

Prior conversation context (may be empty):
{context}""".format(
        operation=operation,
        **{**seed, "context": json.dumps(seed.get("context", []), ensure_ascii=False)},
    )


def synthesize(
    seeds: Iterable[Record],
    *,
    strategy: str,
    count: int,
    generator: TextGenerator | Generate | None = None,
    seed_value: int = 0,
    max_tokens: int = 512,
    temperature: float = 0.7,
    index_offset: int = 0,
    policy_guidance: Iterable[str] | None = None,
) -> list[Record]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown synthesis strategy: {strategy}")
    if count < 0:
        raise ValueError("count must be non-negative")
    normalized = [normalize_seed(item, i) for i, item in enumerate(seeds)]
    if policy_guidance:
        guidance = [str(item).strip() for item in policy_guidance if str(item).strip()]
        normalized = [{**item, "policy_guidance": guidance} for item in normalized]
    if not normalized:
        raise ValueError("at least one seed record is required")
    if strategy == "seed-only":
        return [
            {
                **item,
                "synthesis_strategy": "seed-only",
                "generation_calls": 0,
            }
            for item in normalized
        ]
    if generator is None:
        raise ValueError(f"strategy {strategy} requires a text generator")

    call: Generate
    if hasattr(generator, "generate"):
        call = generator.generate  # type: ignore[assignment]
    else:
        call = generator  # type: ignore[assignment]
    rng = random.Random(seed_value)
    results = []
    for index in range(count):
        parent = _seed_for(normalized, index, rng)
        if strategy == "fixed-prompt":
            task_input, output = _parse_candidate(
                call(_fixed_prompt(parent), max_tokens=max_tokens, temperature=temperature)
            )
            results.append(
                _record(task_input, output, strategy=strategy, seed=parent, index=index_offset + index, calls=1)
            )
            continue

        if strategy == "self-instruct":
            task_input = call(
                _self_instruct_input(parent), max_tokens=max_tokens // 2, temperature=temperature
            ).strip()
            if not task_input:
                raise ValueError("Self-Instruct adapter generated an empty task input")
            output = call(
                _answer_prompt(task_input, parent),
                max_tokens=max_tokens - max_tokens // 2,
                temperature=temperature,
            ).strip()
            if not output:
                raise ValueError("Self-Instruct adapter generated an empty response")
            results.append(
                _record(task_input, output, strategy=strategy, seed=parent, index=index_offset + index, calls=2)
            )
            continue

        operation = EVOLUTION_OPERATIONS[(index_offset + index) % len(EVOLUTION_OPERATIONS)]
        task_input = call(
            _evolve_input(parent, operation), max_tokens=max_tokens // 2, temperature=temperature
        ).strip()
        if not task_input:
            raise ValueError("Evol-Instruct adapter generated an empty evolved task input")
        output = call(
            _answer_prompt(task_input, parent),
            max_tokens=max_tokens - max_tokens // 2,
            temperature=temperature,
        ).strip()
        if not output:
            raise ValueError("Evol-Instruct adapter generated an empty response")
        results.append(
            _record(
                task_input,
                output,
                strategy=strategy,
                seed=parent,
                index=index_offset + index,
                calls=2,
                operation=operation,
            )
        )
    return results
