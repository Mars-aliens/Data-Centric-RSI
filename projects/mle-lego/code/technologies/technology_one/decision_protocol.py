from __future__ import annotations

"""Protocols shared by the real ReAct and Reflexion policy adapters.

The policy model is asked for a short action rationale, not hidden chain of
thought.  Reflexion stores only an auditable, bounded text record that can be
shown to later policy calls.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from core_system.mle_core.models import Observation, to_jsonable


REACT_SYSTEM_PROMPT = """You are the fixed decision policy for a data-processing Harness.
Act with ReAct: inspect the observation, choose exactly one permitted action,
and use the action result on a later turn. Do not modify Harness code, prompts,
Skills, Memory rules, or versions. Return one JSON object only with:
action_type, arguments, expected_versions, rationale, estimated_cost.
The rationale is a short operational justification (at most 80 words), not
private chain-of-thought. Never request final_test before it is unlocked.
Follow this fixed action sequence: select seed records, train the newest
dataset artifact, evaluate held_in, evaluate held_out, then finish. For
select_records use operation=select_records and version=1.0. Do not use
generated records or arbitrary operation names.
"""

REFLEXION_SYSTEM_PROMPT = """You are the fixed decision policy for a data-processing Harness.
Act with ReAct and verbal Reflexion. The Harness implementation remains fixed;
you may only choose the next permitted action for the current batch. Before
the action, use eligible prior reflection records when their conditions match.
If the previous batch produced a failure or regression, return a bounded
reflection object with problem, evidence, lesson, conditions, and source_batch.
Do not write code or change persistent Harness components. Reflexion is
text-only episodic memory and never an executable Harness update. Return one
JSON object only with proposal and optional reflection. Do not provide private
chain-of-thought.
Follow the same fixed action sequence: select seed records, train the newest
dataset artifact, evaluate held_in, evaluate held_out, then finish.
"""


@dataclass(frozen=True)
class ReflectionRecord:
    reflection_id: str
    source_batch: str
    problem: str
    evidence: dict[str, Any]
    lesson: str
    conditions: dict[str, Any]
    created_by_model: str
    model_revision: str
    prompt_hash: str
    token_count: int
    eligible_from_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReflectionStore:
    """Append-only JSONL memory with deterministic, bounded retrieval."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[ReflectionRecord] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(ReflectionRecord(**json.loads(line)))

    def add(self, record: ReflectionRecord) -> None:
        if any(item.reflection_id == record.reflection_id for item in self.records):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")
        self.records.append(record)

    def eligible(self, *, decision_step: int, limit: int = 4) -> list[ReflectionRecord]:
        # New reflections become visible only on the next policy decision. Sorting
        # by id makes retrieval reproducible when scores tie.
        visible = [item for item in self.records if item.eligible_from_step <= decision_step]
        return sorted(visible, key=lambda item: item.reflection_id)[-limit:]


def prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def build_policy_prompt(
    observation: Observation,
    *,
    method: str,
    reflections: Iterable[ReflectionRecord] = (),
) -> tuple[str, str]:
    if method == "fixed-harness-react":
        system = REACT_SYSTEM_PROMPT
    elif method == "react-reflexion":
        system = REFLEXION_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported decision baseline: {method}")
    payload: dict[str, Any] = {
        "observation": to_jsonable(observation),
        "permitted_action_types": [
            "select_records",
            "transform_records",
            "filter_records",
            "generate_records",
            "train",
            "evaluate",
            "request_feedback",
            "finish",
        ],
        "action_sequence": [
            "select_records(seed)",
            "train(newest_dataset_artifact)",
            "evaluate(held_in)",
            "evaluate(held_out)",
            "finish",
        ],
    }
    if method == "react-reflexion":
        payload["eligible_reflections"] = [item.to_dict() for item in reflections]
    user = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return system, user


def reflection_from_response(
    raw: dict[str, Any],
    *,
    observation: Observation,
    model_id: str,
    model_revision: str,
    system_prompt: str,
) -> ReflectionRecord | None:
    value = raw.get("reflection")
    if not isinstance(value, dict):
        return None
    if not observation.metrics:
        return None
    problem = str(value.get("problem", "")).strip()
    lesson = str(value.get("lesson", "")).strip()
    if not problem or not lesson:
        return None
    if len(problem.split()) > 40 or len(lesson.split()) > 80:
        raise ValueError("Reflection problem/lesson exceeds the bounded text limit")
    evidence = value.get("evidence", {})
    conditions = value.get("conditions", {})
    if not isinstance(evidence, dict) or not isinstance(conditions, dict):
        raise ValueError("reflection.evidence and reflection.conditions must be objects")
    token_count = int(raw.get("reflection_token_count", 0))
    reflection_id = str(value.get("reflection_id", f"{observation.batch_id}-reflection"))
    return ReflectionRecord(
        reflection_id=reflection_id,
        source_batch=str(value.get("source_batch", observation.batch_id)),
        problem=problem,
        evidence=evidence,
        lesson=lesson,
        conditions=conditions,
        created_by_model=model_id,
        model_revision=model_revision,
        prompt_hash=prompt_hash(system_prompt),
        token_count=token_count,
        eligible_from_step=observation.batch_index + 1,
    )
