from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from core_system.mle_core.models import Budget, TaskRecord

from .calibration import ReliabilityCalibrator
from .system import (
    FeedbackQuery,
    HumanFeedback,
    QualityAssessment,
    RequirementHypothesis,
    RequirementState,
)
from .strategies import strategy_instruction, strategy_for


JsonTransport = Callable[[dict[str, Any]], dict[str, Any]]


def _json_from_text(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        if start < 0:
            raise ValueError("Model response does not contain a JSON object") from None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(value[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("Model response contains invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    return parsed


class OpenAICompatibleJsonClient:
    """Small auditable client for vLLM/OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        transport: JsonTransport | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an http(s) URL")
        if not model.strip():
            raise ValueError("model cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.budget = Budget()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=True),
                },
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
                if response_schema is not None
                else {"type": "json_object"}
            ),
        }
        if self.transport is not None:
            response = self.transport(payload)
        else:
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key
                        else {}
                    ),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Chat endpoint returned HTTP {exc.code}: {detail[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Cannot reach chat endpoint: {exc.reason}") from exc
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Chat endpoint response has no assistant content") from exc
        usage = response.get("usage") or {}
        total_tokens = int(usage.get("total_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(
                1, (len(str(payload["messages"])) + 3) // 4
            )
        if completion_tokens <= 0:
            # Older compatible servers may expose only total_tokens. Preserve
            # that accounting for audit compatibility; vLLM exposes both.
            completion_tokens = (
                total_tokens
                if total_tokens > 0
                else max(1, (len(str(content)) + 3) // 4)
            )
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.budget = self.budget.add(Budget(llm_calls=1, llm_tokens=total_tokens))
        return _json_from_text(str(content))


class LLMRequirementScorer:
    def __init__(self, client: OpenAICompatibleJsonClient):
        self.client = client
        self._cache: dict[tuple[str, str], float] = {}

    @property
    def budget(self) -> Budget:
        return self.client.budget

    def score(self, hypothesis: RequirementHypothesis, record: TaskRecord) -> float:
        key = (hypothesis.hypothesis_id, record.record_id)
        if key not in self._cache:
            result = self.client.complete_json(
                system_prompt=(
                    "Judge whether the candidate satisfies the stated requirement. "
                    "Return only JSON: {\"score\": number from 0 to 1}."
                ),
                user_payload={
                    "requirement": hypothesis.text,
                    "candidate_input": str(record.input),
                    "candidate_target": record.target,
                },
                max_tokens=64,
            )
            score = float(result["score"])
            if not 0 <= score <= 1:
                raise ValueError("requirement score must be in [0, 1]")
            self._cache[key] = score
        return self._cache[key]


class LLMQualityScorer:
    def __init__(
        self,
        client: OpenAICompatibleJsonClient,
        rubric_variant: str = "default",
    ):
        self.client = client
        self.rubric_variant = rubric_variant
        self._cache: dict[str, QualityAssessment] = {}
        self.prompt_context = "No confirmed evaluator corrections are available."

    def set_prompt_context(self, value: str) -> None:
        self.prompt_context = value

    @property
    def budget(self) -> Budget:
        return self.client.budget

    def assess(self, record: TaskRecord) -> QualityAssessment:
        if record.record_id not in self._cache:
            result = self.client.complete_json(
                system_prompt=(
                    "Assess whether this synthetic training example is coherent, "
                    "self-contained, and label-consistent. Return only JSON with "
                    "score and confidence in [0,1], plus a short reason. "
                    f"Rubric variant: {self.rubric_variant}. "
                    + self.prompt_context
                ),
                user_payload={"input": str(record.input), "target": record.target},
                max_tokens=128,
            )
            assessment = QualityAssessment(
                score=float(result["score"]),
                confidence=float(result["confidence"]),
                reason=str(result.get("reason", "")),
            )
            if not 0 <= assessment.score <= 1 or not 0 <= assessment.confidence <= 1:
                raise ValueError("quality score and confidence must be in [0, 1]")
            self._cache[record.record_id] = assessment
        return self._cache[record.record_id]


class LLMCandidateGenerator:
    def __init__(
        self,
        client: OpenAICompatibleJsonClient,
        *,
        seed_example_limit: int = 16,
        seed_example_chars: int = 700,
    ):
        if seed_example_limit <= 0:
            raise ValueError("seed_example_limit must be positive")
        if seed_example_chars <= 0:
            raise ValueError("seed_example_chars must be positive")
        self.client = client
        self.seed_example_limit = seed_example_limit
        self.seed_example_chars = seed_example_chars
        self.last_lineage_repairs = 0
        self.last_lineage_drops = 0

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "target": {"type": "string"},
                            "parent_record_id": {"type": "string"},
                            "quality_slice": {"type": "string"},
                            "review_cost": {"type": "number"},
                        },
                        "required": [
                            "input",
                            "target",
                            "parent_record_id",
                            "quality_slice",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

    @property
    def budget(self) -> Budget:
        return self.client.budget

    def generate(
        self,
        *,
        seed_records: Sequence[TaskRecord],
        requirements: Sequence[RequirementHypothesis],
        confirmed_edits: Sequence[str],
        strategy: str,
        count: int,
        round_index: int,
        cost_budget: float | None,
    ) -> tuple[list[TaskRecord], Budget]:
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return [], Budget()
        max_tokens = int(cost_budget) if cost_budget is not None else max(512, count * 160)
        if max_tokens <= 0:
            return [], Budget()
        self.last_lineage_repairs = 0
        self.last_lineage_drops = 0
        # The full seed set can be much larger than the model context. Select a
        # stable, bounded context while retaining the full seed map for lineage.
        ranked_seeds = sorted(
            seed_records,
            key=lambda item: hashlib.sha256(
                f"{round_index}|{strategy}|{item.record_id}".encode("utf-8")
            ).hexdigest(),
        )
        prompt_seeds = ranked_seeds[: self.seed_example_limit]
        budget_before = self.client.budget
        completion_before = self.client.completion_tokens
        initial_max_tokens = max(256, min(max_tokens, int(max_tokens * 0.75)))
        result = self.client.complete_json(
            system_prompt=(
                "Generate diverse labeled training examples using only the supplied "
                "seed examples and requirements. Never reproduce a seed verbatim. "
                "Keep each candidate input concise (at most 240 characters). "
                "Return JSON {\"candidates\":[{\"input\":str,\"target\":str," 
                "\"parent_record_id\":str,\"quality_slice\":str}]}. "
                "Do not return more candidates than requested. "
                + strategy_instruction(strategy)
            ),
            user_payload={
                "strategy": strategy,
                "strategy_definition": strategy_for(strategy).__dict__,
                "requested_count": count,
                "requirements": [item.text for item in requirements],
                "confirmed_edits": list(confirmed_edits),
                "seed_examples": [
                    {
                        "record_id": item.record_id,
                        "input": str(item.input)[: self.seed_example_chars],
                        "target": item.target,
                    }
                    for item in prompt_seeds
                ],
                "seed_context_policy": {
                    "source_seed_count": len(seed_records),
                    "provided_seed_count": len(prompt_seeds),
                    "max_example_chars": self.seed_example_chars,
                },
            },
            max_tokens=initial_max_tokens,
            response_schema=self._response_schema(),
        )
        rows = result.get("candidates")
        if not isinstance(rows, list):
            raise ValueError("generator response must contain a candidates list")
        if len(rows) > count:
            raise ValueError("generator returned more candidates than requested")
        seed_by_id = {item.record_id: item for item in seed_records}
        valid_rows = []
        invalid_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            parent_id = str(row.get("parent_record_id", ""))
            if parent_id in seed_by_id:
                valid_rows.append(row)
            else:
                invalid_rows.append(row)

        if invalid_rows:
            completion_used = self.client.completion_tokens - completion_before
            repair_budget = max_tokens - completion_used
            if repair_budget >= 128:
                repaired = self.client.complete_json(
                    system_prompt=(
                        "Repair candidate metadata. For every supplied candidate, "
                        "keep input, target, and quality_slice unchanged and add a "
                        "non-empty parent_record_id chosen only from the allowed "
                        "IDs. Return exactly the same number and order of candidates. "
                        "Return JSON only."
                    ),
                    user_payload={
                        "allowed_parent_record_ids": sorted(seed_by_id),
                        "candidates": invalid_rows,
                    },
                    max_tokens=min(512, repair_budget),
                    response_schema=self._response_schema(),
                )
                repaired_rows = repaired.get("candidates")
                if isinstance(repaired_rows, list) and len(repaired_rows) == len(invalid_rows):
                    repaired_valid = [
                        row
                        for row in repaired_rows
                        if isinstance(row, dict)
                        and str(row.get("parent_record_id", "")) in seed_by_id
                    ]
                    if len(repaired_valid) == len(invalid_rows):
                        valid_rows.extend(repaired_valid)
                        self.last_lineage_repairs = len(invalid_rows)
                    else:
                        self.last_lineage_drops = len(invalid_rows)
                else:
                    self.last_lineage_drops = len(invalid_rows)
            else:
                self.last_lineage_drops = len(invalid_rows)
        generated = []
        for index, row in enumerate(valid_rows):
            if not isinstance(row, dict):
                raise ValueError("each generated candidate must be an object")
            parent_id = str(row.get("parent_record_id", ""))
            input_text = str(row.get("input", "")).strip()
            target = row.get("target")
            if not input_text or target is None or not str(target).strip():
                raise ValueError("generated candidates need non-empty input and target")
            digest = hashlib.sha256(
                f"{round_index}|{strategy}|{parent_id}|{input_text}|{target}".encode()
            ).hexdigest()[:16]
            parent = seed_by_id[parent_id]
            generated.append(
                TaskRecord(
                    record_id=f"generated-{round_index}-{index}-{digest}",
                    task_id=parent.task_id,
                    split="seed",
                    input=input_text,
                    target=str(target),
                    source_group_id=parent.source_group_id or parent.record_id,
                    metadata={
                        "generator_strategy": strategy,
                        "generator_model": self.client.model,
                        "quality_slice": str(
                            row.get("quality_slice") or parent.task_id
                        ),
                        "review_cost": float(row.get("review_cost", 30.0)),
                    },
                    parent_record_ids=(parent_id,),
                )
            )
        delta = Budget(
            llm_calls=self.client.budget.llm_calls - budget_before.llm_calls,
            llm_tokens=self.client.completion_tokens - completion_before,
        )
        return generated, delta


class OfflineReplayCandidateGenerator:
    """Replay pre-labeled records while hiding their labels until feedback."""

    def __init__(self, source_records: Sequence[TaskRecord]):
        if not source_records:
            raise ValueError("offline replay source pool cannot be empty")
        missing = [item.record_id for item in source_records if item.target is None]
        if missing:
            raise ValueError(
                "offline replay source records need labels; missing "
                f"target for {missing[:3]}"
            )
        self.source_records = tuple(source_records)
        self._source_by_id = {item.record_id: item for item in source_records}
        if len(self._source_by_id) != len(self.source_records):
            raise ValueError("offline replay source record_ids must be unique")
        self.last_lineage_repairs = 0
        self.last_lineage_drops = 0

    @property
    def budget(self) -> Budget:
        return Budget()

    def generate(
        self,
        *,
        seed_records: Sequence[TaskRecord],
        requirements: Sequence[RequirementHypothesis],
        confirmed_edits: Sequence[str],
        strategy: str,
        count: int,
        round_index: int,
        cost_budget: float | None,
    ) -> tuple[list[TaskRecord], Budget]:
        del requirements, confirmed_edits, strategy, cost_budget
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return [], Budget()
        if not seed_records:
            raise ValueError("offline replay requires at least one seed record")
        ranked_sources = sorted(
            self.source_records,
            key=lambda item: hashlib.sha256(
                f"offline-replay|{round_index}|{item.record_id}".encode("utf-8")
            ).hexdigest(),
        )
        ranked_seeds = sorted(
            seed_records,
            key=lambda item: hashlib.sha256(
                f"offline-replay-parent|{round_index}|{item.record_id}".encode("utf-8")
            ).hexdigest(),
        )
        generated: list[TaskRecord] = []
        for index, source in enumerate(ranked_sources[:count]):
            parent = ranked_seeds[index % len(ranked_seeds)]
            digest = hashlib.sha256(
                f"{round_index}|{source.record_id}|{parent.record_id}".encode("utf-8")
            ).hexdigest()[:16]
            metadata = {
                "offline_replay": True,
                "offline_source_record_id": source.record_id,
                "dataset": source.metadata.get("dataset", "unknown"),
                "quality_slice": str(
                    source.metadata.get("quality_slice") or source.task_id
                ),
                "review_cost": float(source.metadata.get("review_cost", 30.0)),
                "label_release_policy": "released only by OfflineReplayFeedbackReviewer",
            }
            generated.append(
                TaskRecord(
                    record_id=f"replay-{round_index}-{index}-{digest}",
                    task_id=source.task_id,
                    split="seed",
                    input=source.input,
                    target=None,
                    source_group_id=source.source_group_id or source.record_id,
                    metadata=metadata,
                    parent_record_ids=(parent.record_id,),
                )
            )
        return generated, Budget()


class OfflineReplayFeedbackReviewer:
    """Release an existing human label only when its candidate is queried."""

    def __init__(self, source_records: Sequence[TaskRecord]):
        self._source_by_id = {item.record_id: item for item in source_records}

    def review(self, record: TaskRecord, query: FeedbackQuery) -> HumanFeedback:
        source_id = record.metadata.get("offline_source_record_id")
        source = self._source_by_id.get(str(source_id))
        if source is None or source.target is None:
            raise ValueError(
                f"offline replay source label is unavailable for {record.record_id}"
            )
        return HumanFeedback(
            accept_for_training=True,
            target=str(source.target),
            edited_requirement=None,
            review_seconds=query.review_cost,
        )


class MappingFeedbackReviewer:
    """Use explicit human feedback loaded from a file; missing rows stay deferred."""

    def __init__(self, feedback: dict[str, HumanFeedback]):
        self.feedback = feedback

    def review(self, record: TaskRecord, query: FeedbackQuery) -> HumanFeedback | None:
        del query
        return self.feedback.get(record.record_id)


class MappingQualityAuditor:
    """Independent quality labels loaded separately from training feedback."""

    def __init__(self, audits: dict[str, bool]):
        self.audits = audits

    def audit(self, record: TaskRecord, proposed_decision: str) -> bool | None:
        del proposed_decision
        return self.audits.get(record.record_id)


class ConsoleFeedbackReviewer:
    """Interactive reviewer. Input is recorded as human feedback, never simulated."""

    def review(self, record: TaskRecord, query: FeedbackQuery) -> HumanFeedback | None:
        print("\n--- HUMAN REVIEW ---")
        print(f"record_id: {record.record_id}")
        print(f"input: {record.input}")
        print(f"proposed target: {record.target}")
        print(f"query score: {query.score:.4f}; estimated seconds: {query.review_cost:.1f}")
        answer = input("accept [y], reject [n], or skip [s]: ").strip().lower()
        if answer in ("", "s", "skip"):
            return None
        if answer not in ("y", "yes", "n", "no"):
            raise ValueError("review response must be y, n, or s")
        target = None
        edit = None
        if answer in ("y", "yes"):
            target = input("corrected target (blank keeps proposed): ").strip() or None
            edit = input("confirmed requirement edit (optional): ").strip() or None
        seconds_text = input("review seconds (blank uses estimate): ").strip()
        seconds = float(seconds_text) if seconds_text else query.review_cost
        return HumanFeedback(answer in ("y", "yes"), target, edit, seconds)


def load_task_records(path: str | Path) -> list[TaskRecord]:
    source = Path(path)
    rows = []
    # JSONL records are separated by LF; splitlines() also treats Unicode
    # paragraph separators inside valid JSON strings as record boundaries.
    for line_number, line in enumerate(source.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {source}:{line_number}") from exc
        required = ("record_id", "task_id", "split", "input")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Missing {missing} on {source}:{line_number}")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be an object on {source}:{line_number}")
        rows.append(
            TaskRecord(
                record_id=str(row["record_id"]),
                task_id=str(row["task_id"]),
                split=str(row["split"]),
                input=row["input"],
                target=row.get("target"),
                source_group_id=(
                    str(row["source_group_id"])
                    if row.get("source_group_id") is not None
                    else None
                ),
                metadata=metadata,
                parent_record_ids=tuple(str(item) for item in row.get("parent_record_ids", [])),
            )
        )
    if not rows:
        raise ValueError(f"No records found in {source}")
    return rows


def load_requirement_state(path: str | Path) -> RequirementState:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = value.get("hypotheses")
    if not isinstance(rows, list) or not rows:
        raise ValueError("requirements file needs a non-empty hypotheses list")
    hypotheses = [
        RequirementHypothesis(str(row["id"]), str(row["text"])) for row in rows
    ]
    weights = value.get("weights")
    state = (
        RequirementState(hypotheses, {str(k): float(v) for k, v in weights.items()}, [])
        if isinstance(weights, dict)
        else RequirementState.uniform(hypotheses)
    )
    for edit in value.get("confirmed_edits", []):
        state.add_confirmed_edit(str(edit))
    return state


def load_calibrator(path: str | Path) -> ReliabilityCalibrator:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    calibrator = ReliabilityCalibrator(
        minimum_audits=int(value.get("minimum_audits", 20)),
        minimum_agreement_lower_bound=float(
            value.get("minimum_agreement_lower_bound", 0.8)
        ),
        confidence_z=float(value.get("confidence_z", 1.96)),
    )
    for row in value.get("audits", []):
        calibrator.add_audit(
            str(row["slice"]),
            str(row["proposed_decision"]),
            bool(row["human_accept"]),
        )
    return calibrator


def load_feedback(path: str | Path) -> dict[str, HumanFeedback]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = value.get("feedback", value)
    if not isinstance(rows, dict):
        raise ValueError("feedback file must contain an object keyed by record_id")
    return {
        str(record_id): HumanFeedback(
            accept_for_training=bool(row["accept_for_training"]),
            target=(str(row["target"]) if row.get("target") is not None else None),
            edited_requirement=(
                str(row["edited_requirement"])
                if row.get("edited_requirement") is not None
                else None
            ),
            review_seconds=float(row.get("review_seconds", 0.0)),
        )
        for record_id, row in rows.items()
    }


def load_quality_audits(path: str | Path) -> dict[str, bool]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = value.get("audits", value)
    if not isinstance(rows, dict):
        raise ValueError("quality audit file must contain an object keyed by record_id")
    return {str(record_id): bool(row) for record_id, row in rows.items()}


def api_key_from_env(name: str | None) -> str | None:
    return os.environ.get(name) if name else None
