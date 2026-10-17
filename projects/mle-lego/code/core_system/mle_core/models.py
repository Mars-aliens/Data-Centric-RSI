from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING_FOR_FEEDBACK = "waiting_for_feedback"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    version: str
    content_hash: str


@dataclass(frozen=True)
class TaskRecord:
    record_id: str
    task_id: str
    split: str
    input: Any
    target: Any | None = None
    source_group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    input_schema: str
    target_schema: str
    allowed_splits: tuple[str, ...] = (
        "seed",
        "development",
        "held_out",
        "final_test",
    )


@dataclass(frozen=True)
class Budget:
    llm_calls: int = 0
    llm_tokens: int = 0
    training_steps: int = 0
    training_seconds: float = 0.0
    evaluator_calls: int = 0
    feedback_requests: int = 0
    wall_clock_seconds: float = 0.0

    def add(self, other: "Budget") -> "Budget":
        return Budget(
            llm_calls=self.llm_calls + other.llm_calls,
            llm_tokens=self.llm_tokens + other.llm_tokens,
            training_steps=self.training_steps + other.training_steps,
            training_seconds=self.training_seconds + other.training_seconds,
            evaluator_calls=self.evaluator_calls + other.evaluator_calls,
            feedback_requests=self.feedback_requests + other.feedback_requests,
            wall_clock_seconds=self.wall_clock_seconds + other.wall_clock_seconds,
        )

    def exceeds(self, limit: "Budget") -> bool:
        pairs = (
            ("llm_calls", self.llm_calls, limit.llm_calls),
            ("llm_tokens", self.llm_tokens, limit.llm_tokens),
            ("training_steps", self.training_steps, limit.training_steps),
            ("training_seconds", self.training_seconds, limit.training_seconds),
            ("evaluator_calls", self.evaluator_calls, limit.evaluator_calls),
            ("feedback_requests", self.feedback_requests, limit.feedback_requests),
            ("wall_clock_seconds", self.wall_clock_seconds, limit.wall_clock_seconds),
        )
        return any(limit_value >= 0 and value > limit_value for _, value, limit_value in pairs)


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    task_id: str
    backbone_llm: str
    task_model: str
    evaluator: str
    harness_version: str
    data_version: str
    initial_checkpoint: str | None
    max_batches: int
    budget_limit: Budget
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    run_id: str
    batch_id: str
    batch_index: int
    status: RunStatus
    task: TaskSpec
    config: RunConfig
    artifacts: tuple[ArtifactRef, ...]
    metrics: dict[str, float]
    pending_feedback: tuple[str, ...]
    remaining_budget: Budget
    feedback: tuple[FeedbackRecord, ...] = ()


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_versions: dict[str, str] = field(default_factory=dict)
    estimated_cost: Budget = field(default_factory=Budget)
    rationale: str = ""
    model_id: str | None = None
    model_revision: str | None = None
    system_prompt_hash: str | None = None
    user_prompt_hash: str | None = None
    input_artifact_ids: tuple[str, ...] = ()
    output_token_count: int = 0


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    target_artifact_id: str
    target_version: str
    feedback_type: str
    value: Any
    provider_type: str
    provider_version: str
    permitted_use: tuple[str, ...]
    requested_at: str
    received_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
