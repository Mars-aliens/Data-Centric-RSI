from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataSystemVersion:
    """Version vector for one runnable Data System state."""

    version_id: str
    data_version: str
    task_model_version: str
    harness_version: str
    memory_version: str
    skill_version: str
    evaluator_version: str
    tool_versions: tuple[tuple[str, str], ...] = ()
    parent_version_id: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.version_id,
            self.data_version,
            self.task_model_version,
            self.harness_version,
            self.memory_version,
            self.skill_version,
            self.evaluator_version,
        )
        if any(not value.strip() for value in required):
            raise ValueError("system version identifiers must be non-empty")
        names = [name for name, _ in self.tool_versions]
        if len(names) != len(set(names)):
            raise ValueError("tool version names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "data_version": self.data_version,
            "task_model_version": self.task_model_version,
            "harness_version": self.harness_version,
            "memory_version": self.memory_version,
            "skill_version": self.skill_version,
            "evaluator_version": self.evaluator_version,
            "tool_versions": dict(self.tool_versions),
            "parent_version_id": self.parent_version_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataSystemVersion":
        return cls(
            version_id=value["version_id"],
            data_version=value["data_version"],
            task_model_version=value["task_model_version"],
            harness_version=value["harness_version"],
            memory_version=value["memory_version"],
            skill_version=value["skill_version"],
            evaluator_version=value["evaluator_version"],
            tool_versions=tuple(sorted(value.get("tool_versions", {}).items())),
            parent_version_id=value.get("parent_version_id"),
        )


@dataclass(frozen=True)
class CandidateVersion:
    candidate_id: str
    version: DataSystemVersion
    created_run_id: str
    created_batch_id: str

    def __post_init__(self) -> None:
        if not all((self.candidate_id, self.created_run_id, self.created_batch_id)):
            raise ValueError("candidate identifiers must be non-empty")


@dataclass(frozen=True)
class FeedbackEvent:
    """Feedback keeps the version and batch that produced it, not just arrival time."""

    feedback_id: str
    target_version_id: str
    source_run_id: str
    source_batch_id: str
    source_batch_index: int
    released_batch_index: int
    feedback_type: str
    value: Any
    produced_at: str
    received_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (
            self.feedback_id,
            self.target_version_id,
            self.source_run_id,
            self.source_batch_id,
            self.feedback_type,
            self.produced_at,
            self.received_at,
        )
        if any(not value.strip() for value in required):
            raise ValueError("feedback provenance fields must be non-empty")
        if self.source_batch_index < 0:
            raise ValueError("source_batch_index cannot be negative")
        if self.released_batch_index < self.source_batch_index:
            raise ValueError("feedback cannot be released before its source batch")
