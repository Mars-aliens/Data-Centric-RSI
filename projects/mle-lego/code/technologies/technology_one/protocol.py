from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core_system.mle_core.models import TaskRecord


@dataclass(frozen=True)
class DataProtocol:
    task_id: str
    required_fields: tuple[str, ...] = ("record_id", "split", "input", "target")
    allowed_splits: tuple[str, ...] = ("seed", "development", "held_in", "held_out", "final_test")
    batch_field: str = "batch_id"
    feedback_field: str = "feedback"
    delayed_feedback_batches: tuple[int, ...] = (0, 2, 5)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataProtocol":
        feedback = value.get("feedback", {})
        return cls(
            task_id=str(value["task_id"]),
            required_fields=tuple(value.get("required_fields", cls.required_fields)),
            allowed_splits=tuple(value.get("allowed_splits", cls.allowed_splits)),
            batch_field=str(value.get("batch_field", "batch_id")),
            feedback_field=str(value.get("feedback_field", "feedback")),
            delayed_feedback_batches=tuple(
                feedback.get("allowed_delays", value.get("delayed_feedback_batches", (0, 2, 5)))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "required_fields": list(self.required_fields),
            "allowed_splits": list(self.allowed_splits),
            "batch_field": self.batch_field,
            "feedback_field": self.feedback_field,
            "delayed_feedback_batches": list(self.delayed_feedback_batches),
        }


def load_protocol(path: str | Path) -> DataProtocol:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return DataProtocol.from_dict(value)


@dataclass(frozen=True)
class DelayedFeedback:
    feedback_id: str
    target_record_id: str
    source_batch: int
    release_batch: int
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class DelayedFeedbackQueue:
    def __init__(self, feedback: list[DelayedFeedback] | None = None):
        self._pending = list(feedback or [])
        self._released: set[str] = set()

    def release_for_batch(self, batch_index: int) -> list[DelayedFeedback]:
        ready = [
            item
            for item in self._pending
            if item.feedback_id not in self._released
            and item.release_batch <= batch_index
        ]
        self._released.update(item.feedback_id for item in ready)
        return sorted(ready, key=lambda item: (item.release_batch, item.feedback_id))


class JsonlTaskAdapter:
    """Load a future real dataset without coupling it to the toy scenario."""

    def __init__(self, protocol: DataProtocol):
        self.protocol = protocol

    def load(self, path: str | Path) -> list[TaskRecord]:
        records: list[TaskRecord] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                missing = set(self.protocol.required_fields) - set(item)
                if missing:
                    raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
                if item["split"] not in self.protocol.allowed_splits:
                    raise ValueError(f"Line {line_number} has unsupported split {item['split']!r}")
                metadata = dict(item.get("metadata", {}))
                if self.protocol.batch_field in item:
                    metadata[self.protocol.batch_field] = item[self.protocol.batch_field]
                if self.protocol.feedback_field in item:
                    metadata[self.protocol.feedback_field] = item[self.protocol.feedback_field]
                records.append(
                    TaskRecord(
                        record_id=str(item["record_id"]),
                        task_id=str(item.get("task_id", self.protocol.task_id)),
                        split=str(item["split"]),
                        input=item["input"],
                        target=item.get("target"),
                        source_group_id=item.get("source_group_id"),
                        metadata=metadata,
                        parent_record_ids=tuple(item.get("parent_record_ids", [])),
                    )
                )
        return records

    def group_batches(self, records: Iterable[TaskRecord]) -> dict[str, list[TaskRecord]]:
        grouped: dict[str, list[TaskRecord]] = {}
        for record in records:
            batch_id = str(record.metadata.get(self.protocol.batch_field, "unassigned"))
            grouped.setdefault(batch_id, []).append(record)
        return grouped

