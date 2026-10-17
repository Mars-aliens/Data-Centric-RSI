from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Iterable

from core_system.mle_core.models import TaskRecord

from .protocol import DataProtocol


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    task_id: str
    record_count: int
    split_counts: dict[str, int]
    batch_count: int
    source_group_count: int
    group_size_histogram: dict[str, int]
    feedback_count: int
    delayed_feedback_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "record_count": self.record_count,
            "split_counts": self.split_counts,
            "batch_count": self.batch_count,
            "source_group_count": self.source_group_count,
            "group_size_histogram": self.group_size_histogram,
            "feedback_count": self.feedback_count,
            "delayed_feedback_count": self.delayed_feedback_count,
        }


def validate_records(
    records: Iterable[TaskRecord],
    protocol: DataProtocol,
) -> DatasetSummary:
    records = list(records)
    if not records:
        raise DatasetValidationError("Dataset contains no records")
    by_id: dict[str, TaskRecord] = {}
    split_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}
    group_splits: dict[str, str] = {}
    feedback_count = 0
    delayed_feedback_count = 0

    for record in records:
        if record.record_id in by_id:
            raise DatasetValidationError(f"Duplicate record_id: {record.record_id}")
        by_id[record.record_id] = record
        if record.task_id != protocol.task_id:
            raise DatasetValidationError(
                f"Record {record.record_id} has task_id {record.task_id!r}; "
                f"expected {protocol.task_id!r}"
            )
        if record.split not in protocol.allowed_splits:
            raise DatasetValidationError(
                f"Record {record.record_id} uses unsupported split {record.split!r}"
            )
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        group_id = record.source_group_id
        if group_id:
            previous_split = group_splits.setdefault(group_id, record.split)
            if previous_split != record.split:
                raise DatasetValidationError(
                    f"Source group {group_id!r} crosses split boundary: "
                    f"{previous_split!r} and {record.split!r}"
                )
        batch_id = record.metadata.get(protocol.batch_field)
        if record.split in {"seed", "development", "held_in", "held_out"} and not batch_id:
            raise DatasetValidationError(
                f"Record {record.record_id} has no {protocol.batch_field!r}"
            )
        if batch_id:
            batch_key = str(batch_id)
            batch_counts[batch_key] = batch_counts.get(batch_key, 0) + 1
        if protocol.feedback_field in record.metadata:
            feedback_count += 1
            feedback = record.metadata[protocol.feedback_field]
            if not isinstance(feedback, dict):
                raise DatasetValidationError(
                    f"Feedback on {record.record_id} must be an object"
                )
            source_batch = feedback.get("source_batch")
            release_batch = feedback.get("release_batch")
            if source_batch is not None or release_batch is not None:
                if not isinstance(source_batch, int) or not isinstance(release_batch, int):
                    raise DatasetValidationError(
                        f"Feedback on {record.record_id} needs integer source/release batches"
                    )
                if release_batch < source_batch:
                    raise DatasetValidationError(
                        f"Feedback on {record.record_id} is released before its source batch"
                    )
                if release_batch - source_batch not in protocol.delayed_feedback_batches:
                    raise DatasetValidationError(
                        f"Feedback on {record.record_id} uses unsupported delay "
                        f"{release_batch - source_batch}"
                    )
                delayed_feedback_count += 1

    for record in records:
        for parent_id in record.parent_record_ids:
            parent = by_id.get(parent_id)
            if parent is None:
                raise DatasetValidationError(
                    f"Record {record.record_id} references missing parent {parent_id}"
                )
            if parent.split != record.split:
                raise DatasetValidationError(
                    f"Record {record.record_id} crosses split boundary from parent {parent_id}"
                )

    group_sizes = {
        group_id: sum(record.source_group_id == group_id for record in records)
        for group_id in group_splits
    }
    return DatasetSummary(
        task_id=protocol.task_id,
        record_count=len(records),
        split_counts=split_counts,
        batch_count=len(batch_counts),
        source_group_count=len(group_splits),
        group_size_histogram=dict(
            sorted(Counter(str(size) for size in group_sizes.values()).items())
        ),
        feedback_count=feedback_count,
        delayed_feedback_count=delayed_feedback_count,
    )

