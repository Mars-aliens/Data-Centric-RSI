from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import TaskRecord, TaskSpec


class SplitViolation(ValueError):
    pass


@dataclass
class TaskRegistry:
    tasks: dict[str, TaskSpec]
    records: dict[str, TaskRecord]

    def __init__(self) -> None:
        self.tasks = {}
        self.records = {}

    def register_task(self, task: TaskSpec) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Task already registered: {task.task_id}")
        self.tasks[task.task_id] = task

    def add_records(self, records: Iterable[TaskRecord]) -> None:
        for record in records:
            task = self.tasks.get(record.task_id)
            if task is None:
                raise KeyError(f"Unknown task: {record.task_id}")
            if record.split not in task.allowed_splits:
                raise SplitViolation(
                    f"Split {record.split!r} is not allowed for task {record.task_id}"
                )
            if record.record_id in self.records:
                raise ValueError(f"Record already registered: {record.record_id}")
            self.records[record.record_id] = record

    def task(self, task_id: str) -> TaskSpec:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task: {task_id}") from exc

    def records_for(
        self,
        task_id: str,
        *,
        split: str | None = None,
        include_final_test: bool = False,
    ) -> list[TaskRecord]:
        if split == "final_test" and not include_final_test:
            raise SplitViolation("final_test requires explicit reporting access")
        records = [
            record
            for record in self.records.values()
            if record.task_id == task_id and (split is None or record.split == split)
        ]
        if not include_final_test:
            records = [record for record in records if record.split != "final_test"]
        return sorted(records, key=lambda item: item.record_id)

    def validate_provenance(self, record: TaskRecord) -> None:
        for parent_id in record.parent_record_ids:
            parent = self.records.get(parent_id)
            if parent is None:
                raise SplitViolation(f"Unknown parent record: {parent_id}")
            if parent.split != record.split:
                raise SplitViolation(
                    f"Derived record {record.record_id} crosses split boundary "
                    f"from {parent.split} to {record.split}"
                )
