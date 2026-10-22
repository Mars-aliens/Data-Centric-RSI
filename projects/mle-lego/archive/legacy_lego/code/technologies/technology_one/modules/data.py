from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_system.mle_core.models import TaskRecord, TaskSpec
from core_system.mle_core.registry import TaskRegistry


TASK_ID = "lego-local-intent"


@dataclass(frozen=True)
class StreamBatch:
    batch_id: str
    records: tuple[TaskRecord, ...]


def build_lego_scenario_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register_task(
        TaskSpec(
            task_id=TASK_ID,
            name="LEGO local harness scenario",
            input_schema="short English text",
            target_schema="intent label",
            allowed_splits=("seed", "held_in", "held_out", "final_test"),
        )
    )
    registry.add_records(
        [
            TaskRecord("seed-account-1", TASK_ID, "seed", "reset my password", "account"),
            TaskRecord("seed-account-2", TASK_ID, "seed", "my account is locked", "account"),
            TaskRecord("seed-billing-1", TASK_ID, "seed", "where is my invoice", "billing"),
            TaskRecord("seed-billing-2", TASK_ID, "seed", "send the receipt", "billing"),
            TaskRecord(
                "held-in-billing", TASK_ID, "held_in", "the payment receipt is missing", "billing"
            ),
            TaskRecord(
                "held-in-billing-2", TASK_ID, "held_in", "I need my payment receipt", "billing"
            ),
            TaskRecord(
                "held-out-account", TASK_ID, "held_out", "help with sign in details", "account"
            ),
            TaskRecord(
                "held-out-billing", TASK_ID, "held_out", "I need a copy of my payment bill", "billing"
            ),
            TaskRecord(
                "final-account", TASK_ID, "final_test", "my login does not work", "account"
            ),
            TaskRecord(
                "final-billing", TASK_ID, "final_test", "where is the bill for my order", "billing"
            ),
        ]
    )
    return registry


def lego_stream_batches() -> tuple[StreamBatch, ...]:
    def record(record_id: str, text: str, label: str, route: str, quality: float) -> TaskRecord:
        return TaskRecord(
            record_id,
            TASK_ID,
            "seed",
            text,
            label,
            metadata={
                "batch_id": "",
                "route": route,
                "quality": quality,
                "accepted": True,
                "source": "controlled_stream",
            },
        )

    return (
        StreamBatch(
            "batch-0001",
            (
                record("stream-account-1", "help me unlock access", "account", "account", 0.95),
            ),
        ),
        StreamBatch(
            "batch-0002",
            (
                record("stream-billing-1", "my payment receipt is missing", "billing", "billing", 0.78),
            ),
        ),
        StreamBatch(
            "batch-0003",
            (
                record("stream-account-2", "I forgot my login", "account", "account", 0.95),
                record("stream-billing-2", "please send my payment bill", "billing", "billing", 0.78),
            ),
        ),
    )


def record_to_dict(record: TaskRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "task_id": record.task_id,
        "split": record.split,
        "input": record.input,
        "target": record.target,
        "metadata": record.metadata,
    }

