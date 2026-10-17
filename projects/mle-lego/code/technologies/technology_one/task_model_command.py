from __future__ import annotations

"""JSON stdin/stdout trainer command for the LEGO Action System."""

import json
import sys
import tempfile
from pathlib import Path

from core_system.mle_core.models import TaskRecord
from core_system.mle_core.sklearn_components import TfidfLogisticTrainer
from core_system.mle_core.store import ArtifactStore


def _records(raw: object) -> list[TaskRecord]:
    if not isinstance(raw, list):
        raise ValueError("trainer request records must be a list")
    records: list[TaskRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("trainer record must be an object")
        records.append(
            TaskRecord(
                record_id=str(item["record_id"]),
                task_id=str(item["task_id"]),
                split=str(item["split"]),
                input=item["input"],
                target=item.get("target"),
                source_group_id=item.get("source_group_id"),
                metadata=dict(item.get("metadata", {})),
                parent_record_ids=tuple(item.get("parent_record_ids", [])),
            )
        )
    return records


def train_once(request: dict[str, object]) -> dict[str, object]:
    if not isinstance(request, dict) or request.get("protocol") != "lego-trainer-v1":
        raise ValueError("Expected a lego-trainer-v1 JSON request")
    config = request.get("config")
    if not isinstance(config, dict):
        raise ValueError("trainer request config must be an object")
    trainer = TfidfLogisticTrainer()
    with tempfile.TemporaryDirectory(prefix="lego-trainer-") as directory:
        checkpoint, cost = trainer.train(
            train_records=_records(request.get("records")),
            initial_checkpoint=request.get("initial_checkpoint"),
            config=dict(config),
            seed=int(request.get("seed", 17)),
            store=ArtifactStore(Path(directory)),
        )
    return {
        "checkpoint": checkpoint,
        "cost": {
            "training_steps": cost.training_steps,
            "training_seconds": cost.training_seconds,
        },
    }


def main() -> None:
    response = train_once(json.load(sys.stdin))
    json.dump(response, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
