from __future__ import annotations

"""JSON stdin/stdout evaluator command for the LEGO Action System."""

import json
import sys
from typing import Any

from sklearn.metrics import f1_score

from core_system.mle_core.models import TaskRecord
from core_system.mle_core.sklearn_components import _checkpoint_scores


def _records(raw: object) -> list[TaskRecord]:
    if not isinstance(raw, list):
        raise ValueError("evaluator request records must be a list")
    records: list[TaskRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("evaluator record must be an object")
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


def evaluate_once(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("protocol") != "lego-evaluator-v1":
        raise ValueError("Expected a lego-evaluator-v1 JSON request")
    checkpoint = request.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("evaluator request checkpoint must be an object")
    records = _records(request.get("records"))
    labeled = [record for record in records if record.target is not None]
    labels: list[str] = []
    predictions: list[str] = []
    for record in labeled:
        scores = _checkpoint_scores(checkpoint, str(record.input))
        labels.append(str(record.target))
        predictions.append(max(scores, key=lambda label: (scores[label], label)))
    accuracy = sum(left == right for left, right in zip(labels, predictions)) / len(labels) if labels else 0.0
    return {
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)) if labels else 0.0,
            "num_records": float(len(labeled)),
        },
        "cost": {"evaluator_calls": 1},
    }


def main() -> None:
    response = evaluate_once(json.load(sys.stdin))
    json.dump(response, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
