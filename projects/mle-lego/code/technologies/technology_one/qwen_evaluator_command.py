from __future__ import annotations

"""JSON evaluator command for the Qwen + LoRA HelpSteer2 task model."""

import json
import os
import sys
from typing import Any

from .qwen_preference import _records, predict


def _macro_f1(labels: list[str], predictions: list[str]) -> float:
    scores: list[float] = []
    for label in ("preferred", "rejected", "tie"):
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(labels, predictions, strict=True)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(labels, predictions, strict=True)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(labels, predictions, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def evaluate_once(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("protocol") != "lego-evaluator-v1":
        raise ValueError("Expected a lego-evaluator-v1 JSON request")
    checkpoint = request.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("evaluator request checkpoint must be an object")
    records = _records(request.get("records"))
    config = request.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("evaluator request config must be an object")
    predictions = predict(
        checkpoint,
        records,
        batch_size=int(config.get("batch_size") or os.environ.get("LEGO_EVAL_BATCH_SIZE", 8)),
    )
    labels = [str(record["target"]).strip().lower() for record in records]
    accuracy = sum(
        left == right for left, right in zip(labels, [checkpoint["label_map"][item] for item in predictions], strict=True)
    ) / len(labels)
    decoded = [checkpoint["label_map"][item] for item in predictions]
    return {
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": _macro_f1(labels, decoded),
            "num_records": float(len(labels)),
        },
        "cost": {"evaluator_calls": 1},
    }


def main() -> None:
    response = evaluate_once(json.load(sys.stdin))
    json.dump(response, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
