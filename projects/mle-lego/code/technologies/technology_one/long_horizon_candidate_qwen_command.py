from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .long_horizon_qwen_command import _training_records
from .qwen_preference import LABEL_TO_TEXT, predict, train_checkpoint


def run_step(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("protocol") != "lego-t1-candidate-step-v1":
        raise ValueError("Expected a lego-t1-candidate-step-v1 request")
    candidate_id = request.get("candidate_id")
    parent_version_id = request.get("parent_version_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id is required")
    if not isinstance(parent_version_id, str) or not parent_version_id:
        raise ValueError("parent_version_id is required")
    config = request.get("config")
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    if config.get("base_model_path") != "/models/task":
        raise ValueError("candidate task model must use the read-only /models/task mount")
    checkpoint = request.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is required")

    raw_records = request.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("records must be a non-empty list")
    records = []
    record_ids: set[str] = set()
    ordered_record_ids: list[str] = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise ValueError("each record must be an object")
        if "target" in item or "scores" in item:
            raise ValueError("candidate current-batch inputs must not include labels")
        record_id, input_text = item.get("record_id"), item.get("input")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("current batch has an invalid or duplicate record_id")
        if not isinstance(input_text, str) or not input_text:
            raise ValueError(f"current batch input is missing for {record_id}")
        record_ids.add(record_id)
        ordered_record_ids.append(record_id)
        records.append({"input": input_text})

    started = time.perf_counter()
    predicted = predict(
        checkpoint,
        records,
        batch_size=int(request.get("config", {}).get("prediction_batch_size", 8)),
    )
    if len(predicted) != len(records):
        raise ValueError("task model returned a different number of predictions than inputs")
    predictions = [
        {"record_id": record_id, "prediction": LABEL_TO_TEXT[label]}
        for record_id, label in zip(ordered_record_ids, predicted, strict=True)
    ]
    prediction_seconds = time.perf_counter() - started

    next_checkpoint = checkpoint
    training_cost: dict[str, int | float] = {
        "training_steps": 0,
        "training_records_seen": 0,
        "training_seconds": 0.0,
    }
    raw_seed = request.get("seed_examples")
    raw_feedback = request.get("released_feedback")
    if not isinstance(raw_seed, list) or not isinstance(raw_feedback, list):
        raise ValueError("seed_examples and released_feedback must be lists")
    if raw_feedback:
        records_by_id: dict[str, dict[str, str]] = {}
        for raw in [*raw_seed, *raw_feedback]:
            if not isinstance(raw, dict):
                raise ValueError("training examples must be objects")
            normalized = _training_records([raw])[0]
            if normalized["record_id"] in records_by_id:
                raise ValueError(f"duplicate released training record: {normalized['record_id']}")
            records_by_id[normalized["record_id"]] = normalized
        artifact_root = Path("/candidate-output") / "checkpoint"
        next_checkpoint, training_cost = train_checkpoint(
            list(records_by_id.values()),
            config=config,
            seed=int(request.get("seed", 17)),
            output_dir=artifact_root,
            initial_checkpoint=checkpoint,
        )

    return {
        "protocol": "lego-t1-candidate-step-v1",
        "candidate_id": candidate_id,
        "parent_version_id": parent_version_id,
        "predictions": predictions,
        "checkpoint": next_checkpoint,
        "cost": {
            **training_cost,
            "prediction_records": len(records),
            "prediction_seconds": prediction_seconds,
        },
    }


def main() -> None:
    request = json.load(sys.stdin)
    response = run_step(request)
    json.dump(response, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
