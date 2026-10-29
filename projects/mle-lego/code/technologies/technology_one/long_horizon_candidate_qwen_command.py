from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

from .long_horizon_qwen_command import _training_records
from .candidate_strategy import validate_candidate_strategy
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
    if not isinstance(config.get("base_model_path"), str) or not config["base_model_path"].strip():
        raise ValueError("candidate task model path is required")
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
    raw_seed = request.get("seed_examples")
    raw_feedback = request.get("released_feedback")
    if not isinstance(raw_seed, list) or not isinstance(raw_feedback, list):
        raise ValueError("seed_examples and released_feedback must be lists")
    data_processing = {
        "strategy_sha256": None,
        "seed_records_seen": len(raw_seed),
        "feedback_records_seen": len(raw_feedback),
        "training_records_selected": 0,
        "training_records_dropped": 0,
    }
    training_cost: dict[str, int | float] = {
        "training_steps": 0,
        "training_records_seen": 0,
        "training_seconds": 0.0,
    }
    strategy = validate_candidate_strategy(
        config.get("candidate_strategy", {
            "schema": "lego-t1-data-strategy-v1",
            "data_policy": {
                "min_input_chars": 1,
                "deduplicate_inputs": True,
                "max_feedback_examples": None,
            },
        })
    )
    data_processing["strategy_sha256"] = hashlib.sha256(
        json.dumps(strategy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if raw_feedback:
        policy = strategy["data_policy"]
        records_by_id: dict[str, dict[str, str]] = {}
        selected_inputs: set[str] = set()
        selected_feedback: list[dict[str, str]] = []
        for source, rows in (("seed", raw_seed), ("feedback", raw_feedback)):
            for raw in rows:
                if not isinstance(raw, dict):
                    raise ValueError("training examples must be objects")
                normalized = _training_records([raw])[0]
                text = normalized["input"]
                if len(text) < policy["min_input_chars"]:
                    continue
                key = " ".join(text.casefold().split())
                if policy["deduplicate_inputs"] and key in selected_inputs:
                    continue
                selected_inputs.add(key)
                if normalized["record_id"] in records_by_id:
                    raise ValueError(f"duplicate released training record: {normalized['record_id']}")
                records_by_id[normalized["record_id"]] = normalized
                if source == "feedback":
                    selected_feedback.append(normalized)
        max_feedback = policy["max_feedback_examples"]
        feedback_count_before_cap = len(selected_feedback)
        if max_feedback is not None and len(selected_feedback) > max_feedback:
            keep_ids = {item["record_id"] for item in selected_feedback[-max_feedback:]}
            records_by_id = {
                key: value for key, value in records_by_id.items()
                if key not in {item["record_id"] for item in selected_feedback} or key in keep_ids
            }
        if not records_by_id:
            raise ValueError("candidate data policy removed every training example")
        data_processing["training_records_selected"] = len(records_by_id)
        data_processing["training_records_dropped"] = (
            len(raw_seed) + len(raw_feedback) - len(records_by_id)
        )
        data_processing["feedback_records_dropped_by_cap"] = (
            feedback_count_before_cap - min(feedback_count_before_cap, max_feedback)
            if max_feedback is not None else 0
        )
        artifact_root = Path(config.get("candidate_output_path", "/candidate-output/checkpoint"))
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
        "data_processing": data_processing,
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
