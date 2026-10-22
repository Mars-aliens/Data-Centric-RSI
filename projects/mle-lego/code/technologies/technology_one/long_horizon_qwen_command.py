from __future__ import annotations

"""Qwen command boundary for the new long-horizon task stream only."""

import json
import sys
import time
from pathlib import Path
from typing import Any

from .qwen_preference import LABEL_TO_TEXT, predict, train_checkpoint


_VALID_TARGETS = {value.lower() for value in LABEL_TO_TEXT.values()}


def _training_records(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("records must be a non-empty list of labeled records")
    result: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"training record {index} must be an object")
        record_id = item.get("record_id")
        input_text = item.get("input")
        target = item.get("target")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"training record {index} must include a non-empty record_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate training record_id: {record_id}")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"training record {record_id} must include non-empty input")
        if not isinstance(target, str) or target.strip().lower() not in _VALID_TARGETS:
            raise ValueError(f"training record {record_id} must include a valid labeled record target")
        seen_ids.add(record_id)
        result.append(
            {"record_id": record_id, "input": input_text, "target": target.strip().lower()}
        )
    return result


def train_once(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("protocol") != "long-horizon-qwen-v1":
        raise ValueError("Expected a long-horizon-qwen-v1 JSON request")
    if request.get("operation") != "train":
        raise ValueError("train operation is required")
    config = request.get("config")
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    if not isinstance(config.get("base_model_path"), str) or not config["base_model_path"].strip():
        raise ValueError("config must set base_model_path to a local Hugging Face model directory")
    if "base_model_id" in config and (
        not isinstance(config["base_model_id"], str) or not config["base_model_id"].strip()
    ):
        raise ValueError("base_model_id must be a non-empty model identifier when provided")
    base_revision = config.get("base_model_revision")
    if (
        not isinstance(base_revision, str)
        or not base_revision.strip()
        or base_revision.strip().lower() in {"latest", "main", "master", "unrecorded"}
        or (base_revision.strip().startswith("<") and base_revision.strip().endswith(">"))
    ):
        raise ValueError("config must set base_model_revision to an immutable model revision")
    artifact_root = request.get("artifact_store_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise ValueError("artifact_store_root is required")
    parent = request.get("initial_checkpoint")
    if parent is not None and not isinstance(parent, dict):
        raise ValueError("initial_checkpoint must be null or a checkpoint object")
    output_dir = Path(artifact_root) / "long_horizon_qwen_adapter"
    checkpoint, cost = train_checkpoint(
        _training_records(request.get("records")),
        config=config,
        seed=int(request.get("seed", 17)),
        output_dir=output_dir,
        initial_checkpoint=parent,
    )
    return {"checkpoint": checkpoint, "cost": cost}


def predict_once(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("protocol") != "long-horizon-qwen-v1":
        raise ValueError("Expected a long-horizon-qwen-v1 JSON request")
    if request.get("operation") != "predict":
        raise ValueError("predict operation is required")
    checkpoint = request.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be an object")
    raw_records = request.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("records must be a non-empty list")
    records: list[dict[str, str]] = []
    record_ids: list[str] = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise ValueError("each prediction record must be an object")
        record_id, input_text = item.get("record_id"), item.get("input")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("prediction record_id must be non-empty")
        if not isinstance(input_text, str) or not input_text:
            raise ValueError("prediction input must be non-empty")
        if "target" in item or "scores" in item:
            raise ValueError("prediction records must not contain targets or scores")
        if record_id in record_ids:
            raise ValueError(f"duplicate prediction record_id: {record_id}")
        record_ids.append(record_id)
        records.append({"input": input_text})
    batch_size = int(request.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    start = time.perf_counter()
    raw_predictions = predict(checkpoint, records, batch_size=batch_size)
    if len(raw_predictions) != len(record_ids):
        raise ValueError("task model returned a different number of predictions than inputs")
    predictions = []
    for label in raw_predictions:
        if not isinstance(label, str) or label not in LABEL_TO_TEXT:
            raise ValueError(f"task model returned an unsupported label token: {label!r}")
        predictions.append(LABEL_TO_TEXT[label])
    elapsed = time.perf_counter() - start
    return {
        "predictions": [
            {"record_id": record_id, "prediction": prediction}
            for record_id, prediction in zip(record_ids, predictions, strict=True)
        ],
        "cost": {"prediction_records": len(records), "prediction_seconds": elapsed},
    }


def main() -> None:
    request = json.load(sys.stdin)
    operation = request.get("operation") if isinstance(request, dict) else None
    if operation == "train":
        response = train_once(request)
    elif operation == "predict":
        response = predict_once(request)
    else:
        raise ValueError("operation must be 'train' or 'predict'")
    json.dump(response, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
