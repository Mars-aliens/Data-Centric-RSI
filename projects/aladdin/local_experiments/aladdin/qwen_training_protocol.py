"""Fairness checks for the real Qwen downstream training experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = (
    "base_model",
    "max_steps",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "max_length",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "eval_batch_size",
)


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"missing training protocol keys: {missing}")
    positive = ("max_steps", "per_device_batch_size", "gradient_accumulation_steps", "max_length", "eval_batch_size")
    for key in positive:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(config["lora_dropout"]) < 0:
        raise ValueError("lora_dropout must be non-negative")
    if not 0 <= float(config["lora_dropout"]) < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if int(config["lora_r"]) <= 0 or int(config["lora_alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    return {
        "status": "ready",
        "training_budget_equalized": True,
        "evaluation_batch_size": int(config["eval_batch_size"]),
        "config": config,
    }


def validate_matrix(matrix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked = {name: validate_protocol(config) for name, config in matrix.items()}
    if not checked:
        raise ValueError("training matrix must contain at least one strategy")
    reference = {key: checked[next(iter(checked))]["config"][key] for key in REQUIRED_KEYS}
    mismatches = {
        name: {key: value["config"][key] for key in reference if value["config"][key] != reference[key]}
        for name, value in checked.items()
    }
    mismatches = {name: values for name, values in mismatches.items() if values}
    if mismatches:
        raise ValueError(f"strategy training budgets differ: {mismatches}")
    return {"status": "ready", "strategies": list(matrix), "checks": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(validate_matrix(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
