from __future__ import annotations

"""JSON trainer command for the Qwen + LoRA HelpSteer2 task model."""

import json
import os
import sys
from pathlib import Path

from .qwen_preference import _records, train_checkpoint


def train_once(request: dict[str, object]) -> dict[str, object]:
    if not isinstance(request, dict) or request.get("protocol") != "lego-trainer-v1":
        raise ValueError("Expected a lego-trainer-v1 JSON request")
    config = request.get("config")
    if not isinstance(config, dict):
        raise ValueError("trainer request config must be an object")
    output_root = request.get("artifact_store_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("trainer request artifact_store_root is required")
    output_dir = Path(output_root) / "qwen_lora_adapter"
    checkpoint, cost = train_checkpoint(
        _records(request.get("records")),
        config=config,
        seed=int(request.get("seed", 17)),
        output_dir=output_dir,
    )
    return {"checkpoint": checkpoint, "cost": cost}


def main() -> None:
    response = train_once(json.load(sys.stdin))
    json.dump(response, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
