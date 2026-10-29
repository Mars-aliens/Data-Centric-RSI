from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STRATEGY_SCHEMA = "lego-t1-data-strategy-v1"
DEFAULT_STRATEGY = {
    "schema": STRATEGY_SCHEMA,
    "data_policy": {
        "min_input_chars": 1,
        "deduplicate_inputs": True,
        "max_feedback_examples": None,
    },
}


def validate_candidate_strategy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != STRATEGY_SCHEMA:
        raise ValueError(f"candidate strategy must use schema {STRATEGY_SCHEMA}")
    if set(value) != {"schema", "data_policy"}:
        raise ValueError("candidate strategy contains unsupported fields")
    policy = value.get("data_policy")
    if not isinstance(policy, Mapping) or set(policy) != {
        "min_input_chars", "deduplicate_inputs", "max_feedback_examples"
    }:
        raise ValueError("data_policy must define exactly the supported selection fields")
    minimum = policy.get("min_input_chars")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= 100_000:
        raise ValueError("min_input_chars must be an integer from 1 to 100000")
    deduplicate = policy.get("deduplicate_inputs")
    if not isinstance(deduplicate, bool):
        raise ValueError("deduplicate_inputs must be a boolean")
    maximum = policy.get("max_feedback_examples")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 100_000
    ):
        raise ValueError("max_feedback_examples must be null or an integer from 1 to 100000")
    return {"schema": STRATEGY_SCHEMA, "data_policy": dict(policy)}


def load_candidate_strategy(candidate_dir: str | Path) -> dict[str, Any]:
    path = Path(candidate_dir) / "candidate_strategy.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("candidate bundle must contain a valid candidate_strategy.json") from exc
    return validate_candidate_strategy(raw)
