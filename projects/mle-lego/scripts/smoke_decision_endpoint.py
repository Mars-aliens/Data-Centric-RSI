from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from technologies.technology_one.candidate_proposal import (  # noqa: E402
    validate_candidate_proposal,
)
from technologies.technology_one.policy_command import call_endpoint  # noqa: E402


def _read_profile(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"profile is missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("decision_model"), dict):
        raise ValueError("profile must contain a decision_model object")
    return value


def smoke_decision_endpoint(profile: dict[str, Any]) -> dict[str, Any]:
    configured = profile.get("decision_model")
    if not isinstance(configured, dict):
        raise ValueError("profile must contain a decision_model object")
    model_id = configured.get("model_id") or configured.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("decision_model must contain the API model ID")
    model_fields = (
        "provider",
        "revision",
        "api_url",
        "api_url_env",
        "api_key_env",
        "temperature",
        "top_p",
        "max_output_tokens",
        "json_mode",
    )
    model = {key: configured[key] for key in model_fields if key in configured}
    model["id"] = model_id
    response = call_endpoint(
        {
            "protocol": "lego-data-system-candidate-v1",
            "model": model,
            "prompt": {
                "system": (
                    "Return JSON only with a proposal object, a non-empty rationale, and one "
                    "write edit for smoke.py. The content must be exactly VALUE = 1."
                ),
                "user": "Return the minimal candidate proposal for an API connectivity smoke test.",
            },
        }
    )
    proposal = validate_candidate_proposal(
        response.get("proposal"),
        candidate_id="decision-endpoint-smoke",
        base_version_id="smoke-base",
    )
    if proposal["edits"] != [
        {
            "path": "smoke.py",
            "operation": "write",
            "content": "VALUE = 1",
        }
    ]:
        raise ValueError("Decision LLM did not return the requested smoke edit")
    cost = response.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("Decision LLM response did not include token/call cost")
    return {
        "status": "ok",
        "provider": model.get("provider", "openai_compatible"),
        "model_id": model_id,
        "schema_valid": True,
        "validated_edit_count": len(proposal["edits"]),
        "cost": cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a configured Decision LLM endpoint and candidate JSON schema."
    )
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = smoke_decision_endpoint(_read_profile(args.profile))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
