from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from technologies.technology_one.candidate_proposal import (  # noqa: E402
    CandidateProposalError,
    build_candidate_prompt,
    validate_candidate_proposal,
    write_candidate_bundle,
)
from technologies.technology_one.policy_command import call_endpoint  # noqa: E402


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decision_model(profile: dict[str, Any]) -> dict[str, Any]:
    configured = profile.get("decision_model")
    if not isinstance(configured, dict):
        raise ValueError("profile must contain a decision_model object")
    model_id = configured.get("model_id") or configured.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("decision_model must contain the provider's API model_id")
    fields = (
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
    result = {key: configured[key] for key in fields if key in configured}
    result["id"] = model_id
    return result


def propose_candidate(
    *,
    profile: dict[str, Any],
    source_dir: Path,
    output_dir: Path,
    observation: dict[str, Any],
    candidate_id: str,
    base_version_id: str,
    selected_context_files: list[str] | None = None,
) -> dict[str, Any]:
    system, user = build_candidate_prompt(
        base_dir=source_dir,
        observation=observation,
        candidate_id=candidate_id,
        base_version_id=base_version_id,
        selected_context_files=selected_context_files,
    )
    model = _decision_model(profile)
    request = {
        "protocol": "lego-data-system-candidate-v1",
        "model": model,
        "prompt": {"system": system, "user": user},
    }
    prompt_hashes: dict[str, str] = {
        "system_sha256": _sha256(system),
        "user_initial_sha256": _sha256(user),
    }
    total_calls = 0
    total_tokens = 0
    proposal: dict[str, Any] | None = None
    validation_error = "proposal was not validated"

    for attempt in range(2):
        response = call_endpoint(request)
        raw_cost = response.get("cost", {})
        if not isinstance(raw_cost, dict):
            raise ValueError("Decision LLM response cost must be an object")
        calls = raw_cost.get("llm_calls", 0)
        tokens = raw_cost.get("llm_tokens", 0)
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls < 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
        ):
            raise ValueError("Decision LLM call and token costs must be non-negative integers")
        total_calls += calls
        total_tokens += tokens
        raw_proposal = response.get("proposal")
        try:
            proposal = validate_candidate_proposal(
                raw_proposal,
                candidate_id=candidate_id,
                base_version_id=base_version_id,
            )
            break
        except CandidateProposalError as exc:
            validation_error = str(exc)
            if attempt:
                raise
            request = {
                **request,
                "prompt": {
                    "system": system,
                    "user": (
                        user
                        + "\n\nSchema correction: the previous proposal failed local "
                        + f"validation ({validation_error[:500]}). Return a corrected JSON "
                        "proposal that obeys every listed constraint."
                    ),
                },
            }
            prompt_hashes["user_correction_sha256"] = _sha256(request["prompt"]["user"])

    if proposal is None:
        raise CandidateProposalError(validation_error)

    cost = {"llm_calls": total_calls, "llm_tokens": total_tokens}
    manifest = write_candidate_bundle(
        base_dir=source_dir,
        output_dir=output_dir,
        proposal=proposal,
        candidate_id=candidate_id,
        base_version_id=base_version_id,
        decision_model={
            "provider": model.get("provider", "openai_compatible"),
            "model_id": model["id"],
            "revision": model.get("revision"),
        },
        prompt_hashes=prompt_hashes,
        proposal_cost=cost,
    )
    return {**manifest, "output_dir": str(output_dir.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a configured Decision LLM for one bounded T1 candidate bundle."
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--observation-json", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--base-version-id", default="runtime-stable")
    parser.add_argument("--context-file", action="append", dest="context_files")
    args = parser.parse_args()

    profile = _json_object(args.profile, "profile")
    observation = _json_object(args.observation_json, "observation")
    result = propose_candidate(
        profile=profile,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        observation=observation,
        candidate_id=args.candidate_id,
        base_version_id=args.base_version_id,
        selected_context_files=args.context_files,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
