from __future__ import annotations

"""JSON stdin/stdout policy client for a vLLM OpenAI-compatible endpoint.

The endpoint is intentionally external: model weights and credentials stay on
the server, while this process remains a small auditable protocol adapter.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if not isinstance(content, str):
        raise ValueError("LLM response content must be a JSON string")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("LLM response must decode to a JSON object")
    return value


def _proposal_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    proposal = result.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    if "action_type" in result:
        return result
    # Accept the common equivalent wrapper emitted by instruction-tuned models.
    action = result.get("action")
    if isinstance(action, dict) and "action_type" in action:
        normalized = dict(result)
        normalized["proposal"] = action
        return normalized
    return None


def call_endpoint(request: dict[str, Any]) -> dict[str, Any]:
    url = os.environ.get("LEGO_POLICY_API_URL")
    if not url:
        raise RuntimeError("LEGO_POLICY_API_URL is required")
    messages = [
        {"role": "system", "content": request["prompt"]["system"]},
        {"role": "user", "content": request["prompt"]["user"]},
    ]
    body = {
        "model": request["model"]["id"],
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(os.environ.get("LEGO_POLICY_MAX_OUTPUT_TOKENS", "512")),
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LEGO_POLICY_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    total_tokens = 0
    last_error = "unknown policy JSON error"
    for attempt in range(2):
        body["messages"] = messages
        encoded = json.dumps(body, ensure_ascii=True).encode("utf-8")
        request_obj = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                request_obj,
                timeout=float(os.environ.get("LEGO_POLICY_TIMEOUT_SECONDS", "300")),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError("Decision LLM endpoint request failed") from exc
        usage = payload.get("usage", {})
        if isinstance(usage, dict):
            total_tokens += int(usage.get("total_tokens", 0))
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Decision LLM response has no choices")
        message = choices[0].get("message", {})
        try:
            result = _json_content(message.get("content"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            result = {}
            last_error = f"invalid JSON ({type(exc).__name__})"
        normalized = _proposal_from_result(result)
        if normalized is not None:
            if "proposal" not in result:
                result = normalized
            result["cost"] = {
                "llm_calls": attempt + 1,
                "llm_tokens": total_tokens,
            }
            if "reflection" in result:
                result["reflection_token_count"] = int(
                    result.get("reflection_token_count", 0)
                )
            return result
        last_error = f"keys={sorted(result)}"
        if attempt == 0:
            if request.get("protocol") == "lego-harness-edit-v1":
                correction = (
                    "Schema correction: return JSON with proposal.base_harness_version, "
                    "proposal.edits (component, patch, rationale), and proposal.rationale."
                )
            else:
                correction = (
                    "Schema correction: your previous response was missing the required "
                    "top-level proposal object. Return JSON with proposal.action_type, "
                    "proposal.arguments, proposal.expected_versions, proposal.rationale, "
                    "and proposal.estimated_cost. You may also include a bounded reflection."
                )
            messages = list(messages) + [{
                "role": "user",
                "content": correction,
            }]
    raise ValueError(f"Decision LLM JSON must contain a proposal object; {last_error}")


def main() -> None:
    request = json.load(sys.stdin)
    if not isinstance(request, dict) or request.get("protocol") not in {
        "lego-policy-v1",
        "lego-harness-edit-v1",
    }:
        raise ValueError("Expected a supported LEGO policy protocol request")
    json.dump(call_endpoint(request), sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
