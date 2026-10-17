from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from local_experiments.aladdin.hh_rlhf_feedback import DATASET_ID, DATASET_REVISION


SMOKE_VERSION = "aladdin-hh-rlhf-llm-smoke-v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLMSmokeError(RuntimeError):
    pass


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LLMSmokeError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("chosen"), str) or not isinstance(row.get("rejected"), str):
                raise LLMSmokeError(f"{path}:{line_number} must contain string chosen/rejected fields")
            yield row


def _response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMSmokeError("LLM response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise LLMSmokeError("LLM response has empty message content")
    return content.strip()


def _parse_choice(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        choice = value.get("choice")
    else:
        choice = text.strip().strip(".").lower()
    if not isinstance(choice, str):
        raise LLMSmokeError(f"LLM output has no choice field: {content[:300]!r}")
    choice = choice.strip().upper()
    if choice not in {"A", "B"}:
        raise LLMSmokeError(f"LLM choice must be A or B, got {choice!r}")
    return choice


def _build_prompt(history: str, candidate_a: str, candidate_b: str) -> tuple[str, str]:
    system = (
        "You are a pairwise preference evaluator. Compare two assistant answers to the "
        "same user conversation. Choose the answer that is more helpful, correct, and "
        "appropriate. Return only a JSON object with exactly one field: {\"choice\": \"A\"} "
        "or {\"choice\": \"B\"}. Do not explain your choice."
    )
    user = (
        "Conversation history:\n"
        f"{history}\n\n"
        "Candidate A:\n"
        f"{candidate_a}\n\n"
        "Candidate B:\n"
        f"{candidate_b}\n"
    )
    return system, user


def _call_llm(
    *,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("ALADDIN_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMSmokeError(
            f"LLM endpoint returned HTTP {exc.code}: {detail[:1000]}"
        ) from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise LLMSmokeError("LLM endpoint request failed") from exc
    if not isinstance(payload, dict):
        raise LLMSmokeError("LLM endpoint response must be a JSON object")
    content = _response_text(payload)
    usage = payload.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    return _parse_choice(content), {
        "content": content,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def run_smoke(
    data_path: Path,
    *,
    output_path: Path,
    endpoint: str,
    model: str,
    model_revision: str,
    limit: int = 8,
    seed: int = 17,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    randomizer = random.Random(seed)
    results: list[dict[str, Any]] = []
    total_tokens = 0
    started_at = time.monotonic()
    for source_line, row in enumerate(_iter_jsonl(data_path), start=1):
        if len(results) >= limit:
            break
        chosen = str(row["chosen"])
        rejected = str(row["rejected"])
        common_prefix = os.path.commonprefix([chosen, rejected])
        marker = "\n\nAssistant:"
        boundary = common_prefix.rfind(marker)
        if boundary < 0:
            raise LLMSmokeError(f"line {source_line} has no shared assistant boundary")
        history = common_prefix[:boundary].strip()
        chosen_response = chosen[boundary + len(marker):].strip()
        rejected_response = rejected[boundary + len(marker):].strip()
        candidates = [("chosen", chosen_response), ("rejected", rejected_response)]
        randomizer.shuffle(candidates)
        label_a, candidate_a = candidates[0]
        label_b, candidate_b = candidates[1]
        system_prompt, user_prompt = _build_prompt(history, candidate_a, candidate_b)
        choice, usage = _call_llm(
            url=endpoint,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
        )
        predicted_label = label_a if choice == "A" else label_b
        total_tokens += usage["total_tokens"]
        result = {
            "source_line": source_line,
            "prompt_hash": _sha256(history),
            "candidate_order": [label_a, label_b],
            "predicted_label": predicted_label,
            "correct": predicted_label == "chosen",
            "usage": usage,
            "system_prompt_hash": _sha256(system_prompt),
            "user_prompt_hash": _sha256(user_prompt),
        }
        results.append(result)
        elapsed = time.monotonic() - started_at
        print(
            f"[{len(results)}/{limit}] llm-smoke model={model} "
            f"predicted={predicted_label} correct={result['correct']} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    if not results:
        raise LLMSmokeError("no smoke-test rows were processed")
    return {
        "experiment": "aladdin-hh-rlhf-real-llm-smoke",
        "status": "completed",
        "smoke_version": SMOKE_VERSION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_path": str(data_path),
        "final_test": "not loaded",
        "model": {
            "id": model,
            "revision": model_revision,
            "endpoint": endpoint,
            "temperature": 0.0,
            "max_tokens": 32,
        },
        "configuration": {"limit": limit, "seed": seed, "timeout_seconds": timeout_seconds},
        "budget": {
            "llm_calls": len(results),
            "llm_tokens": total_tokens,
        },
        "metrics": {
            "processed": len(results),
            "pairwise_accuracy": sum(bool(item["correct"]) for item in results) / len(results),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Aladdin real-LLM HH-RLHF smoke test.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("ALADDIN_LLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("ALADDIN_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-revision", default=os.environ.get("ALADDIN_LLM_MODEL_REVISION", "unset"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    report = run_smoke(
        args.data,
        output_path=args.output,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=args.model_revision,
        limit=args.limit,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"], "budget": report["budget"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
