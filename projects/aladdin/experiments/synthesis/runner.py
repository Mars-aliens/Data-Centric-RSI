"""Command-line runner for the static synthesis baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generator import OpenAICompatibleGenerator
from .strategies import STRATEGIES, synthesize


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = _read_jsonl(input_path)
    if args.strategy == "evol-instruct" and not args.evol_task_compatible:
        raise ValueError(
            "Evol-Instruct is an instruction-evolution adaptation; pass "
            "--evol-task-compatible only after confirming task compatibility"
        )
    generator = None
    if args.strategy != "seed-only":
        if not args.endpoint or not args.model:
            raise ValueError("--endpoint and --model are required for generated strategies")
        generator = OpenAICompatibleGenerator(
            args.endpoint,
            args.model,
            api_key=os.environ.get(args.api_key_env),
            timeout_seconds=args.timeout_seconds,
        )

    rows = synthesize(
        seeds,
        strategy=args.strategy,
        count=args.count,
        generator=generator,
        seed_value=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    usage = getattr(generator, "usage", None)
    manifest = {
        "status": "completed",
        "protocol": "aladdin-synthesis-experiment-1a-v1",
        "strategy": args.strategy,
        "seed": args.seed,
        "input_path_name": input_path.name,
        "input_sha256": _sha256(input_path),
        "seed_count": len(seeds),
        "candidate_count": len(rows),
        "generation_calls": sum(int(row["generation_calls"]) for row in rows),
        "generation_usage": {
            "requests": getattr(usage, "requests", 0),
            "reported_by_endpoint": getattr(usage, "reported", False),
            "reported_requests": getattr(usage, "reported_requests", 0),
            "complete_usage_requests": getattr(usage, "complete_usage_requests", 0),
            "prompt_tokens": getattr(usage, "prompt_tokens", None)
            if getattr(usage, "reported", False)
            else None,
            "completion_tokens": getattr(usage, "completion_tokens", None)
            if getattr(usage, "reported", False)
            else None,
        },
        "model": args.model if generator else None,
        "endpoint": args.endpoint if generator else None,
        "max_tokens_per_call": args.max_tokens,
        "temperature": args.temperature,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path_name": output_path.name,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="seed JSONL path")
    parser.add_argument("--output", required=True, help="output candidate JSONL path")
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--count", type=int, default=32, help="number of generated candidates")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--endpoint", help="OpenAI-compatible chat completions URL")
    parser.add_argument("--model", help="served model identifier")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--evol-task-compatible",
        action="store_true",
        help="confirm the task is suitable for instruction/task-input evolution",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_experiment(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
