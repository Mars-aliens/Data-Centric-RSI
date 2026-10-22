from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from technologies.technology_one.long_horizon_task import (  # noqa: E402
    LongHorizonTaskConfig,
    prepare_helpsteer2_long_horizon_task,
    write_agent_preflight_bundle,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leak-audited HelpSteer2 long-horizon task preflight bundle."
    )
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stream-seed", type=int, default=17)
    parser.add_argument("--groups-per-batch", type=int, default=50)
    parser.add_argument("--feedback-delay", choices=(0, 2, 5), type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=_positive_int,
        help="schedule only the first N deterministic development batches (smoke runs)",
    )
    args = parser.parse_args()

    task = prepare_helpsteer2_long_horizon_task(
        _read_jsonl(args.train_jsonl),
        _read_jsonl(args.validation_jsonl),
        config=LongHorizonTaskConfig(
            stream_seed=args.stream_seed,
            groups_per_batch=args.groups_per_batch,
            feedback_delay_batches=args.feedback_delay,
            max_batches=args.max_batches,
        ),
    )
    manifest = write_agent_preflight_bundle(task, args.output_dir)
    print(f"manifest: {args.output_dir / 'agent-visible' / 'task_manifest.json'}")
    print(f"manifest_sha256: {manifest['manifest_sha256']}")
    print(f"split_group_counts: {manifest['split_group_counts']}")
    print(f"split_record_counts: {manifest['split_record_counts']}")
    print(f"development_batches: {manifest['development_batch_count']}")
    print(f"excluded_singletons: {manifest['excluded_singleton_group_counts']}")
    print("agent_bundle_contains_gold_feedback: false")


if __name__ == "__main__":
    main()
