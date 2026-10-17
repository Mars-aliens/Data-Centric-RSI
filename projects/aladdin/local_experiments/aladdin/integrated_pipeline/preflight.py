from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from .adapters import load_calibrator, load_requirement_state, load_task_records
from .system import clean_and_deduplicate


def _models_url(endpoint: str) -> str:
    marker = "/v1/chat/completions"
    if marker in endpoint:
        return endpoint.split(marker, 1)[0] + "/v1/models"
    return endpoint.rstrip("/") + "/v1/models"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate configured MLE-Aladdin inputs without running training."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--feedback-mode",
        choices=("interactive", "file", "offline-replay"),
        default="interactive",
    )
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    records = load_task_records(args.data)
    splits = {item.split for item in records}
    allowed_splits = {"seed", "development"}
    if args.feedback_mode == "offline-replay":
        allowed_splits.add("held_in")
    if splits - allowed_splits:
        raise ValueError(f"preflight input contains disallowed splits: {sorted(splits)}")
    if args.feedback_mode == "offline-replay":
        candidate_pool = [item for item in records if item.split == "held_in"]
        if not candidate_pool:
            raise ValueError("offline-replay preflight requires held_in records")
        unlabeled = [item.record_id for item in candidate_pool if item.target is None]
        if unlabeled:
            raise ValueError(
                "offline-replay candidate pool has missing human labels: "
                f"{unlabeled[:3]}"
            )
    clean, duplicates = clean_and_deduplicate(records)
    state = load_requirement_state(args.requirements)
    calibration = load_calibrator(args.calibration) if args.calibration else None
    endpoint_models = None
    if args.endpoint:
        try:
            with urllib.request.urlopen(
                _models_url(args.endpoint), timeout=args.timeout_seconds
            ) as handle:
                endpoint_models = json.loads(handle.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"endpoint preflight failed: {exc}") from exc
    report = {
        "status": "ready",
        "feedback_mode": args.feedback_mode,
        "record_count": len(records),
        "clean_record_count": len(clean),
        "exact_duplicates": duplicates,
        "split_counts": {
            split: sum(item.split == split for item in records) for split in sorted(splits)
        },
        "requirement_hypotheses": len(state.hypotheses),
        "calibration": calibration.summary() if calibration else None,
        "endpoint_models": endpoint_models,
        "final_test_accessed": False,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
