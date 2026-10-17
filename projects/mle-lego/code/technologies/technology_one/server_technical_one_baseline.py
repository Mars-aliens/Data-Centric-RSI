from __future__ import annotations

"""Run the frozen technology-one fixed baseline with the server Task Model.

This is the control arm for the natural-support technology-one experiment.  It
uses the same data pool, group sampler, selector format, train-group budget,
trainer command, evaluator command, and seed as the candidate matrix, but it
does not construct or select any edited Harness candidate.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from core_system.mle_core.models import TaskRecord

from .data_validation import validate_records
from .harness import HarnessManifest
from .protocol import JsonlTaskAdapter, load_protocol
from .progress import ProgressReporter
from .server_preflight import run_preflight
from .server_runner import _required_string
from .server_technical_one import _run_candidate
from .technical_one_local import (
    SPLITS,
    _eligible_groups,
    _groups,
    _limit_groups,
    _select_groups,
    _train_records,
    initial_manifest,
)


def _group_hash(group_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(group_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _response_word_count(record: TaskRecord) -> int:
    return len(str(record.input).split("Response:", 1)[-1].split())


def _audit_training_records(
    raw_records: list[TaskRecord],
    formatted_records: list[TaskRecord],
    group_ids: list[str],
    *,
    manifest: HarnessManifest,
) -> dict[str, Any]:
    if len(raw_records) != len(formatted_records):
        raise ValueError("Baseline formatting changed the number of training records")

    raw_by_id = {record.record_id: record for record in raw_records}
    formatted_by_id = {record.record_id: record for record in formatted_records}
    truncation = {
        record_id: _response_word_count(raw_by_id[record_id])
        > _response_word_count(formatted_by_id[record_id])
        for record_id in formatted_by_id
    }
    margins = {
        str(record.source_group_id): float(record.metadata.get("score_mean", 0.0))
        for record in raw_records
    }
    group_records: dict[str, list[TaskRecord]] = {}
    for record in raw_records:
        if record.source_group_id is None:
            raise ValueError(f"Training record has no source group: {record.record_id}")
        group_records.setdefault(record.source_group_id, []).append(record)

    group_margins: dict[str, float] = {}
    for group_id, records in group_records.items():
        scores = [float(record.metadata.get("score_mean", 0.0)) for record in records]
        group_margins[group_id] = max(scores) - min(scores) if scores else 0.0

    return {
        "manifest_version": manifest.version,
        "train_group_count": len(group_ids),
        "train_group_hash": _group_hash(group_ids),
        "train_record_count": len(formatted_records),
        "label_counts": dict(sorted(Counter(str(record.target) for record in formatted_records).items())),
        "bucket_counts": dict(
            sorted(
                Counter(
                    int(
                        hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()[:8],
                        16,
                    )
                    % 10
                    for group_id in group_ids
                ).items()
            )
        ),
        "margin_summary": {
            "min": min(group_margins.values()) if group_margins else 0.0,
            "max": max(group_margins.values()) if group_margins else 0.0,
            "mean": (
                sum(group_margins.values()) / len(group_margins)
                if group_margins
                else 0.0
            ),
            "histogram": dict(
                sorted(
                    Counter(
                        f"{value:.1f}" for value in group_margins.values()
                    ).items()
                )
            ),
        },
        "selector": {
            "max_response_words": int(
                manifest.components["selector"].config["max_response_words"]
            ),
            "truncated_record_count": sum(truncation.values()),
            "truncated_record_fraction": (
                sum(truncation.values()) / len(truncation) if truncation else 0.0
            ),
        },
        "filter": dict(manifest.components["filter"].config),
        "generator": dict(manifest.components["generator"].config),
    }


def run_server_technology_one_baseline(
    profile_path: Path,
    *,
    root: Path,
    output_dir: Path,
    run_id: str,
    seed: int | None = None,
    max_train_groups: int = 3000,
    max_eval_groups: int = 1000,
    train_group_budget: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preflight = run_preflight(profile_path, root)
    if preflight["errors"]:
        raise ValueError("Server preflight failed: " + "; ".join(preflight["errors"]))

    protocol = load_protocol(root / profile["dataset"]["protocol"])
    records = JsonlTaskAdapter(protocol).load(root / profile["dataset"]["train_jsonl"])
    validate_records(records, protocol)
    run_seed = int(profile["seeds"][0] if seed is None else seed)
    by_split = {
        split: _groups(record for record in records if record.split == split)
        for split in SPLITS
    }
    train_groups = _limit_groups(by_split["seed"], max_train_groups)
    held_in_groups = _limit_groups(by_split["held_in"], max_eval_groups)
    held_out_groups = _limit_groups(by_split["held_out"], max_eval_groups)
    manifest = initial_manifest(
        train_group_budget=train_group_budget
        or int(profile["stream"].get("train_group_budget", 1000))
    )

    requested_budget = min(
        max_train_groups,
        int(manifest.components["selector"].config["train_group_budget"]),
    )
    eligible_count = len(_eligible_groups(manifest, train_groups))
    if eligible_count < requested_budget:
        raise ValueError(
            "Fixed baseline cannot satisfy the declared train-group budget: "
            f"eligible={eligible_count}, requested={requested_budget}"
        )

    selected_records, selected_group_ids = _select_groups(
        manifest,
        train_groups,
        requested_budget,
        run_seed,
    )
    formatted_records = _train_records(manifest, selected_records)
    held_in_records = [
        record for group in held_in_groups.values() for record in group
    ]
    held_out_records = [
        record for group in held_out_groups.values() for record in group
    ]
    run_root = output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    estimated_seconds = float(
        os.environ.get(
            "LEGO_ESTIMATED_CANDIDATE_SECONDS",
            profile.get("progress", {}).get("estimated_candidate_seconds", 300.0),
        )
    )
    print(
        "[LEGO] Technology 1 fixed baseline estimated total: "
        f"~{estimated_seconds / 60.0:.1f} minutes",
        file=sys.stderr,
        flush=True,
    )
    progress = ProgressReporter(
        "Technology 1 fixed baseline",
        1,
        unit="baseline",
        expected_item_seconds=estimated_seconds,
        initial_elapsed_seconds=time.perf_counter() - started,
    )
    result = _run_candidate(
        root=run_root,
        label="fixed",
        manifest=manifest,
        train_records=formatted_records,
        train_group_ids=selected_group_ids,
        held_in_records=held_in_records,
        held_out_records=held_out_records,
        profile=profile,
        seed=run_seed,
    )
    progress.report(1)
    progress.finish()

    summary = {
        "experiment": "technology-one-server-fixed-baseline",
        "run_id": run_id,
        "baseline_method": "fixed-harness-react",
        "dataset": profile["dataset"],
        "task_model": profile["task_model"],
        "evaluator": profile["evaluator"],
        "decision_policy": "fixed-natural-support-control",
        "decision_llm_used": False,
        "common_support": False,
        "final_test_used": False,
        "seed": run_seed,
        "max_train_groups": max_train_groups,
        "max_eval_groups": max_eval_groups,
        "requested_train_group_budget": requested_budget,
        "eligible_train_groups": eligible_count,
        "held_in_group_count": len(held_in_groups),
        "held_out_group_count": len(held_out_groups),
        "train_group_audit": _audit_training_records(
            selected_records,
            formatted_records,
            selected_group_ids,
            manifest=manifest,
        ),
        "results": {"fixed": result},
        "held_in": result["held_in"],
        "held_out": result["held_out"],
        "timing": {
            "wall_clock_seconds": time.perf_counter() - started,
            "estimated_candidate_seconds": estimated_seconds,
        },
        "provenance": {
            "code_path": str(Path(__file__)),
            "profile_version": profile["profile_version"],
            "dataset_source": profile["dataset"]["source"],
            "dataset_revision": profile["dataset"]["source_revision"],
            "profile_hash": hashlib.sha256(
                profile_path.read_bytes()
            ).hexdigest(),
            "preflight_hashes": preflight["hashes"],
        },
        "interpretation": (
            "Frozen fixed control for the natural-support technology-one "
            "comparison. It is a server Task Model baseline, not a claim of "
            "full decision-LLM-generated Harness evolution."
        ),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed Qwen Task Model baseline for technology one."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-groups", type=int, default=3000)
    parser.add_argument("--max-eval-groups", type=int, default=1000)
    parser.add_argument("--train-group-budget", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run_server_technology_one_baseline(
                args.profile,
                root=args.root,
                output_dir=args.output_dir,
                run_id=args.run_id,
                seed=args.seed,
                max_train_groups=args.max_train_groups,
                max_eval_groups=args.max_eval_groups,
                train_group_budget=args.train_group_budget,
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
