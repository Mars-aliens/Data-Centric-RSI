from __future__ import annotations

"""Run a matched-budget 2x2 technology-one component ablation.

The generator is allowed to change group eligibility. Within each generator
condition, the selector-only comparison uses the same selected groups, so the
selector effect is not mixed with a second sampling change.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from .data_validation import validate_records
from .progress import ProgressReporter
from .protocol import JsonlTaskAdapter, load_protocol
from .server_adapters import CommandEvaluator, CommandTaskModelTrainer
from .server_runner import _command, _required_string
from .server_technical_one import _run_candidate
from .server_preflight import run_preflight
from .technical_one_local import (
    SPLITS,
    _eligible_groups,
    _groups,
    _limit_groups,
    _make_candidates,
    _train_records,
    initial_manifest,
)
from .harness import apply_edits


METHODS = (
    "fixed",
    "generator-only",
    "selector-only",
    "generator-selector",
)


def _group_hash(group_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(group_ids)).encode("utf-8")).hexdigest()


def run_paired_ablation(
    profile_path: Path,
    *,
    root: Path,
    output_dir: Path,
    run_id: str,
    seed: int,
    max_train_groups: int,
    max_eval_groups: int,
    train_group_budget: int | None,
) -> dict[str, Any]:
    experiment_started = time.perf_counter()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preflight = run_preflight(profile_path, root)
    if preflight["errors"]:
        raise ValueError("Server preflight failed: " + "; ".join(preflight["errors"]))
    protocol = load_protocol(root / profile["dataset"]["protocol"])
    records = JsonlTaskAdapter(protocol).load(root / profile["dataset"]["train_jsonl"])
    validate_records(records, protocol)
    by_split = {
        split: _groups(record for record in records if record.split == split)
        for split in SPLITS
    }
    train_groups = _limit_groups(by_split["seed"], max_train_groups)
    held_in_groups = _limit_groups(by_split["held_in"], max_eval_groups)
    held_out_groups = _limit_groups(by_split["held_out"], max_eval_groups)
    current = initial_manifest(
        train_group_budget=train_group_budget
        or int(profile["stream"].get("train_group_budget", 1000))
    )
    edits, suspected, diagnosis = _make_candidates(
        current, train_groups, held_in_groups
    )
    manifests = {
        "fixed": current,
        "generator-only": apply_edits(current, (edits["generator"],)),
        "selector-only": apply_edits(current, (edits["selector"],)),
        "generator-selector": apply_edits(
            current, (edits["generator"], edits["selector"])
        ),
    }
    eligible = {
        name: _eligible_groups(manifest, train_groups)
        for name, manifest in manifests.items()
    }
    requested_budget = min(
        max_train_groups,
        int(current.components["selector"].config["train_group_budget"]),
    )
    insufficient = {
        name: len(groups)
        for name, groups in eligible.items()
        if len(groups) < requested_budget
    }
    if insufficient:
        raise ValueError(
            "Matched-budget ablation cannot satisfy the shared budget: "
            f"eligible_groups={insufficient}, requested_budget={requested_budget}. "
            "Increase --max-train-groups or reduce --train-group-budget."
        )
    # Use the same deterministic sampling rule for every arm. Because fixed
    # and selector-only have identical eligibility, and generator-only and
    # generator-selector do too, each selector comparison is group-matched.
    selected_group_ids = {
        name: random.Random(seed).sample(sorted(groups), requested_budget)
        for name, groups in eligible.items()
    }
    if selected_group_ids["fixed"] != selected_group_ids["selector-only"]:
        raise AssertionError("selector control arms did not receive matched groups")
    if selected_group_ids["generator-only"] != selected_group_ids["generator-selector"]:
        raise AssertionError("generator control arms did not receive matched groups")
    held_in_records = [record for group in held_in_groups.values() for record in group]
    held_out_records = [record for group in held_out_groups.values() for record in group]
    run_root = output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    estimated_candidate_seconds = float(
        os.environ.get(
            "LEGO_ESTIMATED_CANDIDATE_SECONDS",
            profile.get("progress", {}).get("estimated_candidate_seconds", 180.0),
        )
    )
    setup_seconds = time.perf_counter() - experiment_started
    estimated_total_seconds = setup_seconds + estimated_candidate_seconds * len(METHODS)
    print(
        "[LEGO] Technology 1 paired ablation estimated total: "
        f"~{estimated_total_seconds / 60.0:.1f} minutes "
        f"({estimated_candidate_seconds:.0f}s/candidate; setup {setup_seconds:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    progress = ProgressReporter(
        "Technology 1 paired ablation",
        len(METHODS),
        unit="candidates",
        expected_item_seconds=estimated_candidate_seconds,
        initial_elapsed_seconds=setup_seconds,
    )
    for method in METHODS:
        selected_records = [
            record
            for group_id in selected_group_ids[method]
            for record in train_groups[group_id]
        ]
        results[method] = _run_candidate(
            root=run_root,
            label=method,
            manifest=manifests[method],
            train_records=_train_records(manifests[method], selected_records),
            train_group_ids=selected_group_ids[method],
            held_in_records=held_in_records,
            held_out_records=held_out_records,
            profile=profile,
            seed=seed,
        )
        progress.report(len(results))
    progress.finish()
    wall_clock_seconds = time.perf_counter() - experiment_started
    active_method = max(
        METHODS,
        key=lambda method: (
            float(results[method]["held_in"].get("macro_f1", 0.0)),
            method == "fixed",
        ),
    )
    summary = {
        "experiment": "technology-one-server-paired-ablation",
        "run_id": run_id,
        "dataset": profile["dataset"],
        "task_model": profile["task_model"],
        "evaluator": profile["evaluator"],
        "seed": seed,
        "methods": list(METHODS),
        "diagnosis": diagnosis,
        "suspected_components": list(suspected),
        "pairing": {
            "same_train_group_count": True,
            "train_group_count": requested_budget,
            "same_group_ids_within_selector_control": True,
            "selector_control_group_hashes": {
                "fixed": _group_hash(selected_group_ids["fixed"]),
                "selector-only": _group_hash(selected_group_ids["selector-only"]),
                "generator-only": _group_hash(selected_group_ids["generator-only"]),
                "generator-selector": _group_hash(selected_group_ids["generator-selector"]),
            },
            "selected_group_hash_by_method": {
                name: _group_hash(group_ids)
                for name, group_ids in selected_group_ids.items()
            },
            "selected_group_overlap": {
                "fixed_vs_generator_only": len(
                    set(selected_group_ids["fixed"])
                    & set(selected_group_ids["generator-only"])
                ),
            },
            "eligible_group_count_by_method": {
                name: len(value) for name, value in eligible.items()
            },
        },
        "selection": {
            "metric": "held_in_macro_f1",
            "active_method_for_diagnostic_only": active_method,
        },
        "timing": {
            "estimated_candidate_seconds_at_start": estimated_candidate_seconds,
            "estimated_setup_seconds_at_start": setup_seconds,
            "estimated_total_seconds_at_start": estimated_total_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            "wall_clock_minutes": wall_clock_seconds / 60.0,
            "actual_average_candidate_seconds": (
                (wall_clock_seconds - setup_seconds) / len(METHODS)
                if METHODS
                else 0.0
            ),
        },
        "results": results,
        "final_test_used": False,
        "interpretation": (
            "Matched-budget 2x2 ablation: generator changes group eligibility, "
            "while selector comparisons are group-matched within each generator "
            "condition."
        ),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired LEGO technology-one ablation.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-groups", type=int, default=3000)
    parser.add_argument("--max-eval-groups", type=int, default=1000)
    parser.add_argument("--train-group-budget", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run_paired_ablation(
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

