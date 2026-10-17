from __future__ import annotations

"""Validation and reporting helpers for the real Harness evolution runs."""

import json
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59)


def latest_by_seed(root: Path) -> dict[int, Path]:
    latest: dict[int, Path] = {}
    paths = list(root.glob("*/summary.json"))
    if (root / "summary.json").exists():
        paths.append(root / "summary.json")
    for path in paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            seed = int(summary["seed"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if seed not in latest or path.stat().st_mtime > latest[seed].stat().st_mtime:
            latest[seed] = path
    return latest


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def audit_evolution(
    root: Path,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
    expected_batches: int = 3,
) -> dict[str, Any]:
    summaries = latest_by_seed(root)
    expected = set(expected_seeds)
    errors: list[str] = []
    warnings: list[str] = []
    fingerprints: dict[str, set[str]] = {
        "dataset": set(),
        "task_model": set(),
        "evaluator": set(),
    }

    if set(summaries) != expected:
        errors.append(
            f"seed mismatch: missing={sorted(expected - set(summaries))}, "
            f"unexpected={sorted(set(summaries) - expected)}"
        )

    for seed, path in sorted(summaries.items()):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"seed {seed}: cannot read summary: {exc}")
            continue

        if summary.get("experiment") != "technology-one-server-harness-evolution":
            errors.append(f"seed {seed}: unexpected experiment type")
        if summary.get("decision_llm_used") is not True:
            errors.append(f"seed {seed}: decision_llm_used is not true")
        if summary.get("final_test_used") is not False:
            errors.append(f"seed {seed}: final_test_used is not false")
        if summary.get("held_out_used_for_acceptance") is not False:
            errors.append(f"seed {seed}: held-out data is not report-only")

        batch_count = summary.get("batch_count")
        if batch_count != expected_batches:
            errors.append(
                f"seed {seed}: batch_count={batch_count}, expected {expected_batches}"
            )
        batches = summary.get("batches", [])
        transitions = summary.get("transitions", [])
        if not isinstance(batches, list) or len(batches) != expected_batches:
            errors.append(f"seed {seed}: batch records are incomplete")
        if not isinstance(transitions, list) or len(transitions) != max(expected_batches - 1, 0):
            errors.append(f"seed {seed}: transition count is incomplete")

        budget = summary.get("budget", {})
        if int(budget.get("llm_calls", 0)) <= 0:
            errors.append(f"seed {seed}: no decision-LLM calls recorded")
        if int(budget.get("training_steps", 0)) <= 0:
            errors.append(f"seed {seed}: no training steps recorded")
        if int(budget.get("evaluator_calls", 0)) <= 0:
            errors.append(f"seed {seed}: no evaluator calls recorded")

        for field in fingerprints:
            fingerprints[field].add(_canonical(summary.get(field)))

        if isinstance(batches, list):
            for index, batch in enumerate(batches):
                active = batch.get("active_result", {}) if isinstance(batch, dict) else {}
                if not isinstance(active, dict):
                    errors.append(f"seed {seed}: batch {index} has no active result")
                    continue
                for split in ("held_in", "held_out"):
                    if not isinstance(active.get(split), dict):
                        errors.append(f"seed {seed}: batch {index} lacks {split} metrics")

        if isinstance(transitions, list):
            for index, transition in enumerate(transitions):
                if not isinstance(transition, dict):
                    errors.append(f"seed {seed}: transition {index} is not an object")
                    continue
                status = transition.get("status")
                if status not in {"accepted", "rejected"}:
                    errors.append(f"seed {seed}: transition {index} has invalid status")
                if transition.get("held_out_used_for_acceptance") is not False:
                    errors.append(f"seed {seed}: transition {index} used held-out data")
                if status == "rejected" and transition.get("rollback_performed") is not True:
                    errors.append(f"seed {seed}: rejected transition {index} lacks rollback")

        accepted = int(summary.get("accepted_version_count", 0))
        rejected = int(summary.get("rejected_candidate_count", 0))
        if isinstance(transitions, list) and accepted + rejected != len(transitions):
            errors.append(f"seed {seed}: transition counters do not match records")
        if accepted == 0:
            warnings.append(f"seed {seed}: no candidate was accepted; this is valid but should be reported")

    for field, values in fingerprints.items():
        if len(values) > 1:
            errors.append(f"{field} differs across seeds")

    return {
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "run_count": len(summaries),
        "seeds": sorted(summaries),
        "missing_seeds": sorted(expected - set(summaries)),
        "unexpected_seeds": sorted(set(summaries) - expected),
        "expected_batches": expected_batches,
        "warnings": warnings,
        "errors": errors,
        "latest_summary_by_seed": {
            str(seed): str(path) for seed, path in sorted(summaries.items())
        },
    }


def _metric(batch: dict[str, Any], split: str) -> float:
    return float(batch["active_result"][split]["macro_f1"])


def summarize_evolution(root: Path, expected_seeds: tuple[int, ...] = EXPECTED_SEEDS) -> dict[str, Any]:
    paths = latest_by_seed(root)
    rows: list[dict[str, Any]] = []
    for seed in sorted(expected_seeds):
        path = paths.get(seed)
        if path is None:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        batches = summary.get("batches", [])
        if not batches:
            continue
        first = batches[0]
        last = batches[-1]
        first_in = _metric(first, "held_in")
        first_out = _metric(first, "held_out")
        final_in = _metric(last, "held_in")
        final_out = _metric(last, "held_out")
        transitions = summary.get("transitions", [])
        budget = summary.get("budget", {})
        rows.append(
            {
                "seed": seed,
                "initial_held_in": first_in,
                "initial_held_out": first_out,
                "final_active_held_in": final_in,
                "final_active_held_out": final_out,
                "delta_held_in": final_in - first_in,
                "delta_held_out": final_out - first_out,
                "accepted": int(summary.get("accepted_version_count", 0)),
                "rejected": int(summary.get("rejected_candidate_count", 0)),
                "rollbacks": int(summary.get("rollback_count", 0)),
                "llm_calls": int(budget.get("llm_calls", 0)),
                "llm_tokens": int(budget.get("llm_tokens", 0)),
                "training_steps": int(budget.get("training_steps", 0)),
                "evaluator_calls": int(budget.get("evaluator_calls", 0)),
                "wall_clock_seconds": float(summary.get("timing", {}).get("wall_clock_seconds", 0.0)),
                "summary": str(path),
                "transition_count": len(transitions),
            }
        )

    def aggregate(field: str) -> dict[str, float | int]:
        values = [float(row[field]) for row in rows]
        if not values:
            return {"n": 0, "mean": 0.0, "sd": 0.0}
        return {
            "n": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "root": str(root),
        "completed_rows": len(rows),
        "rows": rows,
        "aggregate": {
            field: aggregate(field)
            for field in (
                "initial_held_out",
                "final_active_held_out",
                "delta_held_out",
                "accepted",
                "rejected",
                "rollbacks",
                "llm_calls",
                "llm_tokens",
                "wall_clock_seconds",
            )
        },
    }
