from __future__ import annotations

"""Audit completed Technology One summaries before baseline comparison.

This is a protocol check, not a statistical significance test. It catches
missing seeds, mixed method matrices, accidental final-test use, and basic
model/budget drift before results are aggregated.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59)
EXPECTED_METHODS = (
    "fixed",
    "independent-edit",
    "selector-only-edit",
    "filter-only-edit",
    "dependency-aware-joint-edit",
    "generator-filter-joint-edit",
    "selector-filter-joint-edit",
    "generator-selector-filter-joint-edit",
)


def _latest_by_seed(root: Path) -> dict[int, Path]:
    latest: dict[int, Path] = {}
    for path in root.glob("*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        seed = int(summary["seed"])
        if seed not in latest or path.stat().st_mtime > latest[seed].stat().st_mtime:
            latest[seed] = path
    return latest


def _method_names(summary: dict[str, Any]) -> tuple[str, ...]:
    return tuple(summary.get("results", {}).keys())


def audit(root: Path, expected_seeds: tuple[int, ...] = EXPECTED_SEEDS) -> dict[str, Any]:
    latest = _latest_by_seed(root)
    expected_set = set(expected_seeds)
    actual_set = set(latest)
    errors: list[str] = []
    warnings: list[str] = []

    if actual_set != expected_set:
        errors.append(
            f"seed mismatch: missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}"
        )

    method_sets = Counter(_method_names(json.loads(path.read_text(encoding="utf-8"))) for path in latest.values())
    if len(method_sets) != 1:
        errors.append(f"method matrix differs across runs: {method_sets}")
    elif next(iter(method_sets)) != EXPECTED_METHODS:
        errors.append(
            "method matrix does not match the frozen primary matrix: "
            f"{next(iter(method_sets))}"
        )

    model_revisions: set[str] = set()
    evaluator_revisions: set[str] = set()
    budgets: set[int] = set()
    common_support_values: set[bool] = set()
    final_test_values: set[bool] = set()

    for seed in sorted(latest):
        summary = json.loads(latest[seed].read_text(encoding="utf-8"))
        model = summary.get("task_model", {})
        evaluator = summary.get("evaluator", {})
        model_revisions.add(json.dumps(model, sort_keys=True))
        evaluator_revisions.add(json.dumps(evaluator, sort_keys=True))
        budgets.add(int(summary.get("shared_train_group_budget", -1)))
        design = summary.get("comparison_design", {})
        common_support_values.add(bool(design.get("common_support_requested", False)))
        final_test_values.add(bool(summary.get("final_test_used", False)))

        if summary.get("final_test_used", False):
            errors.append(f"seed {seed}: final_test_used=true")
        if bool(design.get("common_support_requested", False)):
            warnings.append(f"seed {seed}: common-support run is not primary natural-support data")

    if len(model_revisions) > 1:
        errors.append("task_model configuration differs across seeds")
    if len(evaluator_revisions) > 1:
        errors.append("evaluator configuration differs across seeds")
    if len(budgets) > 1:
        errors.append(f"shared training budget differs across seeds: {sorted(budgets)}")
    if final_test_values != {False}:
        errors.append(f"final_test flags are not uniformly false: {final_test_values}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "run_count": len(latest),
        "seeds": sorted(latest),
        "missing_seeds": sorted(expected_set - actual_set),
        "unexpected_seeds": sorted(actual_set - expected_set),
        "method_matrix": list(next(iter(method_sets)) if method_sets else ()),
        "shared_budgets": sorted(budgets),
        "common_support_values": sorted(common_support_values),
        "final_test_values": sorted(final_test_values),
        "warnings": warnings,
        "errors": errors,
        "latest_summary_by_seed": {
            str(seed): str(path) for seed, path in sorted(latest.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, tuple(args.seeds))
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(f"status: {report['status']}")
        print(f"run_count: {report['run_count']}")
        print(f"seeds: {report['seeds']}")
        print(f"missing: {report['missing_seeds']}")
        print(f"unexpected: {report['unexpected_seeds']}")
        print(f"budgets: {report['shared_budgets']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
