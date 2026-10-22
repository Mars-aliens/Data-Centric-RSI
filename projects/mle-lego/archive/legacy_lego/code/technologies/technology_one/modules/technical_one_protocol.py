from __future__ import annotations

"""Shared protocol and audit helpers for the complete Technology One study.

This module does not run a model. It defines the required arms and validates
the properties that make their results comparable: pinned seeds, locked final
test, declared proposal modes, and equal per-candidate training/evaluation
budgets.
"""

import json
from pathlib import Path
from typing import Any, Iterable


TECHNOLOGY_ONE_SEEDS = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59)

COMPONENT_MATRIX_METHODS = (
    "fixed",
    "independent-edit",
    "selector-only-edit",
    "filter-only-edit",
    "dependency-aware-joint-edit",
    "generator-filter-joint-edit",
    "selector-filter-joint-edit",
    "generator-selector-filter-joint-edit",
)

DECISION_BASELINE_METHODS = (
    "fixed-harness-react",
    "react-reflexion",
)

EVOLUTION_MODES = ("independent", "dependency-aware-joint")

REQUIRED_TECHNOLOGY_ONE_EVIDENCE = (
    "component_matrix",
    "decision_baselines",
    "independent_evolution",
    "dependency_aware_joint_evolution",
)


def _summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/summary.json")) if root.exists() else []


def summaries_by_seed(root: Path, *, expected_seeds: Iterable[int] = TECHNOLOGY_ONE_SEEDS) -> dict[int, Path]:
    """Find the latest completed summary for every requested seed."""
    expected = tuple(int(seed) for seed in expected_seeds)
    found: dict[int, Path] = {}
    for path in _summary_paths(root):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            seed = int(summary["seed"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if summary.get("status") not in ("complete", "completed", "COMPLETED"):
            continue
        rank = (path.stat().st_mtime_ns, str(path))
        previous_rank = (
            (found[seed].stat().st_mtime_ns, str(found[seed]))
            if seed in found
            else None
        )
        if previous_rank is None or rank > previous_rank:
            found[seed] = path
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "summary seed audit failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return found


def _budget_signature(summary: dict[str, Any]) -> tuple[int, int, int, int]:
    # Compare declared limits, not consumed cost. Rejection rates and proposal
    # retries legitimately make consumed LLM cost differ between arms.
    budget = summary.get("budget_limit") or summary.get("budget", {})
    return (
        int(budget.get("training_steps", 0)),
        int(budget.get("evaluator_calls", 0)),
        int(summary.get("requested_train_group_budget", 0)),
        int(summary.get("batch_count", summary.get("stream", {}).get("max_batches", 0))),
    )


def validate_summary(
    path: Path,
    *,
    expected_seed: int | None = None,
    expected_proposal_mode: str | None = None,
    expected_methods: Iterable[str] | None = None,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if expected_seed is not None and int(summary.get("seed", -1)) != expected_seed:
        raise ValueError(f"{path}: seed does not match expected {expected_seed}")
    if summary.get("final_test_used") is not False:
        raise ValueError(f"{path}: final_test_used must remain false before final freeze")
    if expected_proposal_mode is not None and summary.get("proposal_mode") != expected_proposal_mode:
        raise ValueError(
            f"{path}: proposal_mode={summary.get('proposal_mode')!r}, "
            f"expected {expected_proposal_mode!r}"
        )
    if expected_methods is not None:
        actual = set(summary.get("methods", []))
        missing = sorted(set(expected_methods) - actual)
        if missing:
            raise ValueError(f"{path}: missing required methods {missing}")
    return summary


def audit_complete_technology_one(
    *,
    component_root: Path,
    independent_root: Path,
    dependency_root: Path,
    fixed_root: Path,
    reflexion_root: Path,
    expected_seeds: Iterable[int] = TECHNOLOGY_ONE_SEEDS,
) -> dict[str, Any]:
    """Audit all mandatory Technology One evidence without reading final_test."""
    seeds = tuple(int(seed) for seed in expected_seeds)
    component = summaries_by_seed(component_root, expected_seeds=seeds)
    independent = summaries_by_seed(independent_root, expected_seeds=seeds)
    dependency = summaries_by_seed(dependency_root, expected_seeds=seeds)

    component_summaries = {
        seed: validate_summary(
            path, expected_seed=seed, expected_methods=COMPONENT_MATRIX_METHODS
        )
        for seed, path in component.items()
    }
    independent_summaries = {
        seed: validate_summary(path, expected_seed=seed, expected_proposal_mode="independent")
        for seed, path in independent.items()
    }
    dependency_summaries = {
        seed: validate_summary(
            path, expected_seed=seed, expected_proposal_mode="dependency-aware-joint"
        )
        for seed, path in dependency.items()
    }
    missing_joint_exploration = {
        seed: str(path)
        for seed, path in dependency.items()
        if not any(
            len(transition.get("edits", [])) >= 2
            for transition in dependency_summaries[seed].get("transitions", [])
        )
    }
    if missing_joint_exploration:
        raise ValueError(
            "dependency-aware-joint evidence contains no joint proposal for seeds: "
            f"{missing_joint_exploration}"
        )

    decision_baselines: dict[str, dict[int, Path]] = {}
    for name, root in (("fixed-harness-react", fixed_root), ("react-reflexion", reflexion_root)):
        paths = summaries_by_seed(root, expected_seeds=seeds)
        for seed, path in paths.items():
            summary = validate_summary(path, expected_seed=seed)
            if summary.get("method") != name:
                raise ValueError(f"{path}: method={summary.get('method')!r}, expected {name!r}")
            if summary.get("decision_llm_used") is not True:
                raise ValueError(f"{path}: decision_llm_used must be true")
        decision_baselines[name] = paths

    mode_budget_mismatches: list[dict[str, Any]] = []
    for seed in seeds:
        left = _budget_signature(independent_summaries[seed])
        right = _budget_signature(dependency_summaries[seed])
        if left != right:
            mode_budget_mismatches.append(
                {"seed": seed, "independent": left, "dependency_aware": right}
            )
    if mode_budget_mismatches:
        raise ValueError(f"independent/dependency-aware-joint budget mismatch: {mode_budget_mismatches}")

    return {
        "status": "PASS",
        "seeds": list(seeds),
        "evidence": list(REQUIRED_TECHNOLOGY_ONE_EVIDENCE),
        "component_matrix": {str(seed): str(path) for seed, path in component.items()},
        "independent_evolution": {str(seed): str(path) for seed, path in independent.items()},
        "dependency_aware_joint_evolution": {str(seed): str(path) for seed, path in dependency.items()},
        "decision_baselines": {
            name: {str(seed): str(path) for seed, path in paths.items()}
            for name, paths in decision_baselines.items()
        },
        "budget_comparison": {"independent_vs_dependency_aware_joint": "PASS"},
        "final_test": "LOCKED",
    }
