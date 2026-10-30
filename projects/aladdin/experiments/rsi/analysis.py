"""Summarize paired development outcomes and human-feedback costs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONTRASTS = {
    "overall_recursive_minus_fixed": ("recursive_policy", "fixed_policy"),
    "ablation_freeze_synthesis_minus_full": ("freeze_synthesis", "recursive_policy"),
    "ablation_freeze_cleaning_minus_full": ("freeze_cleaning", "recursive_policy"),
    "ablation_freeze_selection_minus_full": ("freeze_selection", "recursive_policy"),
    "query_random_minus_value": ("query_random", "query_feedback_value"),
    "query_uncertainty_minus_value": ("query_uncertainty", "query_feedback_value"),
}


def _per_seed_scores(arm: dict[str, Any]) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for row in arm.get("rounds", []):
        score = row.get("training", {}).get("pairwise_accuracy")
        if score is None:
            continue
        by_seed.setdefault(int(row["seed"]), []).append(float(score))
    return {seed: sum(values) / len(values) for seed, values in by_seed.items() if values}


def _paired_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_scores = _per_seed_scores(left)
    right_scores = _per_seed_scores(right)
    common = sorted(set(left_scores) & set(right_scores))
    deltas = {str(seed): left_scores[seed] - right_scores[seed] for seed in common}
    values = list(deltas.values())
    return {
        "paired_seeds": len(common),
        "mean_delta": sum(values) / len(values) if values else None,
        "per_seed_delta": deltas,
        "wins_ties_losses": [
            sum(value > 0 for value in values),
            sum(value == 0 for value in values),
            sum(value < 0 for value in values),
        ],
        "inferential_claim_supported": False,
    }


def analyze_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "completed":
        raise ValueError("only completed experiment runs can be analyzed")
    data = result.get("data") or {}
    if data.get("final_test_accessed") is not False:
        raise ValueError("development analysis requires final_test_accessed=false")
    arms = result.get("arms") or {}
    comparisons = {}
    for name, (left_name, right_name) in CONTRASTS.items():
        if left_name in arms and right_name in arms:
            comparisons[name] = _paired_delta(arms[left_name], arms[right_name])
    return {
        "status": "analyzed",
        "protocol": result.get("protocol"),
        "run_id": result.get("run_id"),
        "comparison": result.get("comparison"),
        "feedback_mode": result.get("configuration", {}).get("feedback_mode", "human"),
        "feedback_model": result.get("configuration", {}).get("feedback_model") if result.get("configuration", {}).get("feedback_mode") == "model" else None,
        "final_test_accessed": False,
        "arm_summary": result.get("summary", {}),
        "paired_development_comparisons": comparisons,
        "interpretation": (
            "Development-only diagnostics. The results are not final-test evidence; "
            "rounds are averaged within seed and are not independent replicates."
        ),
    }


def analyze_file(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    source = Path(input_path)
    result = json.loads(source.read_text(encoding="utf-8"))
    report = analyze_result(result)
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report
