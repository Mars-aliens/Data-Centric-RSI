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


def _per_seed_final_scores(arm: dict[str, Any]) -> dict[int, float]:
    by_seed: dict[int, tuple[int, float]] = {}
    for row in arm.get("rounds", []):
        score = row.get("training", {}).get("pairwise_accuracy")
        if score is None:
            continue
        seed = int(row["seed"])
        round_index = int(row.get("round", -1))
        previous = by_seed.get(seed)
        if previous is None or round_index > previous[0]:
            by_seed[seed] = (round_index, float(score))
    return {seed: value for seed, (_, value) in by_seed.items()}


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


def _factorial_effects(arms: dict[str, Any]) -> dict[str, Any] | None:
    factors = ("synthesis", "cleaning", "selection")
    names = {
        (0, 0, 0): "fixed_policy",
        (1, 0, 0): "synthesis_only",
        (0, 1, 0): "cleaning_only",
        (0, 0, 1): "selection_only",
        (1, 1, 0): "synthesis_cleaning",
        (1, 0, 1): "synthesis_selection",
        (0, 1, 1): "cleaning_selection",
        (1, 1, 1): "recursive_policy",
    }
    if not set(names.values()).issubset(arms):
        return None
    scores = {bits: _per_seed_final_scores(arms[name]) for bits, name in names.items()}
    common_seeds = sorted(set.intersection(*(set(value) for value in scores.values())))
    per_seed: dict[str, dict[str, float]] = {}
    for seed in common_seeds:
        y = {bits: scores[bits][seed] for bits in names}
        effects = {}
        for index, factor in enumerate(factors):
            enabled = [value for bits, value in y.items() if bits[index] == 1]
            disabled = [value for bits, value in y.items() if bits[index] == 0]
            effects[f"main_{factor}"] = sum(enabled) / len(enabled) - sum(disabled) / len(disabled)
        for left, right, label in ((0, 1, "synthesis_x_cleaning"), (0, 2, "synthesis_x_selection"), (1, 2, "cleaning_x_selection")):
            values = []
            for fixed in (0, 1):
                bits_11 = [0, 0, 0]
                bits_10 = [0, 0, 0]
                bits_01 = [0, 0, 0]
                bits_00 = [0, 0, 0]
                bits_11[left] = bits_11[right] = 1
                bits_10[left] = 1
                bits_01[right] = 1
                bits_11[3 - left - right] = fixed
                bits_10[3 - left - right] = fixed
                bits_01[3 - left - right] = fixed
                bits_00[3 - left - right] = fixed
                values.append(y[tuple(bits_11)] - y[tuple(bits_10)] - y[tuple(bits_01)] + y[tuple(bits_00)])
            effects[f"interaction_{label}"] = sum(values) / len(values)
        effects["interaction_synthesis_x_cleaning_x_selection"] = (
            y[(1, 1, 1)] - y[(1, 1, 0)] - y[(1, 0, 1)] - y[(0, 1, 1)]
            + y[(1, 0, 0)] + y[(0, 1, 0)] + y[(0, 0, 1)] - y[(0, 0, 0)]
        )
        per_seed[str(seed)] = effects
    effect_names = list(next(iter(per_seed.values()))) if per_seed else []
    summary = {
        name: sum(values[name] for values in per_seed.values()) / len(per_seed) if per_seed else None
        for name in effect_names
    }
    return {
        "paired_seeds": len(common_seeds),
        "per_seed_effects": per_seed,
        "mean_effects": summary,
        "inferential_claim_supported": False,
        "outcome": "last completed round per seed",
        "note": "Effects are descriptive until the preregistered seed-level analysis is applied.",
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
        "factorial_update_effects": _factorial_effects(arms),
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
