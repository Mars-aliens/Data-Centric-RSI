from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("report has no results")

    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        strategy = str(result["strategy"])
        seed = int(result["seed"])
        by_seed[seed][strategy] = result
        by_strategy[strategy].append(result)

    strategies = sorted(by_strategy)
    reference = "aladdin" if "aladdin" in by_strategy else strategies[0]
    pairwise_by_seed: dict[str, dict[str, float]] = {}
    held_out_by_seed: dict[str, dict[str, float]] = {}
    for seed, seed_results in sorted(by_seed.items()):
        pairwise_by_seed[str(seed)] = {
            strategy: float(item["metrics"]["pairwise_accuracy"])
            for strategy, item in sorted(seed_results.items())
        }
        held_out_by_seed[str(seed)] = {
            strategy: float(item["metrics"]["accuracy"])
            for strategy, item in sorted(seed_results.items())
        }

    deltas: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        if strategy == reference:
            continue
        pairwise_deltas = [
            float(by_seed[seed][reference]["metrics"]["pairwise_accuracy"])
            - float(by_seed[seed][strategy]["metrics"]["pairwise_accuracy"])
            for seed in sorted(by_seed)
            if reference in by_seed[seed] and strategy in by_seed[seed]
        ]
        held_out_deltas = [
            float(by_seed[seed][reference]["metrics"]["accuracy"])
            - float(by_seed[seed][strategy]["metrics"]["accuracy"])
            for seed in sorted(by_seed)
            if reference in by_seed[seed] and strategy in by_seed[seed]
        ]
        deltas[strategy] = {
            "reference": reference,
            "pairwise_delta_by_seed": pairwise_deltas,
            "pairwise_delta_mean": _mean(pairwise_deltas),
            "pairwise_delta_sample_std": _sample_std(pairwise_deltas),
            "pairwise_delta_positive_seeds": sum(delta > 0 for delta in pairwise_deltas),
            "held_out_delta_by_seed": held_out_deltas,
            "held_out_delta_mean": _mean(held_out_deltas),
            "held_out_delta_sample_std": _sample_std(held_out_deltas),
        }

    selected_groups: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for strategy, strategy_results in by_strategy.items():
        for result in strategy_results:
            ids = result.get("diagnostics", {}).get("selected_group_ids", [])
            selected_groups[strategy][str(result["seed"])] = {str(item) for item in ids}

    overlap_with_reference: dict[str, dict[str, float]] = {}
    if reference in selected_groups:
        for strategy, seed_groups in selected_groups.items():
            if strategy == reference:
                continue
            overlap_with_reference[strategy] = {
                seed: _jaccard(selected_groups[reference].get(seed, set()), groups)
                for seed, groups in sorted(seed_groups.items())
            }

    budgets = []
    for result in results:
        budget = result.get("budget", {})
        budgets.append({
            "strategy": result["strategy"],
            "seed": result["seed"],
            "feedback_requests": budget.get("feedback_requests"),
            "feedback_records_released": budget.get("feedback_records_released"),
            "training_steps": budget.get("training_steps"),
            "evaluator_calls": budget.get("evaluator_calls"),
            "llm_calls": budget.get("llm_calls"),
        })
    budget_fields = (
        "feedback_requests",
        "feedback_records_released",
        "training_steps",
        "evaluator_calls",
        "llm_calls",
    )
    budget_signatures = {
        tuple(item[field] for field in budget_fields)
        for item in budgets
    }

    warnings: list[str] = []
    if len(by_seed) < 5:
        warnings.append("fewer than five seeds are available; treat deltas as descriptive, not significance evidence")
    for strategy, delta in deltas.items():
        if delta["pairwise_delta_mean"] <= 0:
            warnings.append(f"{reference} does not beat {strategy} on mean pairwise accuracy")
        elif delta["pairwise_delta_positive_seeds"] < len(delta["pairwise_delta_by_seed"]):
            warnings.append(f"{reference}'s pairwise lead over {strategy} is not present for every seed")
    if len(budget_signatures) != 1:
        warnings.append("budget signatures are not identical across runs")
    if report.get("final_test_path") is not None:
        warnings.append("final_test_path is recorded; verify that the path was not loaded before final freeze")

    return {
        "source_experiment": report.get("experiment"),
        "dataset": report.get("dataset"),
        "dataset_revision": report.get("dataset_revision"),
        "adapter_version": report.get("adapter_version"),
        "reference_strategy": reference,
        "seed_count": len(by_seed),
        "strategies": strategies,
        "pairwise_accuracy_by_seed": pairwise_by_seed,
        "held_out_accuracy_by_seed": held_out_by_seed,
        "deltas_vs_reference": deltas,
        "selection_jaccard_vs_reference": overlap_with_reference,
        "budget_signatures_equal": len(budget_signatures) == 1,
        "budgets": budgets,
        "warnings": warnings,
        "final_test_status": "not assessed by this analysis; remains sealed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an Aladdin HH-RLHF replay report without loading data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
