from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _paired_summary(values: list[float]) -> dict[str, Any]:
    mean = _mean(values)
    std = _sample_std(values)
    n = len(values)
    standard_error = std / math.sqrt(n) if n else 0.0
    critical = _T_CRITICAL_95.get(n - 1, 1.96) if n > 1 else None
    interval = (
        [mean - critical * standard_error, mean + critical * standard_error]
        if critical is not None
        else [mean, mean]
    )
    return {
        "n": n,
        "mean": mean,
        "sample_std": std,
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "approx_95_percent_ci": interval,
        "ci_method": "paired t interval; descriptive, not a preregistered significance test",
    }


def _metric(result: dict[str, Any], branch: str) -> float:
    return float(result["metrics"][branch]["pairwise_accuracy"])


def analyze_training_pilot(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
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
    reference = "seed-only" if "seed-only" in by_strategy else strategies[0]
    llm_by_seed: dict[str, dict[str, float]] = {}
    oracle_by_seed: dict[str, dict[str, float]] = {}
    for seed, seed_results in sorted(by_seed.items()):
        llm_by_seed[str(seed)] = {
            strategy: _metric(item, "llm_feedback_task_model")
            for strategy, item in sorted(seed_results.items())
        }
        oracle_by_seed[str(seed)] = {
            strategy: _metric(item, "oracle_feedback_task_model")
            for strategy, item in sorted(seed_results.items())
        }

    summary: dict[str, dict[str, Any]] = {}
    for strategy, strategy_results in sorted(by_strategy.items()):
        llm_values = [_metric(item, "llm_feedback_task_model") for item in strategy_results]
        oracle_values = [_metric(item, "oracle_feedback_task_model") for item in strategy_results]
        feedback_values = [
            float(item["metrics"]["feedback_accuracy"])
            for item in strategy_results
            if item["metrics"].get("feedback_accuracy") is not None
        ]
        summary[strategy] = {
            "llm_pairwise_mean": _mean(llm_values),
            "llm_pairwise_sample_std": _sample_std(llm_values),
            "oracle_pairwise_mean": _mean(oracle_values),
            "oracle_pairwise_sample_std": _sample_std(oracle_values),
            "feedback_accuracy_mean": _mean(feedback_values) if feedback_values else None,
            "feedback_accuracy_sample_std": _sample_std(feedback_values) if feedback_values else None,
            "llm_minus_oracle_pairwise_mean": _mean(
                [_metric(item, "llm_feedback_task_model") - _metric(item, "oracle_feedback_task_model") for item in strategy_results]
            ),
        }

    paired_vs_reference: dict[str, Any] = {}
    for strategy in strategies:
        if strategy == reference:
            continue
        common_seeds = sorted(set(by_seed) & {
            seed for seed, seed_results in by_seed.items() if strategy in seed_results and reference in seed_results
        })
        llm_deltas = [
            _metric(by_seed[seed][strategy], "llm_feedback_task_model")
            - _metric(by_seed[seed][reference], "llm_feedback_task_model")
            for seed in common_seeds
        ]
        oracle_deltas = [
            _metric(by_seed[seed][strategy], "oracle_feedback_task_model")
            - _metric(by_seed[seed][reference], "oracle_feedback_task_model")
            for seed in common_seeds
        ]
        paired_vs_reference[strategy] = {
            "reference": reference,
            "seeds": common_seeds,
            "llm_pairwise_delta_by_seed": llm_deltas,
            "llm_pairwise_delta": _paired_summary(llm_deltas),
            "oracle_pairwise_delta_by_seed": oracle_deltas,
            "oracle_pairwise_delta": _paired_summary(oracle_deltas),
        }

    budgets = [
        {
            "strategy": item["strategy"],
            "seed": item["seed"],
            "llm_calls": item.get("budget", {}).get("llm_calls"),
            "llm_tokens": item.get("budget", {}).get("llm_tokens"),
            "feedback_requests": item.get("budget", {}).get("feedback_requests"),
            "training_steps": item.get("budget", {}).get("training_steps"),
            "evaluator_calls": item.get("budget", {}).get("evaluator_calls"),
        }
        for item in results
    ]
    feedback_budget_signatures = {
        (item["feedback_requests"], item["training_steps"], item["evaluator_calls"])
        for item in budgets
        if item["strategy"] != "seed-only"
    }
    token_means: dict[str, float] = {}
    for strategy, strategy_results in by_strategy.items():
        tokens = [float(item.get("budget", {}).get("llm_tokens", 0)) for item in strategy_results]
        token_means[strategy] = _mean(tokens)

    warnings: list[str] = []
    seed_count = len(by_seed)
    if seed_count < 10:
        warnings.append("fewer than ten seeds are available; treat the comparison as preliminary")
    for strategy, comparison in paired_vs_reference.items():
        ci = comparison["llm_pairwise_delta"]["approx_95_percent_ci"]
        if ci[0] <= 0 <= ci[1]:
            warnings.append(f"{strategy} vs {reference}: the descriptive 95% interval crosses zero")
    if len(feedback_budget_signatures) != 1:
        warnings.append("feedback strategies do not share identical request/training/evaluation budgets")
    paid_token_means = {
        strategy: mean
        for strategy, mean in token_means.items()
        if strategy != "seed-only" and mean > 0
    }
    if paid_token_means:
        minimum_tokens = min(paid_token_means.values())
        maximum_tokens = max(paid_token_means.values())
        if maximum_tokens > minimum_tokens * 1.10:
            warnings.append(
                "LLM token consumption differs by more than 10% across feedback strategies; "
                "fixed-call results are not fixed-token-cost results"
            )
    if "aladdin" in summary and "seed-only" in summary:
        warnings.append("Aladdin's positive mean is small; freeze no final-test claim until a real Task Model pilot")
    warnings.append("TF-IDF is used as the pilot Task Model; these results are not Qwen fine-tuning or DPO results")

    return {
        "source_experiment": report.get("experiment"),
        "dataset": report.get("dataset"),
        "dataset_revision": report.get("dataset_revision"),
        "reference_strategy": reference,
        "seed_count": seed_count,
        "strategies": strategies,
        "llm_pairwise_by_seed": llm_by_seed,
        "oracle_pairwise_by_seed": oracle_by_seed,
        "strategy_summary": summary,
        "paired_deltas_vs_reference": paired_vs_reference,
        "feedback_budget_signatures_equal": len(feedback_budget_signatures) == 1,
        "mean_llm_tokens_by_strategy": token_means,
        "budgets": budgets,
        "warnings": warnings,
        "final_test_status": "not assessed by this analysis; remains sealed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an Aladdin real-LLM training pilot.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_training_pilot(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
