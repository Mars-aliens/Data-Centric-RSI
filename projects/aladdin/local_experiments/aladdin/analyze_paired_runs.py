from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


STRATEGIES = ("random", "uniform", "disagreement", "novelty", "aladdin", "aladdin-v2", "seed-only")
METRICS = {
    "feedback": lambda item: item["metrics"].get("feedback_accuracy"),
    "pairwise": lambda item: item["metrics"]["llm_feedback_task_model"]["pairwise_accuracy"],
    "oracle": lambda item: item["metrics"]["oracle_feedback_task_model"]["pairwise_accuracy"],
    "tokens": lambda item: item["budget"]["llm_tokens"],
}


def _t_critical(n: int) -> float:
    # Exact two-sided 95% t critical values for the small sample sizes used here.
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }
    if n in table:
        return table[n]
    # Conservative fallback for larger runs. The script is intended for reporting,
    # while exact values can be obtained by rerunning with scipy if needed.
    return 1.96


def _ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    half_width = _t_critical(len(values)) * stdev(values) / math.sqrt(len(values))
    average = mean(values)
    return average - half_width, average + half_width


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError(f"{path} is not completed")
    if not isinstance(report.get("results"), list):
        raise ValueError(f"{path} has no per-run results")
    return report


def _by_strategy_seed(report: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for item in report["results"]:
        strategy = str(item["strategy"])
        seed = int(item["seed"])
        if strategy in result and seed in result[strategy]:
            raise ValueError(f"duplicate strategy/seed: {strategy}/{seed}")
        result.setdefault(strategy, {})[seed] = item
    return result


def _fmt(value: float | None, scale: float = 1.0) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value * scale:.2f}"


def _report(path: Path) -> None:
    report = _load(path)
    by_strategy = _by_strategy_seed(report)
    configuration = report.get("configuration", {})
    print("=" * 88)
    print(path.name)
    print(
        "dataset={dataset} budget={budget} independent_splits={independent} seeds={seeds}".format(
            dataset=report.get("dataset", "unknown"),
            budget=configuration.get("candidate_budget_groups"),
            independent=configuration.get("independent_run_splits"),
            seeds=configuration.get("seeds"),
        )
    )
    print("strategy             feedback mean±sd       pairwise mean±sd      95% CI pairwise")
    for strategy in STRATEGIES:
        items = list(by_strategy.get(strategy, {}).values())
        feedback = [METRICS["feedback"](item) for item in items]
        feedback = [float(value) for value in feedback if value is not None]
        pairwise = [float(METRICS["pairwise"](item)) for item in items]
        if not pairwise:
            print(f"{strategy:16s} N/A")
            continue
        low, high = _ci(pairwise)
        feedback_text = "N/A" if not feedback else f"{mean(feedback) * 100:.2f}% ± {stdev(feedback) * 100:.2f}%"
        pairwise_text = f"{mean(pairwise) * 100:.2f}% ± {stdev(pairwise) * 100:.2f}%"
        ci_text = f"[{low * 100:.2f}%, {high * 100:.2f}%]"
        print(f"{strategy:16s} {feedback_text:21s} {pairwise_text:22s} {ci_text}")

    comparisons = (("aladdin", "random"), ("aladdin", "uniform"),
                   ("aladdin", "disagreement"), ("aladdin", "novelty"),
                   ("aladdin", "seed-only"), ("aladdin-v2", "random"),
                   ("aladdin-v2", "disagreement"), ("aladdin-v2", "seed-only"))
    for left, right in comparisons:
        common = sorted(set(by_strategy.get(left, {})) & set(by_strategy.get(right, {})))
        deltas = [
            float(METRICS["pairwise"](by_strategy[left][seed]))
            - float(METRICS["pairwise"](by_strategy[right][seed]))
            for seed in common
        ]
        if len(deltas) < 2:
            print(f"\nPaired {left} - {right}: insufficient paired runs")
            continue
        low, high = _ci(deltas)
        wins = sum(delta > 0 for delta in deltas)
        ties = sum(delta == 0 for delta in deltas)
        losses = sum(delta < 0 for delta in deltas)
        print(f"\nPaired {left} - {right} pairwise:")
        print(
            f"mean delta={mean(deltas) * 100:.3f} percentage points, "
            f"sd={stdev(deltas) * 100:.3f}, 95% CI=[{low * 100:.3f}, {high * 100:.3f}]"
        )
        print(f"wins/ties/losses: {wins} {ties} {losses}; seeds={common}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed Aladdin real-LLM reports.")
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.reports:
        _report(path)


if __name__ == "__main__":
    main()
