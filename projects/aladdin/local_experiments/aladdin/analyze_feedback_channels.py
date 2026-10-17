from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_STRATEGIES = ("random", "uniform", "disagreement", "novelty", "aladdin", "aladdin-v2")


def _t_critical(n: int) -> float:
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
        10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
        14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120,
        18: 2.110, 19: 2.101, 20: 2.093,
    }
    return table.get(n, 1.96)


def _ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    half = _t_critical(len(values)) * stdev(values) / math.sqrt(len(values))
    average = mean(values)
    return average - half, average + half


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError(f"{path} is not completed")
    if not isinstance(report.get("results"), list):
        raise ValueError(f"{path} has no per-seed results")
    return report


def _metric(item: dict[str, Any], channel: str) -> float:
    metrics = item["metrics"]
    if channel == "llm":
        return float(metrics["llm_feedback_task_model"]["pairwise_accuracy"])
    if channel == "oracle":
        return float(metrics["oracle_feedback_task_model"]["pairwise_accuracy"])
    raise ValueError(f"unknown channel: {channel}")


def _paired_line(label: str, deltas: list[float]) -> None:
    low, high = _ci(deltas)
    wins = sum(value > 0 for value in deltas)
    ties = sum(value == 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    print(
        f"{label}: mean={mean(deltas) * 100:.3f} pp, "
        f"sd={stdev(deltas) * 100:.3f}, "
        f"95% CI=[{low * 100:.3f}, {high * 100:.3f}], "
        f"wins/ties/losses={wins}/{ties}/{losses}"
    )


def _report(path: Path) -> None:
    report = _load(path)
    rows = {
        (str(item["strategy"]), int(item["seed"])): item
        for item in report["results"]
    }
    strategies = [
        strategy for strategy in DEFAULT_STRATEGIES
        if any(key[0] == strategy for key in rows)
    ]
    seed_only = {
        seed: item for (strategy, seed), item in rows.items()
        if strategy == "seed-only"
    }
    if not seed_only:
        raise ValueError(f"{path} has no seed-only results")

    print("=" * 96)
    print(path.name)
    print(f"dataset={report.get('dataset')} seeds={sorted(seed_only)}")
    print("strategy          LLM mean±sd       Oracle mean±sd     Oracle-LLM mean")

    for strategy in strategies:
        seeds = sorted(set(seed_only) & {seed for candidate, seed in rows if candidate == strategy})
        llm = [_metric(rows[(strategy, seed)], "llm") for seed in seeds]
        oracle = [_metric(rows[(strategy, seed)], "oracle") for seed in seeds]
        gaps = [oracle_value - llm_value for oracle_value, llm_value in zip(oracle, llm)]
        llm_text = f"{mean(llm) * 100:.2f}% +/- {stdev(llm) * 100:.2f}%"
        oracle_text = f"{mean(oracle) * 100:.2f}% +/- {stdev(oracle) * 100:.2f}%"
        print(f"{strategy:16s} {llm_text:18s} {oracle_text:18s} {mean(gaps) * 100:+.3f} pp")

    print("\nChannel diagnosis against seed-only:")
    for strategy in strategies:
        seeds = sorted(set(seed_only) & {seed for candidate, seed in rows if candidate == strategy})
        llm_delta = [
            _metric(rows[(strategy, seed)], "llm") - _metric(seed_only[seed], "llm")
            for seed in seeds
        ]
        oracle_delta = [
            _metric(rows[(strategy, seed)], "oracle") - _metric(seed_only[seed], "llm")
            for seed in seeds
        ]
        print(f"\n{strategy} (n={len(seeds)}):")
        _paired_line("  real LLM - seed-only", llm_delta)
        _paired_line("  oracle - seed-only", oracle_delta)

    if "aladdin" in strategies:
        for reference in ("random", "disagreement", "novelty"):
            if reference not in strategies:
                continue
            seeds = sorted({seed for strategy, seed in rows if strategy == "aladdin"}
                           & {seed for strategy, seed in rows if strategy == reference})
            print(f"\nAladdin - {reference} (n={len(seeds)}):")
            _paired_line(
                "  real LLM",
                [_metric(rows[("aladdin", seed)], "llm") - _metric(rows[(reference, seed)], "llm") for seed in seeds],
            )
            _paired_line(
                "  oracle",
                [_metric(rows[("aladdin", seed)], "oracle") - _metric(rows[(reference, seed)], "oracle") for seed in seeds],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose selection benefit versus LLM feedback noise.")
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.reports:
        _report(path)


if __name__ == "__main__":
    main()
