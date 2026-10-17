"""Summarize the frozen-matrix HelpSteer2 Qwen LoRA experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _paired(results: list[dict[str, Any]], primary: str, baseline: str) -> dict[str, Any]:
    split_values = []
    split_seeds = sorted({int(row["split_seed"]) for row in results})
    for split_seed in split_seeds:
        primary_values = [
            row["metrics"]["pairwise_accuracy"] for row in results
            if row["strategy"] == primary and int(row["split_seed"]) == split_seed
        ]
        baseline_values = [
            row["metrics"]["pairwise_accuracy"] for row in results
            if row["strategy"] == baseline and int(row["split_seed"]) == split_seed
        ]
        if len(primary_values) != len(baseline_values):
            raise ValueError(f"unpaired split {split_seed}: {primary} vs {baseline}")
        split_values.append(_mean(primary_values) - _mean(baseline_values))
    mean = _mean(split_values)
    if len(split_values) > 1:
        se = statistics.stdev(split_values) / math.sqrt(len(split_values))
        critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(len(split_values) - 1, 1.96)
    else:
        se = 0.0
        critical = 0.0
    margin = critical * se
    return {
        "n": len(split_values),
        "mean_delta": mean,
        "ci95_t": [mean - margin, mean + margin],
        "wins_ties_losses": [
            sum(value > 1e-12 for value in split_values),
            sum(abs(value) <= 1e-12 for value in split_values),
            sum(value < -1e-12 for value in split_values),
        ],
        "split_level_deltas": dict(zip(map(str, split_seeds), split_values, strict=True)),
    }


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("results", [])
    strategies = report.get("configuration", {}).get("strategies", [])
    if not rows:
        raise ValueError("report contains no results")
    summary = {}
    for strategy in strategies:
        values = [row for row in rows if row["strategy"] == strategy]
        if values:
            summary[strategy] = {
                "runs": len(values),
                "mean_pairwise_accuracy": _mean(
                    [row["metrics"]["pairwise_accuracy"] for row in values]
                ),
                "mean_training_seconds": _mean(
                    [row["cost"]["training_seconds"] for row in values]
                    if all("training_seconds" in row["cost"] for row in values)
                    else []
                ),
            }
    primary = "impact-aware-random"
    paired = {
        baseline: _paired(rows, primary, baseline)
        for baseline in ("random", "disagreement-only", "seed-only")
        if baseline in {row["strategy"] for row in rows}
    }
    return {
        "status": report.get("status"),
        "protocol_version": report.get("protocol_version"),
        "results": len(rows),
        "final_test_accessed": report.get("configuration", {}).get("final_test_accessed"),
        "summary": summary,
        "paired_primary_vs_baseline": paired,
        "interpretation": (
            "Qwen LoRA downstream validation on a frozen simulated-feedback matrix; "
            "not a new-human-feedback or independent-dataset result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = analyze(json.loads(args.report.read_text(encoding="utf-8")))
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
