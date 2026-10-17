"""Diagnose whether HelpSteer2 impact-aware feedback changes decisions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STRATEGIES = (
    "seed-only",
    "random",
    "disagreement-only",
    "impact-aware",
    "impact-aware-random",
)
PAIRWISE_STRATEGIES = tuple(strategy for strategy in STRATEGIES if strategy != "seed-only")


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _condition_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["split_seed"]),
        int(row["policy_seed"]),
        str(row["target_profile_simulator"]),
    )


def _trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        trace
        for row in rows
        for trace in row.get("query_trace", [])
        if trace.get("expected_selection_change") is not None
    ]
    expected_changes = [float(trace["expected_selection_change"]) for trace in scored]
    impact_values = [float(trace["impact_value"]) for trace in scored]
    disagreements = [float(trace["disagreement"]) for trace in scored]
    scores = [float(trace["selection_score"]) for trace in scored]
    nonzero = sum(value > 1e-12 for value in expected_changes)
    return {
        "runs": len(rows),
        "scored_queries": len(scored),
        "mean_disagreement": _mean(disagreements),
        "mean_expected_selection_change": _mean(expected_changes),
        "mean_impact_value": _mean(impact_values),
        "mean_selection_score": _mean(scores),
        "nonzero_expected_change_fraction": (
            nonzero / len(expected_changes) if expected_changes else None
        ),
        "max_expected_selection_change": max(expected_changes, default=None),
        "max_impact_value": max(impact_values, default=None),
    }


def _pairwise_set_diagnostics(
    rows_by_condition: dict[tuple[int, int, str], dict[str, dict[str, Any]]],
    left_strategy: str,
    right_strategy: str,
) -> dict[str, Any]:
    query_jaccards: list[float] = []
    selected_jaccards: list[float] = []
    query_different = 0
    selected_different = 0
    paired = 0
    for condition, by_strategy in rows_by_condition.items():
        if left_strategy not in by_strategy or right_strategy not in by_strategy:
            continue
        paired += 1
        left = by_strategy[left_strategy]
        right = by_strategy[right_strategy]
        left_queries = {trace["group_id"] for trace in left.get("query_trace", [])}
        right_queries = {trace["group_id"] for trace in right.get("query_trace", [])}
        left_selected = set(left.get("selected_data_ids", []))
        right_selected = set(right.get("selected_data_ids", []))
        query_jaccards.append(_jaccard(left_queries, right_queries))
        selected_jaccards.append(_jaccard(left_selected, right_selected))
        query_different += left_queries != right_queries
        selected_different += left_selected != right_selected
    return {
        "left": left_strategy,
        "right": right_strategy,
        "paired_conditions": paired,
        "mean_query_set_jaccard": _mean(query_jaccards),
        "mean_selected_data_set_jaccard": _mean(selected_jaccards),
        "different_query_set_conditions": query_different,
        "different_selected_data_set_conditions": selected_different,
    }


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("results", [])
    rows_by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_condition: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        strategy = str(row["strategy"])
        rows_by_strategy[strategy].append(row)
        rows_by_condition[_condition_key(row)][strategy] = row

    pairwise = {}
    for left, right in (
        ("impact-aware", "disagreement-only"),
        ("impact-aware-random", "random"),
        ("impact-aware-random", "disagreement-only"),
    ):
        pairwise[f"{left}_vs_{right}"] = _pairwise_set_diagnostics(
            rows_by_condition, left, right
        )

    return {
        "source_status": report.get("status"),
        "source_result_count": len(rows),
        "final_test_accessed": report.get("final_test_accessed"),
        "strategy_trace_summary": {
            strategy: _trace_summary(rows_by_strategy.get(strategy, []))
            for strategy in STRATEGIES
        },
        "paired_set_diagnostics": pairwise,
        "interpretation": {
            "impact_changes_query_order": pairwise[
                "impact-aware_vs_disagreement-only"
            ]["different_query_set_conditions"] > 0,
            "impact_changes_downstream_selection": pairwise[
                "impact-aware_vs_disagreement-only"
            ]["different_selected_data_set_conditions"] > 0,
            "note": (
                "This is a post-hoc mechanism diagnostic. It does not change the "
                "frozen evaluation or provide independent evidence of effectiveness."
            ),
        },
    }


def _format(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def print_report(analysis: dict[str, Any]) -> None:
    print(f"status: {analysis['source_status']}")
    print(f"results: {analysis['source_result_count']}")
    print(f"final_test_accessed: {analysis['final_test_accessed']}")
    print()
    print("Impact diagnostics on scored (non-random) query decisions")
    print("strategy                 scored  nonzero-change  mean-change  mean-impact")
    for strategy, row in analysis["strategy_trace_summary"].items():
        print(
            f"{strategy:24s} {row['scored_queries']:6d}  "
            f"{_format(row['nonzero_expected_change_fraction']):>15s}  "
            f"{_format(row['mean_expected_selection_change']):>11s}  "
            f"{_format(row['mean_impact_value']):>11s}"
        )
    print()
    print("Paired set diagnostics")
    print("comparison                              paired  query-Jaccard  selected-Jaccard  changed-selected")
    for name, row in analysis["paired_set_diagnostics"].items():
        print(
            f"{name:40s} {row['paired_conditions']:6d}  "
            f"{_format(row['mean_query_set_jaccard']):>13s}  "
            f"{_format(row['mean_selected_data_set_jaccard']):>16s}  "
            f"{row['different_selected_data_set_conditions']:>16d}"
        )
    print()
    interpretation = analysis["interpretation"]
    print("impact changes query order:", interpretation["impact_changes_query_order"])
    print("impact changes downstream selection:", interpretation["impact_changes_downstream_selection"])
    print(interpretation["note"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = analyze_report(report)
    print_report(analysis)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote analysis: {args.output}")


if __name__ == "__main__":
    main()
