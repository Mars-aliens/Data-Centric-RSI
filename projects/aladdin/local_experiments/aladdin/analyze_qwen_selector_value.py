from __future__ import annotations

"""Analyze whether Qwen-selector uncertainty predicts feedback correctness."""

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

T95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
    8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
    14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
    20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
    26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}
STRATEGIES = ("aladdin-qwen", "random")
CORRECTNESS_FIELDS = {
    "llm_accuracy": "llm_feedback_correct",
    "selector_accuracy_vs_official": "selector_predicts_official_preference",
}
SELECTED_METRIC_FIELDS = {
    "llm_accuracy": "llm_feedback_accuracy",
    "selector_accuracy_vs_official": "selector_pairwise_accuracy_vs_official_label",
}


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    average = mean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": average, "ci95_low": None, "ci95_high": None}
    half = T95.get(len(values), 1.96) * stdev(values) / sqrt(len(values))
    return {
        "n": len(values),
        "mean": average,
        "ci95_low": average - half,
        "ci95_high": average + half,
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss == 0 or right_ss == 0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / sqrt(left_ss * right_ss)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _terciles(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ranked = sorted(
        rows,
        key=lambda row: (float(row["selector_uncertainty"]), str(row["group_id"])),
    )
    if len(ranked) < 3:
        raise ValueError("each seed/strategy needs at least three selected groups")
    low_end = len(ranked) // 3
    high_start = (2 * len(ranked)) // 3
    return {
        "low_uncertainty": ranked[:low_end],
        "middle_uncertainty": ranked[low_end:high_start],
        "high_uncertainty": ranked[high_start:],
    }


def _bin_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"groups": len(rows)}
    for name, field in CORRECTNESS_FIELDS.items():
        values = [row[field] for row in rows if row.get(field) is not None]
        result[name] = mean(float(value) for value in values) if values else None
        result[f"{name}_n"] = len(values)
    result["mean_uncertainty"] = mean(
        float(row["selector_uncertainty"]) for row in rows
    ) if rows else None
    return result


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "completed":
        raise ValueError(f"selection audit is not completed: {data.get('status')!r}")
    seeds = data.get("by_seed", {})
    if not seeds:
        raise ValueError("selection audit has no per-seed rows")

    result: dict[str, Any] = {
        "source_report": data.get("source_report"),
        "source_status": data.get("status"),
        "inference_scope": (
            "Uncertainty bins are computed within each seed/strategy. Confidence intervals "
            "use seed-level paired high-minus-low differences, not individual groups as independent samples. "
            "All seeds share the same prompt split; results are conditional on that split."
        ),
        "by_seed": {},
    }
    paired: dict[str, dict[str, list[float]]] = {
        strategy: {metric: [] for metric in CORRECTNESS_FIELDS}
        for strategy in STRATEGIES
    }
    seed_quality_deltas: dict[str, list[float]] = {
        metric: [] for metric in CORRECTNESS_FIELDS
    }
    quality_downstream_pairs: dict[str, dict[str, list[tuple[float, float]]]] = {
        metric: {channel: [] for channel in ("llm_feedback", "oracle_feedback")}
        for metric in CORRECTNESS_FIELDS
    }

    for seed in sorted(seeds, key=lambda value: int(value)):
        seed_row = seeds[seed]
        per_strategy: dict[str, Any] = {}
        for strategy in STRATEGIES:
            rows = seed_row["selected"][strategy].get("selected_group_audit")
            if not rows:
                raise ValueError(f"missing selected-group records for seed={seed}, strategy={strategy}")
            bins = _terciles(rows)
            metrics_by_bin = {name: _bin_metrics(values) for name, values in bins.items()}
            per_strategy[strategy] = {
                "selected_groups": len(rows),
                "uncertainty_terciles": metrics_by_bin,
                "high_minus_low": {},
            }
            for metric in CORRECTNESS_FIELDS:
                low = metrics_by_bin["low_uncertainty"][metric]
                high = metrics_by_bin["high_uncertainty"][metric]
                delta = high - low if high is not None and low is not None else None
                per_strategy[strategy]["high_minus_low"][metric] = delta
                if delta is not None:
                    paired[strategy][metric].append(float(delta))

        result["by_seed"][seed] = {
            "selection_overlap_groups": seed_row.get("selection_overlap_groups"),
            "strategies": per_strategy,
            "quality_delta_aladdin_qwen_minus_random": {},
            "downstream_delta_aladdin_qwen_minus_random": seed_row.get(
                "existing_downstream_delta_aladdin_qwen_minus_random", {}
            ),
        }
        for metric in CORRECTNESS_FIELDS:
            selected_field = SELECTED_METRIC_FIELDS[metric]
            left = seed_row["selected"]["aladdin-qwen"][selected_field]
            right = seed_row["selected"]["random"][selected_field]
            delta = left - right if left is not None and right is not None else None
            result["by_seed"][seed]["quality_delta_aladdin_qwen_minus_random"][metric] = delta
            if delta is not None:
                seed_quality_deltas[metric].append(float(delta))
        downstream = result["by_seed"][seed]["downstream_delta_aladdin_qwen_minus_random"]
        for quality_metric in CORRECTNESS_FIELDS:
            quality_delta = result["by_seed"][seed]["quality_delta_aladdin_qwen_minus_random"][quality_metric]
            if quality_delta is None:
                continue
            for channel in ("llm_feedback", "oracle_feedback"):
                downstream_delta = downstream.get(channel)
                if downstream_delta is not None:
                    quality_downstream_pairs[quality_metric][channel].append(
                        (float(quality_delta), float(downstream_delta))
                    )

    result["high_minus_low_uncertainty_paired_by_seed"] = {
        strategy: {
            metric: _summary(values)
            for metric, values in metric_values.items()
        }
        for strategy, metric_values in paired.items()
    }
    result["mean_quality_delta_aladdin_qwen_minus_random"] = {
        metric: _summary(values) for metric, values in seed_quality_deltas.items()
    }
    result["seed_level_pearson_correlations_exploratory"] = {
        quality_metric: {
            channel: _pearson(
                [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            )
            for channel, pairs in quality_downstream_pairs[quality_metric].items()
        }
        for quality_metric in seed_quality_deltas
    }
    result["status"] = "completed"
    return result


def _print(result: dict[str, Any]) -> None:
    print("Within-seed uncertainty terciles: low / middle / high")
    print("strategy       seed   LLM-correct low->high   selector-correct low->high")
    for seed, seed_row in result["by_seed"].items():
        for strategy in STRATEGIES:
            row = seed_row["strategies"][strategy]["uncertainty_terciles"]
            llm = [row[name]["llm_accuracy"] for name in (
                "low_uncertainty", "middle_uncertainty", "high_uncertainty"
            )]
            selector = [row[name]["selector_accuracy_vs_official"] for name in (
                "low_uncertainty", "middle_uncertainty", "high_uncertainty"
            )]
            fmt = lambda values: "/".join("NA" if value is None else f"{value:.3f}" for value in values)
            print(f"{strategy:14s} {seed:>4s}   {fmt(llm):20s}   {fmt(selector)}")

    print("\nPaired high-uncertainty minus low-uncertainty across seeds:")
    for strategy, metrics in result["high_minus_low_uncertainty_paired_by_seed"].items():
        for metric, summary in metrics.items():
            if summary["mean"] is None:
                print(f"{strategy} {metric}: unavailable")
                continue
            ci = "CI unavailable" if summary["ci95_low"] is None else (
                f"95% CI=[{summary['ci95_low'] * 100:+.2f}, "
                f"{summary['ci95_high'] * 100:+.2f}] pp"
            )
            print(
                f"{strategy} {metric}: {summary['mean'] * 100:+.2f} pp "
                f"({ci}, n={summary['n']})"
            )

    print("\nMean selected-set quality delta (aladdin-qwen - random):")
    for metric, summary in result["mean_quality_delta_aladdin_qwen_minus_random"].items():
        if summary["mean"] is None:
            print(f"{metric}: unavailable")
        else:
            print(
                f"{metric}: {summary['mean'] * 100:+.2f} pp "
                f"(n={summary['n']}, seed-level paired summary)"
            )

    print("\nExploratory seed-level Pearson r: quality delta vs downstream delta")
    for metric, channels in result["seed_level_pearson_correlations_exploratory"].items():
        print(
            f"{metric}: " + ", ".join(
                f"{channel}={value:.3f}" if value is not None else f"{channel}=NA"
                for channel, value in channels.items()
            )
        )
    print("\nCorrelations are descriptive only; n is seeds and all seeds share one prompt split.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = analyze(json.loads(args.audit_json.read_text(encoding="utf-8")))
    _print(analysis)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(analysis, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(f"wrote analysis: {args.output}")


if __name__ == "__main__":
    main()
