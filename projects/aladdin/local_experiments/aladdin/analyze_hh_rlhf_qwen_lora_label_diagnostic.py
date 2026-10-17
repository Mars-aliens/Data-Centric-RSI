from __future__ import annotations

"""Paired analysis for the HH-RLHF Qwen label-noise diagnostic."""

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

FORMAL_PROTOCOL_VERSION = "formal-hh-rlhf-qwen-v1"
FORMAL_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
FORMAL_STRATEGIES = {"seed-only", "random", "aladdin-v2"}
FORMAL_SEEDS = set(range(17, 27))
QWEN_DIAGNOSTIC_STRATEGIES = {
    "seed-only",
    "random",
    "aladdin-qwen",
    "aladdin-qwen-reliable",
}


def _paired(
    by_strategy: dict[str, dict[int, float]], left: str, right: str
) -> tuple[list[int], list[float]]:
    seeds = sorted(set(by_strategy.get(left, {})) & set(by_strategy.get(right, {})))
    return seeds, [by_strategy[left][seed] - by_strategy[right][seed] for seed in seeds]


def _report(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol_version") == FORMAL_PROTOCOL_VERSION:
        _validate_formal_report(data, path)
    elif (
        data.get("protocol_version") == "custom-diagnostic-v1"
        and QWEN_DIAGNOSTIC_STRATEGIES.intersection(
            {row.get("strategy") for row in data.get("results", [])}
        )
        and data.get("status") == "completed"
    ):
        _validate_qwen_diagnostic_report(data, path)
    rows = data.get("results", [])
    strategies = sorted({row["strategy"] for row in rows})
    channels = ("llm_feedback", "oracle_feedback")
    by: dict[str, dict[str, dict[int, float]]] = {
        channel: {strategy: {} for strategy in strategies} for channel in channels
    }
    feedback: dict[str, list[float]] = {strategy: [] for strategy in strategies}
    for row in rows:
        strategy = row["strategy"]
        seed = int(row["seed"])
        for channel in channels:
            metric = row.get("metrics", {}).get(channel, {}).get("pairwise_accuracy")
            if metric is not None:
                by[channel][strategy][seed] = float(metric)
        if row.get("feedback_accuracy") is not None:
            feedback[strategy].append(float(row["feedback_accuracy"]))

    print("=" * 96)
    print(path.name)
    print("status:", data.get("status"), "seeds:", sorted({int(r["seed"]) for r in rows}))
    configuration = data.get("configuration", {})
    if "split_unit" in configuration:
        print(
            "split protocol:",
            f"unit={configuration['split_unit']}, "
            f"fail_on_overlap={configuration.get('fail_on_split_overlap', False)}",
        )
        audits = configuration.get("split_audit_by_seed", {})
        if audits:
            split_seeds = sorted(
                {int(audit["split_seed"]) for audit in audits.values() if "split_seed" in audit}
            )
            overlap_counts = [
                int(audit.get("prompts_with_multiple_splits", 0))
                for audit in audits.values()
            ]
            print(
                "split audit:",
                f"seeds={len(audits)}, "
                f"split_seeds={split_seeds}, "
                f"prompt-overlap range={min(overlap_counts)}..{max(overlap_counts)}",
            )
    steps_by_strategy: dict[str, set[int]] = {}
    for row in rows:
        steps = row.get("training_max_steps")
        if steps is not None:
            steps_by_strategy.setdefault(row["strategy"], set()).add(int(steps))
    if steps_by_strategy:
        print(
            "training steps:",
            ", ".join(
                f"{strategy}={sorted(values)}"
                for strategy, values in sorted(steps_by_strategy.items())
            ),
        )
    print("strategy        feedback acc       LLM-label pairwise     Oracle-label pairwise")
    for strategy in strategies:
        acc = mean(feedback[strategy]) if feedback[strategy] else float("nan")
        channel_values = []
        for channel in channels:
            values = list(by[channel][strategy].values())
            deviation = stdev(values) if len(values) > 1 else 0.0
            channel_values.append(
                f"{mean(values) * 100:.2f}% +/- {deviation * 100:.2f}%"
                if values
                else "N/A"
            )
        acc_text = f"{acc * 100:.2f}%" if feedback[strategy] else "N/A"
        print(f"{strategy:15s} {acc_text:15s} {channel_values[0]:22s} {channel_values[1]}")

    primary_strategy = next(
        (
            strategy
            for strategy in (
                "aladdin-v2",
                "aladdin-qwen-reliable",
                "aladdin-qwen",
            )
            if strategy in strategies
        ),
        None,
    )
    print("\nPaired strategy differences (percentage points):")
    if primary_strategy is None:
        print("  no Aladdin strategy found")
    else:
        print("  primary strategy:", primary_strategy)
    for channel in channels:
        print(f"\n{channel}:")
        if primary_strategy is None or primary_strategy not in by[channel]:
            continue
        for baseline in ("random", "seed-only"):
            if baseline not in by[channel]:
                continue
            seeds, deltas = _paired(by[channel], primary_strategy, baseline)
            if len(deltas) < 2:
                print(f"  {primary_strategy} - {baseline}: insufficient paired runs")
                continue
            avg = mean(deltas)
            half = T95.get(len(deltas), 1.96) * stdev(deltas) / sqrt(len(deltas))
            wins = sum(value > 0 for value in deltas)
            ties = sum(value == 0 for value in deltas)
            losses = sum(value < 0 for value in deltas)
            print(
                f"  {primary_strategy} - {baseline}: {avg * 100:+.3f} pp "
                f"95% CI=[{(avg-half)*100:+.3f}, {(avg+half)*100:+.3f}], "
                f"wins/ties/losses={wins}/{ties}/{losses}, n={len(seeds)}"
            )

    print("\nLabel-channel effect (LLM label - oracle label):")
    for strategy in strategies:
        seeds = sorted(set(by["llm_feedback"][strategy]) & set(by["oracle_feedback"][strategy]))
        deltas = [
            by["llm_feedback"][strategy][seed] - by["oracle_feedback"][strategy][seed]
            for seed in seeds
        ]
        if deltas:
            print(f"  {strategy}: mean={mean(deltas) * 100:+.3f} pp, paired n={len(deltas)}")

    seed_only_rows = [row for row in rows if row["strategy"] == "seed-only"]
    if seed_only_rows:
        checks = [
            row.get("reproducibility", {}).get("identical_seed_only_adapters")
            for row in seed_only_rows
        ]
        available_checks = [value for value in checks if value is not None]
        deltas = [
            float(row["reproducibility"]["seed_only_metric_delta"])
            for row in seed_only_rows
            if row.get("reproducibility", {}).get("seed_only_metric_delta") is not None
        ]
        print("\nReproducibility check (seed-only):")
        if available_checks:
            print(
                "  identical adapter hashes: "
                f"{sum(bool(value) for value in available_checks)}/{len(available_checks)}"
            )
        else:
            print("  adapter hash comparison: unavailable in this report")
        for row in seed_only_rows:
            reproduction = row.get("reproducibility", {})
            checkpoints = reproduction.get("branch_checkpoints", {})
            llm_hash = checkpoints.get("llm_feedback", {}).get("adapter_hash")
            oracle_hash = checkpoints.get("oracle_feedback", {}).get("adapter_hash")
            delta = reproduction.get("seed_only_metric_delta")
            if llm_hash is not None and oracle_hash is not None:
                delta_text = f"{float(delta) * 100:+.3f} pp" if delta is not None else "N/A"
                print(
                    f"  seed {row['seed']}: hashes_match={llm_hash == oracle_hash}, "
                    f"metric_delta={delta_text}"
                )
        if deltas:
            print(
                f"  mean repeated-training metric delta: {mean(deltas) * 100:+.3f} pp, "
                f"max absolute delta: {max(abs(value) for value in deltas) * 100:.3f} pp"
            )


def _validate_formal_report(data: dict[str, Any], path: Path) -> None:
    """Refuse to present an incomplete formal run as a final result."""
    errors: list[str] = []
    if data.get("status") != "completed":
        errors.append(f"status={data.get('status')!r}, required 'completed'")
    if data.get("base_model_revision") != FORMAL_MODEL_REVISION:
        errors.append("base model revision does not match the frozen snapshot")
    configuration = data.get("configuration", {})
    if configuration.get("llm_model_revision") != FORMAL_MODEL_REVISION:
        errors.append("feedback model revision does not match the frozen snapshot")
    rows = data.get("results", [])
    if len(rows) != 30:
        errors.append(f"results={len(rows)}, required 30")
    required = {
        "split_unit": "prompt",
        "split_seed": 17,
        "fail_on_split_overlap": True,
        "candidate_budget": 32,
        "seed_group_limit": 64,
        "pool_group_limit": 256,
        "eval_group_limit": 512,
    }
    for key, expected in required.items():
        if configuration.get(key) != expected:
            errors.append(f"{key}={configuration.get(key)!r}, required {expected!r}")
    training = configuration.get("training_schedule", {})
    if training.get("budget_mode") != "equal-epochs":
        errors.append("training budget mode is not equal-epochs")
    observed = {(row.get("strategy"), int(row.get("seed"))) for row in rows}
    expected = {(strategy, seed) for strategy in FORMAL_STRATEGIES for seed in FORMAL_SEEDS}
    if observed != expected:
        errors.append("strategy/seed matrix is not the frozen 3 x 10 matrix")
    audits = configuration.get("split_audit_by_seed", {})
    if set(audits) != {str(seed) for seed in FORMAL_SEEDS}:
        errors.append("split audit does not cover all formal seeds")
    if any(audit.get("prompts_with_multiple_splits") != 0 for audit in audits.values()):
        errors.append("prompt split overlap detected in split audit")
    if any("training_records_seen" not in row for row in rows):
        errors.append("training_records_seen is missing from at least one result")
    if any(
        any(value is None for value in row.get("training_records_seen", {}).values())
        for row in rows
    ):
        errors.append("training_records_seen is incomplete in at least one result")
    if errors:
        raise ValueError(
            f"formal report {path} is not eligible for final analysis: " + "; ".join(errors)
        )


def _validate_qwen_diagnostic_report(data: dict[str, Any], path: Path) -> None:
    """Validate completed custom reports used by the reliable-selector study."""
    errors: list[str] = []
    configuration = data.get("configuration", {})
    strategies = tuple(configuration.get("strategies", ()))
    seeds = tuple(int(seed) for seed in configuration.get("seeds", ()))
    rows = data.get("results", [])

    if data.get("base_model_revision") != FORMAL_MODEL_REVISION:
        errors.append("base model revision does not match the frozen snapshot")
    if configuration.get("llm_model_revision") != FORMAL_MODEL_REVISION:
        errors.append("feedback model revision does not match the frozen snapshot")
    if not seeds or set(seeds) != FORMAL_SEEDS or len(seeds) != len(set(seeds)):
        errors.append("seeds must be the frozen 17..26 set")
    if not strategies or len(strategies) != len(set(strategies)):
        errors.append("strategies must be a non-empty unique sequence")
    if not set(strategies).issubset(QWEN_DIAGNOSTIC_STRATEGIES):
        errors.append("report contains an unsupported diagnostic strategy")
    required = {
        "split_unit": "prompt",
        "fail_on_split_overlap": True,
        "candidate_budget": 32,
        "seed_group_limit": 64,
        "pool_group_limit": 256,
        "eval_group_limit": 512,
    }
    for key, expected in required.items():
        if configuration.get(key) != expected:
            errors.append(f"{key}={configuration.get(key)!r}, required {expected!r}")
    if configuration.get("vary_split_by_seed"):
        errors.append("vary_split_by_seed must be false for paired confirmation runs")
    if configuration.get("training_schedule", {}).get("budget_mode") != "equal-epochs":
        errors.append("training budget mode is not equal-epochs")

    expected_keys = {(strategy, seed) for strategy in strategies for seed in seeds}
    observed_keys = {(row.get("strategy"), int(row.get("seed"))) for row in rows}
    if len(rows) != len(expected_keys) or observed_keys != expected_keys:
        errors.append("strategy/seed matrix is incomplete or contains duplicates")

    audits = configuration.get("split_audit_by_seed", {})
    if set(audits) != {str(seed) for seed in seeds}:
        errors.append("split audit does not cover every seed")
    if any(audit.get("prompts_with_multiple_splits") != 0 for audit in audits.values()):
        errors.append("prompt split overlap detected in split audit")

    for row in rows:
        strategy = row.get("strategy")
        selected_ids = row.get("selected_group_ids", [])
        if not isinstance(selected_ids, list) or len(selected_ids) != len(set(selected_ids)):
            errors.append(f"duplicate selected group ids for {strategy}/{row.get('seed')}")
        expected_selected = 0 if strategy == "seed-only" else required["candidate_budget"]
        if row.get("selected_groups") != expected_selected or len(selected_ids) != expected_selected:
            errors.append(f"selected-group budget mismatch for {strategy}/{row.get('seed')}")
        judgments = row.get("judgments", [])
        if len(judgments) != expected_selected:
            errors.append(f"judgment count mismatch for {strategy}/{row.get('seed')}")
        if expected_selected and {item.get("group_id") for item in judgments} != set(selected_ids):
            errors.append(f"judgments do not match selected groups for {strategy}/{row.get('seed')}")
        seen = row.get("training_records_seen")
        if not isinstance(seen, dict) or any(seen.get(channel) is None for channel in ("llm_feedback", "oracle_feedback")):
            errors.append(f"training_records_seen is incomplete for {strategy}/{row.get('seed')}")
        for channel in ("llm_feedback", "oracle_feedback"):
            metric = row.get("metrics", {}).get(channel, {})
            value = metric.get("pairwise_accuracy")
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"invalid {channel} metric for {strategy}/{row.get('seed')}")

    if errors:
        raise ValueError(
            f"Qwen diagnostic report {path} is not eligible for final analysis: "
            + "; ".join(sorted(set(errors)))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.reports:
        _report(path)


if __name__ == "__main__":
    main()
