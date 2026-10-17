"""Multi-round HelpSteer2 replay for the first MLE-Aladdin technique.

This keeps the original semi-synthetic feedback protocol but evaluates the
downstream task model after every feedback round.  It is a local validation of
the feedback-allocation mechanism, not a human study or a full RSI system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from core_system.mle_core.sklearn_components import TfidfLogisticTrainer
from local_experiments.aladdin.active_feedback_helpsteer2 import (
    PROFILES,
    STRATEGIES,
    _file_sha256,
    _labeled_records,
    _oracle_preference,
    _posterior_labels,
    _predict_groups,
    _query_group,
    _seed_attribute_models,
    _select_and_query,
)
from local_experiments.aladdin.helpsteer2_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    _pairwise_metrics,
    load_helpsteer2_groups,
)


PROTOCOL_VERSION = "active-feedback-helpsteer2-multiround-v1"


def _train_and_evaluate(
    seed_groups: list[Any],
    selected_groups: list[Any],
    selected_labels: dict[str, str],
    heldout_groups: list[Any],
    heldout_labels: dict[str, str],
    *,
    seed_labels: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    train_records = _labeled_records(seed_groups, seed_labels, "seed")
    train_records.extend(_labeled_records(selected_groups, selected_labels, "held_in"))
    checkpoint, _ = TfidfLogisticTrainer().train(
        train_records=train_records,
        initial_checkpoint=None,
        config={"ngram_range": (1, 2), "max_iter": 200},
        seed=seed,
        store=None,
    )
    metrics = _pairwise_metrics(
        checkpoint,
        _labeled_records(heldout_groups, heldout_labels, "held_out"),
    )
    return {
        "train_group_count": len(seed_groups) + len(selected_groups),
        "selected_group_count": len(selected_groups),
        "heldout_pairwise": metrics,
    }


def _run_one(
    *,
    seed_groups: list[Any],
    query_source: list[Any],
    selection_source: list[Any],
    heldout_groups: list[Any],
    models: dict[str, dict[str, Any]],
    target_profile_name: str,
    policy_seed: int,
    strategy: str,
    rounds: int,
    round_feedback_budget_seconds: float,
    round_max_queries: int,
    selection_budget: int,
    random_reserve_fraction: float,
    tie_margin: float,
    split_seed: int,
) -> dict[str, Any]:
    target_profile = PROFILES[target_profile_name]
    query_groups = [_query_group(group) for group in query_source]
    selection_groups = [_query_group(group) for group in selection_source]
    seed_query_groups = [_query_group(group) for group in seed_groups]
    all_prediction_groups = [*seed_query_groups, *query_groups, *selection_groups]
    predicted_labels, predicted_values = _predict_groups(
        all_prediction_groups, models, tie_margin=tie_margin
    )
    seed_labels = {
        group.group_id: _oracle_preference(group, target_profile, tie_margin=tie_margin)
        for group in seed_groups
    }
    posterior = {name: 1.0 / len(PROFILES) for name in PROFILES}
    for group in seed_query_groups:
        from local_experiments.aladdin.active_feedback_helpsteer2 import _update_posterior

        posterior = _update_posterior(
            group, seed_labels[group.group_id], predicted_labels, posterior
        )

    heldout_labels = {
        group.group_id: _oracle_preference(group, target_profile, tie_margin=tie_margin)
        for group in heldout_groups
    }
    query_by_id = {group.group_id: group for group in query_source}
    remaining = list(query_groups)
    round_rows = []
    cumulative_cost = 0.0
    cumulative_queries = 0
    selected_ids: set[str] = set()

    for round_index in range(rounds):
        if strategy == "seed-only":
            trace = []
            round_cost = 0.0
        else:
            selection = _select_and_query(
                remaining,
                selection_groups,
                predicted_labels,
                predicted_values,
                lambda group_id: _oracle_preference(
                    query_by_id[group_id], target_profile, tie_margin=tie_margin
                ),
                initial_posterior=posterior,
                strategy=strategy,
                time_budget_seconds=round_feedback_budget_seconds,
                max_queries=round_max_queries,
                selection_budget=selection_budget,
                random_reserve_fraction=random_reserve_fraction,
                seed=(split_seed * 1_000_003 + policy_seed * 101 + round_index),
            )
            trace = selection["trace"]
            posterior = selection["posterior"]
            round_cost = float(selection["total_cost_seconds"])
            selected_ids = set(selection["selected_data_ids"])
            queried_ids = {row["group_id"] for row in trace}
            remaining = [group for group in remaining if group.group_id not in queried_ids]

        selected_groups = [
            group for group in selection_source if group.group_id in selected_ids
        ]
        selected_labels = _posterior_labels(
            selected_groups, predicted_values, posterior, tie_margin=tie_margin
        )
        evaluation = _train_and_evaluate(
            seed_groups,
            selected_groups,
            selected_labels,
            heldout_groups,
            heldout_labels,
            seed_labels=seed_labels,
            seed=split_seed,
        )
        cumulative_cost += round_cost
        cumulative_queries += len(trace)
        round_rows.append({
            "round": round_index + 1,
            "feedback_cost_seconds": round_cost,
            "feedback_requests": len(trace),
            "cumulative_cost_seconds": cumulative_cost,
            "cumulative_feedback_requests": cumulative_queries,
            "posterior": posterior,
            "selected_data_ids": sorted(selected_ids),
            "query_trace": trace,
            **evaluation,
        })

    pool_ids = {
        "seed": sorted(group.group_id for group in seed_groups),
        "query": sorted(group.group_id for group in query_source),
        "selection": sorted(group.group_id for group in selection_source),
        "heldout": sorted(group.group_id for group in heldout_groups),
    }
    return {
        "strategy": strategy,
        "split_seed": split_seed,
        "policy_seed": policy_seed,
        "target_profile_simulator": target_profile_name,
        "rounds": round_rows,
        "feedback_cost_seconds": cumulative_cost,
        "feedback_requests": cumulative_queries,
        "selected_data_ids": sorted(selected_ids),
        "selected_data_ids_by_round": [
            row["selected_data_ids"] for row in round_rows
        ],
        "selected_data_count": len(selected_ids),
        "heldout_pairwise": round_rows[-1]["heldout_pairwise"],
        "pool_ids": pool_ids,
        "pool_overlap_counts": {
            f"{left}_{right}": len(
                set(pool_ids[left]) & set(pool_ids[right])
            )
            for index, left in enumerate(pool_ids)
            for right in list(pool_ids)[index + 1:]
        },
    }


def _paired_deltas(
    results: list[dict[str, Any]],
    *,
    primary: str,
    baseline: str,
    split_seeds: tuple[int, ...],
    policy_seeds: tuple[int, ...],
    target_profiles: tuple[str, ...],
) -> dict[str, Any]:
    split_deltas = []
    for split_seed in split_seeds:
        deltas = []
        for policy_seed in policy_seeds:
            for profile in target_profiles:
                def metric(strategy: str) -> float:
                    row = next(
                        item for item in results
                        if item["strategy"] == strategy
                        and item["split_seed"] == split_seed
                        and item["policy_seed"] == policy_seed
                        and item["target_profile_simulator"] == profile
                    )
                    return float(row["heldout_pairwise"]["pairwise_accuracy"])
                deltas.append(metric(primary) - metric(baseline))
        split_deltas.append(sum(deltas) / len(deltas))
    mean = sum(split_deltas) / len(split_deltas)
    import statistics
    standard_error = statistics.stdev(split_deltas) / len(split_deltas) ** 0.5 if len(split_deltas) > 1 else 0.0
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(split_deltas) - 1, 1.96)
    margin = critical * standard_error
    return {
        "primary": primary,
        "baseline": baseline,
        "unit": "split_seed (profile and policy-seed deltas averaged within split)",
        "n": len(split_deltas),
        "mean_pairwise_accuracy_delta": mean,
        "ci95_t": [mean - margin, mean + margin],
        "wins_ties_losses": [
            sum(value > 1e-12 for value in split_deltas),
            sum(abs(value) <= 1e-12 for value in split_deltas),
            sum(value < -1e-12 for value in split_deltas),
        ],
        "split_level_deltas": dict(zip(map(str, split_seeds), split_deltas, strict=True)),
    }


def run_multiround(
    data_path: Path,
    *,
    split_seeds: tuple[int, ...] = (17, 23, 29, 37),
    policy_seeds: tuple[int, ...] = (17, 18, 19),
    target_profiles: tuple[str, ...] = tuple(PROFILES),
    rounds: int = 3,
    round_feedback_budget_seconds: float = 600.0,
    round_max_queries: int = 16,
    selection_budget: int = 64,
    random_reserve_fraction: float = 0.25,
    seed_group_limit: int = 64,
    query_group_limit: int = 256,
    selection_group_limit: int = 512,
    heldout_group_limit: int = 512,
    tie_margin: float = 0.0,
) -> dict[str, Any]:
    if rounds <= 0 or round_feedback_budget_seconds <= 0 or round_max_queries <= 0:
        raise ValueError("round configuration must be positive")
    groups = load_helpsteer2_groups(data_path)
    seed_source = [g for g in groups if g.split == "seed" and len(g.hidden_records) == 2]
    pool_source = [g for g in groups if g.split == "held_in" and len(g.hidden_records) == 2]
    heldout_source = [g for g in groups if g.split == "held_out" and len(g.hidden_records) == 2]
    if len(seed_source) < seed_group_limit:
        raise ValueError("insufficient seed groups")
    if len(pool_source) < query_group_limit + selection_group_limit:
        raise ValueError("insufficient held_in groups")
    if len(heldout_source) < heldout_group_limit:
        raise ValueError("insufficient held_out groups")

    results = []
    started = time.monotonic()
    total = len(split_seeds) * len(policy_seeds) * len(target_profiles) * len(STRATEGIES)
    completed = 0
    for split_seed in split_seeds:
        seed_rng = random.Random(split_seed * 1009 + 17)
        seed_groups = seed_rng.sample(sorted(seed_source, key=lambda g: g.group_id), seed_group_limit)
        pool_rng = random.Random(split_seed * 1013 + 23)
        heldin = pool_rng.sample(sorted(pool_source, key=lambda g: g.group_id), query_group_limit + selection_group_limit)
        query_source, selection_source = heldin[:query_group_limit], heldin[query_group_limit:]
        heldout_rng = random.Random(split_seed * 1019 + 31)
        heldout_groups = heldout_rng.sample(sorted(heldout_source, key=lambda g: g.group_id), heldout_group_limit)
        models = _seed_attribute_models(seed_groups, split_seed)
        for profile in target_profiles:
            for policy_seed in policy_seeds:
                for strategy in STRATEGIES:
                    row = _run_one(
                        seed_groups=seed_groups,
                        query_source=query_source,
                        selection_source=selection_source,
                        heldout_groups=heldout_groups,
                        models=models,
                        target_profile_name=profile,
                        policy_seed=policy_seed,
                        strategy=strategy,
                        rounds=rounds,
                        round_feedback_budget_seconds=round_feedback_budget_seconds,
                        round_max_queries=round_max_queries,
                        selection_budget=selection_budget,
                        random_reserve_fraction=random_reserve_fraction,
                        tie_margin=tie_margin,
                        split_seed=split_seed,
                    )
                    results.append(row)
                    completed += 1
                    elapsed = time.monotonic() - started
                    eta = elapsed / completed * (total - completed)
                    print(
                        f"[{completed}/{total}] split={split_seed} policy={policy_seed} "
                        f"profile={profile} strategy={strategy} "
                        f"rounds={rounds} final_pairwise={row['heldout_pairwise']['pairwise_accuracy']:.4f} "
                        f"queries={row['feedback_requests']} eta={eta:.1f}s",
                        flush=True,
                    )

    summary = {}
    for strategy in STRATEGIES:
        rows = [row for row in results if row["strategy"] == strategy]
        summary[strategy] = {
            "mean_final_pairwise_accuracy": sum(row["heldout_pairwise"]["pairwise_accuracy"] for row in rows) / len(rows),
            "mean_feedback_requests": sum(row["feedback_requests"] for row in rows) / len(rows),
            "mean_feedback_cost_seconds": sum(row["feedback_cost_seconds"] for row in rows) / len(rows),
            "mean_selected_data_count": sum(row["selected_data_count"] for row in rows) / len(rows),
            "mean_pairwise_by_round": [
                sum(row["rounds"][round_index]["heldout_pairwise"]["pairwise_accuracy"] for row in rows) / len(rows)
                for round_index in range(rounds)
            ],
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_sha256": _file_sha256(data_path),
        "configuration": {
            "split_seeds": list(split_seeds),
            "policy_seeds": list(policy_seeds),
            "target_profiles": list(target_profiles),
            "strategies": list(STRATEGIES),
            "rounds": rounds,
            "round_feedback_budget_seconds": round_feedback_budget_seconds,
            "total_feedback_budget_seconds": rounds * round_feedback_budget_seconds,
            "round_max_queries": round_max_queries,
            "selection_budget": selection_budget,
            "random_reserve_fraction": random_reserve_fraction,
            "seed_group_limit": seed_group_limit,
            "query_group_limit": query_group_limit,
            "selection_group_limit": selection_group_limit,
            "heldout_group_limit": heldout_group_limit,
            "selection_recomputed_after_each_round": True,
            "heldout_labels_used_for_selection": False,
            "final_test_loaded": False,
        },
        "summary": summary,
        "paired_comparisons_primary_vs_baseline": {
            baseline: _paired_deltas(
                results,
                primary="impact-aware-random",
                baseline=baseline,
                split_seeds=split_seeds,
                policy_seeds=policy_seeds,
                target_profiles=target_profiles,
            )
            for baseline in ("random", "disagreement-only", "seed-only")
        },
        "pool_overlap_counts_all_zero": all(
            len(set(row["pool_ids"][left]) & set(row["pool_ids"][right])) == 0
            for row in results
            for left, right in (("seed", "query"), ("seed", "selection"), ("seed", "heldout"), ("query", "selection"), ("query", "heldout"), ("selection", "heldout"))
        ),
        "final_test_accessed": False,
        "results": results,
        "limitations": [
            "Historical HelpSteer2 scores simulate feedback; no new human annotations were collected.",
            "The feedback-time cost remains an uncalibrated proxy.",
            "TF-IDF logistic regression is a downstream task-model proxy, not a production generation model.",
            "This validates multi-round feedback allocation, not data synthesis, augmentation, or a complete RSI loop.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/helpsteer2-train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[17, 23, 29, 37])
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--target-profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-feedback-budget-seconds", type=float, default=600.0)
    parser.add_argument("--round-max-queries", type=int, default=16)
    parser.add_argument("--selection-budget", type=int, default=64)
    parser.add_argument("--random-reserve-fraction", type=float, default=0.25)
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--query-group-limit", type=int, default=256)
    parser.add_argument("--selection-group-limit", type=int, default=512)
    parser.add_argument("--heldout-group-limit", type=int, default=512)
    parser.add_argument("--tie-margin", type=float, default=0.0)
    args = parser.parse_args()
    report = run_multiround(
        args.data,
        split_seeds=tuple(args.split_seeds),
        policy_seeds=tuple(args.policy_seeds),
        target_profiles=tuple(args.target_profiles),
        rounds=args.rounds,
        round_feedback_budget_seconds=args.round_feedback_budget_seconds,
        round_max_queries=args.round_max_queries,
        selection_budget=args.selection_budget,
        random_reserve_fraction=args.random_reserve_fraction,
        seed_group_limit=args.seed_group_limit,
        query_group_limit=args.query_group_limit,
        selection_group_limit=args.selection_group_limit,
        heldout_group_limit=args.heldout_group_limit,
        tie_margin=args.tie_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
