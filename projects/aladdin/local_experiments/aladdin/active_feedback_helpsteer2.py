"""Semi-synthetic HelpSteer2 replay for impact-aware feedback selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core_system.mle_core.models import TaskRecord
from core_system.mle_core.sklearn_components import (
    TfidfLogisticTrainer,
    _checkpoint_scores,
)
from local_experiments.aladdin.helpsteer2_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    HelpSteer2Group,
    _pairwise_metrics,
    load_helpsteer2_groups,
)


PROTOCOL_VERSION = "active-feedback-helpsteer2-local-v2"
SCORE_FIELDS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
STRATEGIES = ("seed-only", "random", "disagreement-only", "impact-aware", "impact-aware-random")
MISMATCH_LIKELIHOOD = 0.05
READING_WORDS_PER_MINUTE = 220.0
DECISION_SECONDS = 5.0


@dataclass(frozen=True)
class RequirementProfile:
    name: str
    description: str
    weights: tuple[float, ...]


@dataclass(frozen=True)
class QueryGroup:
    group_id: str
    prompt: str
    records: tuple[TaskRecord, ...]
    estimated_cost_seconds: float


PROFILES = {
    "helpfulness-only": RequirementProfile(
        "helpfulness-only",
        "Prioritize the HelpSteer helpfulness rating as the preference criterion.",
        (1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    "correctness-only": RequirementProfile(
        "correctness-only",
        "Prioritize the HelpSteer correctness rating as the preference criterion.",
        (0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    "coherence-only": RequirementProfile(
        "coherence-only",
        "Prioritize the HelpSteer coherence rating as the preference criterion.",
        (0.0, 0.0, 1.0, 0.0, 0.0),
    ),
}


def _softmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values())
    exponentials = {key: math.exp(value - maximum) for key, value in values.items()}
    total = sum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


def _cost_seconds(group: HelpSteer2Group) -> float:
    words = sum(len(str(record.input).split()) for record in group.hidden_records)
    return DECISION_SECONDS + words / READING_WORDS_PER_MINUTE * 60.0


def _query_group(group: HelpSteer2Group) -> QueryGroup:
    return QueryGroup(group.group_id, group.prompt, group.hidden_records, _cost_seconds(group))


def _seed_attribute_models(seed_groups: list[HelpSteer2Group], seed: int) -> dict[str, dict[str, Any]]:
    models = {}
    for field in SCORE_FIELDS:
        records = []
        for group in seed_groups:
            for record in group.released_records:
                score = record.metadata.get("scores", {}).get(field)
                if score is None:
                    continue
                records.append(
                    TaskRecord(
                        record_id=f"{record.record_id}-{field}",
                        task_id=f"helpsteer2-{field}-attribute",
                        split="seed",
                        input=record.input,
                        target=str(int(score)),
                        source_group_id=record.source_group_id,
                    )
                )
        if len({record.target for record in records}) < 2:
            raise ValueError(f"seed data does not contain two labels for attribute {field}")
        models[field], _ = TfidfLogisticTrainer().train(
            train_records=records,
            initial_checkpoint=None,
            config={"ngram_range": (1, 2), "max_iter": 200},
            seed=seed,
            store=None,
        )
    return models


def _predicted_attribute_values(
    records: tuple[TaskRecord, ...], models: dict[str, dict[str, Any]]
) -> list[dict[str, float]]:
    predictions = []
    for record in records:
        values = {}
        for field, model in models.items():
            probabilities = _softmax(_checkpoint_scores(model, str(record.input)))
            values[field] = sum(float(label) * probability for label, probability in probabilities.items())
        predictions.append(values)
    return predictions


def _utility(values: dict[str, float], profile: RequirementProfile) -> float:
    return sum(weight * values[field] for field, weight in zip(SCORE_FIELDS, profile.weights, strict=True))


def _predict_groups(
    groups: list[QueryGroup],
    models: dict[str, dict[str, Any]],
    *,
    tie_margin: float,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, tuple[float, float]]]]:
    labels: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, tuple[float, float]]] = {}
    for group in groups:
        predictions = _predicted_attribute_values(group.records, models)
        if len(predictions) != 2:
            raise ValueError(f"expected pairwise group, got {len(predictions)} records for {group.group_id}")
        labels[group.group_id] = {}
        values[group.group_id] = {}
        for name, profile in PROFILES.items():
            difference = _utility(predictions[0], profile) - _utility(predictions[1], profile)
            labels[group.group_id][name] = (
                "preferred" if difference > tie_margin
                else "rejected" if difference < -tie_margin
                else "tie"
            )
            values[group.group_id][name] = (
                _utility(predictions[0], profile),
                _utility(predictions[1], profile),
            )
    return labels, values


def _oracle_preference(group: HelpSteer2Group, profile: RequirementProfile, *, tie_margin: float) -> str:
    if len(group.released_records) != 2:
        raise ValueError(f"expected pairwise group, got {len(group.released_records)} records for {group.group_id}")
    scores = [record.metadata["scores"] for record in group.released_records]
    difference = _utility({field: float(scores[0][field]) for field in SCORE_FIELDS}, profile)
    difference -= _utility({field: float(scores[1][field]) for field in SCORE_FIELDS}, profile)
    if difference > tie_margin:
        return "preferred"
    if difference < -tie_margin:
        return "rejected"
    return "tie"


def _label_distribution(
    group: QueryGroup,
    predicted_labels: dict[str, dict[str, str]],
    posterior: dict[str, float],
) -> dict[str, float]:
    distribution = {label: 0.0 for label in ("preferred", "rejected", "tie")}
    for name in PROFILES:
        label = predicted_labels[group.group_id][name]
        distribution[label] += posterior.get(name, 0.0)
    return distribution


def _ranked_selection(
    candidates: list[QueryGroup],
    predicted_values: dict[str, dict[str, tuple[float, float]]],
    posterior: dict[str, float],
    selection_budget: int,
) -> set[str]:
    ranked = []
    for group in candidates:
        expected_best_response = max(
            sum(
                posterior.get(name, 0.0) * predicted_values[group.group_id][name][response_index]
                for name in PROFILES
            )
            for response_index in (0, 1)
        )
        ranked.append((expected_best_response, group.group_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {group_id for _, group_id in ranked[:selection_budget]}


def _query_impact(
    group: QueryGroup,
    selection_pool: list[QueryGroup],
    predicted_labels: dict[str, dict[str, str]],
    predicted_values: dict[str, dict[str, tuple[float, float]]],
    posterior: dict[str, float],
    *,
    selection_budget: int,
) -> dict[str, float | int]:
    distribution = _label_distribution(group, predicted_labels, posterior)
    disagreement = 1.5 * (1.0 - sum(probability**2 for probability in distribution.values()))
    current = _ranked_selection(selection_pool, predicted_values, posterior, selection_budget)
    expected_change = 0.0
    max_change = 0
    for label, probability in distribution.items():
        if probability <= 0:
            continue
        filtered = {
            name: weight
            for name, weight in posterior.items()
            if predicted_labels[group.group_id][name] == label
        }
        total = sum(filtered.values())
        if total <= 0:
            continue
        counterfactual = _ranked_selection(
            selection_pool,
            predicted_values,
            {name: weight / total for name, weight in filtered.items()},
            selection_budget,
        )
        change = len(current.symmetric_difference(counterfactual))
        expected_change += probability * change
        max_change = max(max_change, change)
    return {
        "disagreement": disagreement,
        "expected_selection_change": expected_change,
        "affected_selection_count": max_change,
        "impact_value": disagreement * expected_change,
    }


def _update_posterior(
    group: QueryGroup,
    observed: str,
    predicted_labels: dict[str, dict[str, str]],
    posterior: dict[str, float],
) -> dict[str, float]:
    updated = {}
    for name, weight in posterior.items():
        predicted = predicted_labels[group.group_id][name]
        updated[name] = weight * (1.0 if predicted == observed else MISMATCH_LIKELIHOOD)
    total = sum(updated.values())
    if total <= 0:
        return {name: 1.0 / len(updated) for name in updated}
    return {name: weight / total for name, weight in updated.items()}


def _select_and_query(
    query_pool: list[QueryGroup],
    selection_pool: list[QueryGroup],
    predicted_labels: dict[str, dict[str, str]],
    predicted_values: dict[str, dict[str, tuple[float, float]]],
    feedback_provider: Callable[[str], str],
    *,
    initial_posterior: dict[str, float],
    strategy: str,
    time_budget_seconds: float,
    max_queries: int,
    selection_budget: int,
    random_reserve_fraction: float,
    seed: int,
) -> dict[str, Any]:
    posterior = dict(initial_posterior)
    remaining = {group.group_id: group for group in query_pool}
    rng = random.Random(seed)
    reserve_budget = time_budget_seconds * random_reserve_fraction
    reserve_spent = 0.0
    spent = 0.0
    trace = []
    while len(trace) < max_queries:
        feasible = [group for group in remaining.values() if spent + group.estimated_cost_seconds <= time_budget_seconds]
        if not feasible:
            break
        random_pick = strategy == "random" or (
            strategy == "impact-aware-random" and reserve_spent < reserve_budget
        )
        if random_pick:
            selected = rng.choice(sorted(feasible, key=lambda group: group.group_id))
            diagnostics = {"disagreement": None, "expected_selection_change": None,
                           "affected_selection_count": None, "impact_value": None,
                           "selection_score": None, "random_reserve": strategy == "impact-aware-random"}
        else:
            scored = []
            for candidate in feasible:
                info = _query_impact(
                    candidate, selection_pool, predicted_labels, predicted_values, posterior,
                    selection_budget=selection_budget,
                )
                if strategy == "disagreement-only":
                    score = float(info["disagreement"]) / candidate.estimated_cost_seconds
                else:
                    score = float(info["impact_value"]) / candidate.estimated_cost_seconds
                scored.append((score, candidate.group_id, candidate, info))
            score, _, selected, diagnostics = max(scored, key=lambda row: (row[0], row[1]))
            diagnostics = {**diagnostics, "selection_score": score, "random_reserve": False}

        observed = feedback_provider(selected.group_id)
        posterior = _update_posterior(
            selected, observed, predicted_labels, posterior
        )
        spent += selected.estimated_cost_seconds
        if random_pick and strategy == "impact-aware-random":
            reserve_spent += selected.estimated_cost_seconds
        remaining.pop(selected.group_id)
        trace.append({
            "group_id": selected.group_id,
            "estimated_cost_seconds": selected.estimated_cost_seconds,
            "cumulative_cost_seconds": spent,
            "observed_feedback": observed,
            "posterior": posterior,
            **diagnostics,
        })
    selected_ids = _ranked_selection(selection_pool, predicted_values, posterior, selection_budget)
    return {
        "trace": trace,
        "posterior": posterior,
        "total_cost_seconds": spent,
        "selected_data_ids": sorted(selected_ids),
    }


def _labeled_records(
    groups: list[HelpSteer2Group],
    labels: dict[str, str],
    split: str,
) -> list[TaskRecord]:
    records = []
    for group in groups:
        label = labels[group.group_id]
        if label == "tie":
            pair_labels = ("tie", "tie")
        elif label == "preferred":
            pair_labels = ("preferred", "rejected")
        else:
            pair_labels = ("rejected", "preferred")
        for record, target in zip(group.hidden_records, pair_labels, strict=True):
            records.append(TaskRecord(
                record_id=record.record_id,
                task_id="helpsteer2-profile-preference",
                split=split,
                input=record.input,
                target=target,
                source_group_id=record.source_group_id,
                metadata={"feedback_released": True, "simulated_profile_label": True},
            ))
    return records


def _posterior_labels(
    groups: list[HelpSteer2Group],
    predicted_values: dict[str, dict[str, tuple[float, float]]],
    posterior: dict[str, float],
    *,
    tie_margin: float,
) -> dict[str, str]:
    labels = {}
    for group in groups:
        first = sum(
            posterior.get(name, 0.0) * predicted_values[group.group_id][name][0]
            for name in PROFILES
        )
        second = sum(
            posterior.get(name, 0.0) * predicted_values[group.group_id][name][1]
            for name in PROFILES
        )
        labels[group.group_id] = (
            "preferred" if first - second > tie_margin
            else "rejected" if second - first > tie_margin
            else "tie"
        )
    return labels


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_comparison(
    results: list[dict[str, Any]],
    *,
    primary: str,
    baseline: str,
    target_profiles: tuple[str, ...],
    split_seeds: tuple[int, ...],
    policy_seeds: tuple[int, ...],
) -> dict[str, Any]:
    # Average profile conditions within each split before inference so that
    # multiple simulated preference profiles do not inflate the sample size.
    split_deltas = []
    for split_seed in split_seeds:
        deltas = []
        for policy_seed in policy_seeds:
            for profile in target_profiles:
                def metric(strategy: str) -> float:
                    return next(
                        row["heldout_pairwise"]["pairwise_accuracy"]
                        for row in results
                        if row["strategy"] == strategy
                        and row["split_seed"] == split_seed
                        and row["policy_seed"] == policy_seed
                        and row["target_profile_simulator"] == profile
                    )
                deltas.append(metric(primary) - metric(baseline))
        split_deltas.append(sum(deltas) / len(deltas))

    n = len(split_deltas)
    average = sum(split_deltas) / n
    standard_error = statistics.stdev(split_deltas) / math.sqrt(n) if n > 1 else 0.0
    t_critical_95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }.get(n - 1, 1.96)
    margin = t_critical_95 * standard_error
    wins = sum(value > 1e-12 for value in split_deltas)
    losses = sum(value < -1e-12 for value in split_deltas)
    return {
        "primary": primary,
        "baseline": baseline,
        "unit": "split_seed (profile and policy-seed deltas averaged within split)",
        "n": n,
        "mean_pairwise_accuracy_delta": average,
        "ci95_t": [average - margin, average + margin],
        "wins_ties_losses": [wins, n - wins - losses, losses],
        "split_level_deltas": dict(zip(map(str, split_seeds), split_deltas, strict=True)),
    }


def run_experiment(
    data_path: Path,
    *,
    seeds: tuple[int, ...] | None = None,
    split_seeds: tuple[int, ...] = (17, 23, 29, 37),
    policy_seeds: tuple[int, ...] = (17, 18, 19),
    target_profiles: tuple[str, ...] = tuple(PROFILES),
    seed_group_limit: int = 64,
    query_group_limit: int = 256,
    selection_group_limit: int = 512,
    heldout_group_limit: int = 512,
    feedback_time_budget_seconds: float = 1800.0,
    max_queries: int = 48,
    selection_budget: int = 64,
    random_reserve_fraction: float = 0.25,
    tie_margin: float = 0.0,
) -> dict[str, Any]:
    if seeds is not None:
        if split_seeds != (17, 23, 29, 37) or policy_seeds != (17, 18, 19):
            raise ValueError("use either legacy seeds or split_seeds/policy_seeds, not both")
        split_seeds = tuple(seeds)
        policy_seeds = tuple(seeds)
    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds must be non-empty and unique")
    if not policy_seeds or len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("policy_seeds must be non-empty and unique")
    if not target_profiles or len(set(target_profiles)) != len(target_profiles):
        raise ValueError("target_profiles must be non-empty and unique")
    unknown_profiles = sorted(set(target_profiles) - set(PROFILES))
    if unknown_profiles:
        raise ValueError(f"unknown target profiles: {unknown_profiles}")
    if min(seed_group_limit, query_group_limit, selection_group_limit, heldout_group_limit, max_queries, selection_budget) <= 0:
        raise ValueError("all group and query limits must be positive")
    if query_group_limit <= max_queries:
        raise ValueError("query_group_limit must exceed max_queries to preserve selection pressure")
    if selection_group_limit < selection_budget:
        raise ValueError("selection_group_limit must be at least selection_budget")
    if feedback_time_budget_seconds <= 0 or not 0.0 <= random_reserve_fraction < 1.0:
        raise ValueError("invalid feedback budget or random reserve")
    source_groups = load_helpsteer2_groups(data_path)
    seed_source = [group for group in source_groups if group.split == "seed" and len(group.hidden_records) == 2]
    pool_source = [group for group in source_groups if group.split == "held_in" and len(group.hidden_records) == 2]
    heldout_source = [group for group in source_groups if group.split == "held_out" and len(group.hidden_records) == 2]
    if len(seed_source) < seed_group_limit or len(pool_source) < query_group_limit + selection_group_limit:
        raise ValueError("HelpSteer2 train split has insufficient pair groups for requested protocol")
    if len(heldout_source) < heldout_group_limit:
        raise ValueError("HelpSteer2 train held_out split has insufficient pair groups")

    results = []
    total_runs = len(split_seeds) * len(policy_seeds) * len(target_profiles) * len(STRATEGIES)
    completed = 0
    started = time.monotonic()
    for split_seed in split_seeds:
        seed_rng = random.Random(split_seed * 1009 + 17)
        seed_groups = seed_rng.sample(sorted(seed_source, key=lambda group: group.group_id), seed_group_limit)
        pool_rng = random.Random(split_seed * 1013 + 23)
        heldin = pool_rng.sample(sorted(pool_source, key=lambda group: group.group_id), query_group_limit + selection_group_limit)
        query_source = heldin[:query_group_limit]
        selection_source = heldin[query_group_limit:]
        heldout_rng = random.Random(split_seed * 1019 + 31)
        heldout_groups = heldout_rng.sample(sorted(heldout_source, key=lambda group: group.group_id), heldout_group_limit)
        models = _seed_attribute_models(seed_groups, split_seed)
        query_groups = [_query_group(group) for group in query_source]
        selection_groups = [_query_group(group) for group in selection_source]
        seed_query_groups = [_query_group(group) for group in seed_groups]
        predicted_labels, predicted_values = _predict_groups(
            [*seed_query_groups, *query_groups, *selection_groups], models, tie_margin=tie_margin
        )
        for target_profile_name in target_profiles:
            target_profile = PROFILES[target_profile_name]
            seed_labels = {
                group.group_id: _oracle_preference(group, target_profile, tie_margin=tie_margin)
                for group in seed_groups
            }
            initial_posterior = {name: 1.0 / len(PROFILES) for name in PROFILES}
            for group in seed_query_groups:
                initial_posterior = _update_posterior(
                    group,
                    seed_labels[group.group_id],
                    predicted_labels,
                    initial_posterior,
                )
            for policy_seed in policy_seeds:
                for strategy in STRATEGIES:
                    if strategy == "seed-only":
                        selected_data_ids = set()
                        trace = []
                        posterior = dict(initial_posterior)
                        total_cost = 0.0
                    else:
                        query_group_by_id = {group.group_id: group for group in query_source}
                        selection = _select_and_query(
                            query_groups, selection_groups, predicted_labels, predicted_values,
                            lambda group_id: _oracle_preference(
                                query_group_by_id[group_id],
                                target_profile,
                                tie_margin=tie_margin,
                            ),
                            initial_posterior=initial_posterior,
                            strategy=strategy,
                            time_budget_seconds=feedback_time_budget_seconds,
                            max_queries=max_queries,
                            selection_budget=selection_budget,
                            random_reserve_fraction=random_reserve_fraction,
                            seed=split_seed * 1_000_003 + policy_seed * 101 + tuple(PROFILES).index(target_profile_name),
                        )
                        selected_data_ids = set(selection["selected_data_ids"])
                        trace = selection["trace"]
                        posterior = selection["posterior"]
                        total_cost = selection["total_cost_seconds"]

                    selected_groups = [group for group in selection_source if group.group_id in selected_data_ids]
                    selected_labels = _posterior_labels(
                        selected_groups, predicted_values, posterior, tie_margin=tie_margin
                    )
                    train_records = _labeled_records(seed_groups, seed_labels, "seed")
                    train_records.extend(_labeled_records(selected_groups, selected_labels, "held_in"))
                    checkpoint, _ = TfidfLogisticTrainer().train(
                        train_records=train_records,
                        initial_checkpoint=None,
                        config={"ngram_range": (1, 2), "max_iter": 200},
                        seed=split_seed,
                        store=None,
                    )
                    heldout_labels = {
                        group.group_id: _oracle_preference(group, target_profile, tie_margin=tie_margin)
                        for group in heldout_groups
                    }
                    heldout_records = _labeled_records(heldout_groups, heldout_labels, "held_out")
                    metrics = _pairwise_metrics(checkpoint, heldout_records)
                    pool_ids = {
                        "seed": {group.group_id for group in seed_groups},
                        "query": {group.group_id for group in query_source},
                        "selection": {group.group_id for group in selection_source},
                        "heldout": {group.group_id for group in heldout_groups},
                    }
                    overlaps = {
                        f"{left}_{right}": len(pool_ids[left] & pool_ids[right])
                        for left_index, left in enumerate(pool_ids)
                        for right in list(pool_ids)[left_index + 1:]
                    }
                    results.append({
                        "strategy": strategy,
                        "seed": split_seed,
                        "split_seed": split_seed,
                        "policy_seed": policy_seed,
                        "target_profile_simulator": target_profile_name,
                        "feedback_cost_seconds": total_cost,
                        "feedback_budget_seconds": feedback_time_budget_seconds,
                        "feedback_budget_utilization": total_cost / feedback_time_budget_seconds,
                        "feedback_requests": len(trace),
                        "feedback_records_released": sum(len(group.hidden_records) for group in query_source if group.group_id in {row["group_id"] for row in trace}),
                        "query_count": len(trace),
                        "query_trace": trace,
                        "initial_posterior": initial_posterior,
                        "posterior": posterior,
                        "selected_data_ids": sorted(selected_data_ids),
                        "selected_data_count": len(selected_data_ids),
                        "heldout_pairwise": metrics,
                        "pool_counts": {name: len(ids) for name, ids in pool_ids.items()},
                        "pool_overlap_counts": overlaps,
                        "pool_ids": {name: sorted(ids) for name, ids in pool_ids.items()},
                    })
                    completed += 1
                    elapsed = time.monotonic() - started
                    eta = elapsed / completed * (total_runs - completed)
                    print(
                        f"[{completed}/{total_runs}] split={split_seed} policy={policy_seed} "
                        f"profile={target_profile_name} strategy={strategy} "
                        f"pairwise={metrics['pairwise_accuracy']:.4f} "
                        f"queries={len(trace)} cost={total_cost:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )

    means = {}
    for strategy in STRATEGIES:
        rows = [row for row in results if row["strategy"] == strategy]
        means[strategy] = {
            "mean_heldout_pairwise_accuracy": sum(row["heldout_pairwise"]["pairwise_accuracy"] for row in rows) / len(rows),
            "mean_feedback_cost_seconds": sum(row["feedback_cost_seconds"] for row in rows) / len(rows),
            "mean_feedback_budget_utilization": sum(row["feedback_budget_utilization"] for row in rows) / len(rows),
            "mean_feedback_requests": sum(row["feedback_requests"] for row in rows) / len(rows),
            "mean_selected_data_count": sum(row["selected_data_count"] for row in rows) / len(rows),
        }
    comparisons = {
        baseline: _paired_comparison(
            results,
            primary="impact-aware-random",
            baseline=baseline,
            target_profiles=target_profiles,
            split_seeds=split_seeds,
            policy_seeds=policy_seeds,
        )
        for baseline in ("random", "disagreement-only", "seed-only")
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
            "seeds": list(split_seeds),
            "target_profiles": list(target_profiles),
            "strategies": list(STRATEGIES),
            "seed_group_limit": seed_group_limit,
            "query_group_limit": query_group_limit,
            "selection_group_limit": selection_group_limit,
            "heldout_group_limit": heldout_group_limit,
            "feedback_time_budget_seconds": feedback_time_budget_seconds,
            "max_queries": max_queries,
            "selection_budget": selection_budget,
            "random_reserve_fraction": random_reserve_fraction,
            "tie_margin": tie_margin,
            "simulated_human_time_cost": "5 seconds + pair input word count / 220 words per minute; proxy only, not measured annotation time",
            "requirement_profiles": {name: {"description": profile.description, "weights": dict(zip(SCORE_FIELDS, profile.weights, strict=True))} for name, profile in PROFILES.items()},
            "target_profile_assignment": "complete seed by target-profile factorial; target profile is used only by the historical feedback replay, seed supervision, and held-out evaluator",
            "seed_feedback_initializes_posterior": True,
            "selected_data_labels": "derived from the updated requirement posterior and seed-trained attribute predictors; hidden selection-pool human scores are not released",
            "selection_budget_is_identical_across_query_strategies": True,
            "feedback_time_budget_is_identical_across_strategies": True,
            "selection_uses_query_labels_only_after_query": True,
            "final_test_loaded": False,
            "heldout_labels_used_for_selection": False,
        },
        "summary": means,
        "paired_comparisons_primary_vs_baseline": comparisons,
        "label_leakage_audit": {
            "query_group_targets_hidden_before_selection": True,
            "selection_pool_targets_never_released": True,
            "heldout_labels_used_only_for_final_metric": True,
            "final_test_rows_loaded": 0,
            "pool_overlap_counts_all_zero": all(
                value == 0
                for row in results
                for value in row["pool_overlap_counts"].values()
            ),
        },
        "final_test_accessed": False,
        "results": results,
        "limitations": [
            "This is a semi-synthetic replay: HelpSteer2 human attribute scores simulate a fixed target requirement profile; no new human annotations were collected.",
            "The 220 words/minute cost model is an uncalibrated annotation-time proxy; equal simulated time budget is not equal measured human effort.",
            "Candidate requirement profiles are hand-specified and must be frozen before any confirmatory human study.",
            "The local Task Model is TF-IDF logistic regression; server GPU models and human-facing query UX are not evaluated here.",
            "The HelpSteer2 validation/final_test split is not loaded.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/helpsteer2-train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("local_experiments/aladdin/runs/technology-one-helpsteer2.json"))
    parser.add_argument("--seeds", type=int, nargs="+", help="legacy shorthand; uses the same values for split and policy seeds")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[17, 23, 29, 37])
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--target-profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--query-group-limit", type=int, default=256)
    parser.add_argument("--selection-group-limit", type=int, default=512)
    parser.add_argument("--heldout-group-limit", type=int, default=512)
    parser.add_argument("--feedback-time-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--max-queries", type=int, default=48)
    parser.add_argument("--selection-budget", type=int, default=64)
    parser.add_argument("--random-reserve-fraction", type=float, default=0.25)
    parser.add_argument("--tie-margin", type=float, default=0.0)
    args = parser.parse_args()
    seed_options = (
        {"seeds": tuple(args.seeds)}
        if args.seeds is not None
        else {
            "split_seeds": tuple(args.split_seeds),
            "policy_seeds": tuple(args.policy_seeds),
        }
    )
    report = run_experiment(
        args.data,
        **seed_options,
        target_profiles=tuple(args.target_profiles),
        seed_group_limit=args.seed_group_limit,
        query_group_limit=args.query_group_limit,
        selection_group_limit=args.selection_group_limit,
        heldout_group_limit=args.heldout_group_limit,
        feedback_time_budget_seconds=args.feedback_time_budget_seconds,
        max_queries=args.max_queries,
        selection_budget=args.selection_budget,
        random_reserve_fraction=args.random_reserve_fraction,
        tie_margin=args.tie_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
