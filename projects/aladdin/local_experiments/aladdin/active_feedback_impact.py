"""Local prototype for impact-aware active feedback selection.

The prototype is deliberately model-light. It validates the experimental
protocol for technique one before introducing a real LLM or human interface:
multiple requirement hypotheses disagree about candidates, a query can change
the next selected set, query cost is budgeted, and feedback updates the
hypothesis posterior across rounds.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROTOCOL_VERSION = "active-feedback-impact-local-v2"
STRATEGIES = (
    "random",
    "disagreement-only",
    "impact-aware",
    "impact-aware-random",
)


@dataclass(frozen=True)
class Requirement:
    name: str
    weights: tuple[float, ...]
    bias: float

    def score(self, features: tuple[float, ...]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features)) - self.bias

    def label(self, features: tuple[float, ...]) -> int:
        return int(self.score(features) >= 0.0)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    features: tuple[float, ...]
    true_label: int
    query_cost: float


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {key: 1.0 / len(weights) for key in weights}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def predict_label(requirement: Requirement, candidate: Candidate) -> int:
    return requirement.label(candidate.features)


def update_posterior(
    requirements: dict[str, Requirement],
    posterior: dict[str, float],
    candidate: Candidate,
    observed_label: int,
    *,
    mismatch_likelihood: float = 0.05,
) -> dict[str, float]:
    """Update only from the observed feedback label, never from true_label."""
    if observed_label not in (0, 1):
        raise ValueError("observed_label must be binary")
    if not 0.0 < mismatch_likelihood <= 1.0:
        raise ValueError("mismatch_likelihood must be in (0, 1]")
    updated = {
        name: weight
        * (1.0 if predict_label(requirement, candidate) == observed_label else mismatch_likelihood)
        for name, weight in posterior.items()
        for requirement in [requirements[name]]
    }
    return _normalise(updated)


def _selected_ids(
    candidates: list[Candidate],
    requirements: dict[str, Requirement],
    posterior: dict[str, float],
    selection_budget: int,
    *,
    excluded_ids: set[str] | None = None,
) -> set[str]:
    excluded_ids = excluded_ids or set()
    ranked = []
    for candidate in candidates:
        if candidate.candidate_id in excluded_ids:
            continue
        expected_score = sum(
            posterior.get(name, 0.0) * requirement.score(candidate.features)
            for name, requirement in requirements.items()
        )
        ranked.append((expected_score, candidate.candidate_id))
    return {candidate_id for _, candidate_id in sorted(ranked, key=lambda item: (-item[0], item[1]))[:selection_budget]}


def _label_probability(
    requirements: dict[str, Requirement],
    posterior: dict[str, float],
    candidate: Candidate,
) -> float:
    return sum(
        posterior[name]
        for name, requirement in requirements.items()
        if predict_label(requirement, candidate) == 1
    )


def _query_diagnostics(
    query_candidates: list[Candidate],
    requirements: dict[str, Requirement],
    posterior: dict[str, float],
    candidate: Candidate,
    *,
    selection_budget: int,
    selection_candidates: list[Candidate] | None = None,
) -> dict[str, float | int]:
    """Estimate how a label query can change future selection.

    The estimate is counterfactual: it filters the current requirement
    posterior by each possible answer, then measures the symmetric difference
    between the current and counterfactual top-k sets. No hidden label is read.
    """
    probability_one = _label_probability(requirements, posterior, candidate)
    disagreement = 4.0 * probability_one * (1.0 - probability_one)
    if selection_candidates is None:
        selection_candidates = query_candidates
    current = _selected_ids(selection_candidates, requirements, posterior, selection_budget)
    expected_change = 0.0
    affected_count = 0
    for possible_label, probability in ((0, 1.0 - probability_one), (1, probability_one)):
        if probability <= 0.0:
            continue
        filtered = {
            name: weight
            for name, weight in posterior.items()
            if predict_label(requirements[name], candidate) == possible_label
        }
        if not filtered:
            continue
        counterfactual = _selected_ids(
            selection_candidates, requirements, _normalise(filtered), selection_budget
        )
        change = len(current.symmetric_difference(counterfactual))
        expected_change += probability * change
        affected_count = max(affected_count, change)
    return {
        "label_probability_one": probability_one,
        "disagreement": disagreement,
        "expected_selection_change": expected_change,
        "affected_selection_count": affected_count,
        "impact_value": disagreement * expected_change,
    }


def select_queries(
    candidates: list[Candidate],
    requirements: dict[str, Requirement],
    posterior: dict[str, float],
    *,
    strategy: str,
    cost_budget: float,
    selection_budget: int,
    random_reserve_fraction: float = 0.25,
    seed: int = 17,
    selection_candidates: list[Candidate] | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]], float]:
    """Select queries under equal cost, returning candidates and diagnostics."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if cost_budget <= 0 or selection_budget <= 0:
        raise ValueError("cost_budget and selection_budget must be positive")
    if not 0.0 <= random_reserve_fraction < 1.0:
        raise ValueError("random_reserve_fraction must be in [0, 1)")

    rng = random.Random(seed)
    remaining = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[Candidate] = []
    diagnostics: list[dict[str, Any]] = []
    spent = 0.0
    reserve_spent = 0.0
    reserve_budget = cost_budget * random_reserve_fraction

    def feasible() -> list[Candidate]:
        return [
            candidate
            for candidate in remaining.values()
            if spent + candidate.query_cost <= cost_budget + 1e-9
        ]

    while remaining and len(selected) < selection_budget:
        options = feasible()
        if not options:
            break
        use_random = strategy == "random" or (
            strategy == "impact-aware-random" and reserve_spent < reserve_budget - 1e-9
        )
        if use_random:
            candidate = rng.choice(sorted(options, key=lambda item: item.candidate_id))
            score = None
            info = {
                "disagreement": None,
                "expected_selection_change": None,
                "affected_selection_count": None,
                "impact_value": None,
                "selection_score": None,
                "random_reserve": True,
            }
        else:
            scored = []
            for option in options:
                info = _query_diagnostics(
                    candidates,
                    requirements,
                    posterior,
                    option,
                    selection_budget=selection_budget,
                    selection_candidates=selection_candidates,
                )
                if strategy == "disagreement-only":
                    score = float(info["disagreement"]) / option.query_cost
                else:
                    score = float(info["impact_value"]) / option.query_cost
                scored.append((score, option.candidate_id, option, info))
            score, _, candidate, info = max(scored, key=lambda item: (item[0], item[1]))
            info = {**info, "selection_score": score, "random_reserve": False}

        selected.append(candidate)
        remaining.pop(candidate.candidate_id)
        spent += candidate.query_cost
        if use_random and strategy == "impact-aware-random":
            reserve_spent += candidate.query_cost
        diagnostics.append(
            {
                "candidate_id": candidate.candidate_id,
                "query_cost": candidate.query_cost,
                "cumulative_cost": spent,
                **info,
            }
        )
    return selected, diagnostics, spent


def make_synthetic_benchmark(
    *, seed: int = 17, total_candidates: int = 240
) -> tuple[
    dict[str, Requirement],
    list[Candidate],
    list[Candidate],
    list[Candidate],
    list[Candidate],
]:
    """Create disjoint query, downstream-selection, and held-out pools."""
    if total_candidates < 60:
        raise ValueError("total_candidates must be at least 60")
    requirements = {
        "requirement-a": Requirement("requirement-a", (1.0, -0.4, 0.7, -0.2), 0.05),
        "requirement-b": Requirement("requirement-b", (0.4, 0.8, -0.5, 0.3), -0.05),
        "requirement-c": Requirement("requirement-c", (-0.6, 0.2, 0.9, 0.6), 0.10),
        "requirement-d": Requirement("requirement-d", (0.2, -0.9, 0.3, 0.8), -0.08),
    }
    rng = random.Random(seed)
    true_requirement = requirements["requirement-a"]
    generated = []
    for index in range(total_candidates * 3):
        features = tuple(rng.uniform(-1.0, 1.0) for _ in range(4))
        generated.append(
            Candidate(
                candidate_id=f"candidate-{index:04d}",
                features=features,
                true_label=predict_label(true_requirement, Candidate("tmp", features, 0, 1.0)),
                query_cost=1.0 + (index % 3) * 0.25,
            )
        )
    seed_end = max(16, total_candidates // 10)
    consensus = [
        candidate
        for candidate in generated
        if len({predict_label(requirement, candidate) for requirement in requirements.values()}) == 1
    ]
    if len(consensus) < seed_end:
        raise RuntimeError("synthetic benchmark could not construct an ambiguous seed set")
    seed_candidates = consensus[:seed_end]
    seed_ids = {candidate.candidate_id for candidate in seed_candidates}
    remaining = [candidate for candidate in generated if candidate.candidate_id not in seed_ids]
    rng.shuffle(remaining)
    query_count = max(48, total_candidates // 2)
    query_candidates = remaining[:query_count]
    data_start = query_count
    data_end = data_start + total_candidates
    data_candidates = remaining[data_start:data_end]
    heldout = remaining[data_end:data_end + total_candidates]
    if len(data_candidates) < 24 or len(heldout) < 24:
        raise RuntimeError("synthetic benchmark could not construct disjoint query/data/eval pools")
    return requirements, seed_candidates, query_candidates, data_candidates, heldout


def _initial_posterior(
    requirements: dict[str, Requirement], seed_candidates: Iterable[Candidate]
) -> dict[str, float]:
    posterior = {name: 1.0 for name in requirements}
    for candidate in seed_candidates:
        posterior = update_posterior(
            requirements, posterior, candidate, candidate.true_label
        )
    return posterior


def _knn_accuracy(training: list[Candidate], evaluation: list[Candidate], k: int = 5) -> float:
    correct = 0
    for candidate in evaluation:
        neighbours = sorted(
            (
                sum((a - b) ** 2 for a, b in zip(candidate.features, other.features)),
                other.true_label,
            )
            for other in training
        )[:k]
        predicted = int(sum(label for _, label in neighbours) * 2 >= len(neighbours))
        correct += predicted == candidate.true_label
    return correct / len(evaluation) if evaluation else 0.0


def run_strategy(
    *,
    seed: int,
    strategy: str,
    rounds: int = 2,
    cost_budget_per_round: float = 12.0,
    selection_budget: int = 12,
    random_reserve_fraction: float = 0.25,
) -> dict[str, Any]:
    requirements, seed_candidates, query_pool, data_pool, heldout = make_synthetic_benchmark(seed=seed)
    posterior = _initial_posterior(requirements, seed_candidates)
    remaining_queries = list(query_pool)
    round_rows = []
    for round_index in range(rounds):
        selected, diagnostics, spent = select_queries(
            remaining_queries,
            requirements,
            posterior,
            strategy=strategy,
            cost_budget=cost_budget_per_round,
            selection_budget=selection_budget,
            random_reserve_fraction=random_reserve_fraction,
            seed=seed * 100 + round_index,
            selection_candidates=data_pool,
        )
        if not selected:
            break
        for candidate in selected:
            posterior = update_posterior(
                requirements, posterior, candidate, candidate.true_label
            )
        selected_ids = {candidate.candidate_id for candidate in selected}
        remaining_queries = [
            candidate
            for candidate in remaining_queries
            if candidate.candidate_id not in selected_ids
        ]
        selected_data_ids = sorted(
            _selected_ids(data_pool, requirements, posterior, selection_budget)
        )
        round_rows.append(
            {
                "round": round_index,
                "queried_count": len(selected),
                "queried_ids": [candidate.candidate_id for candidate in selected],
                "feedback_cost": spent,
                "selected_data_ids_after_feedback": selected_data_ids,
                "diagnostics": diagnostics,
                "posterior": posterior,
            }
        )
    selected_data_ids = sorted(
        _selected_ids(data_pool, requirements, posterior, selection_budget)
    )
    data_by_id = {candidate.candidate_id: candidate for candidate in data_pool}
    training = [*seed_candidates, *(data_by_id[item] for item in selected_data_ids)]
    pool_ids = {
        "seed": {candidate.candidate_id for candidate in seed_candidates},
        "query": {candidate.candidate_id for candidate in query_pool},
        "selection": {candidate.candidate_id for candidate in data_pool},
        "heldout": {candidate.candidate_id for candidate in heldout},
    }
    overlap_counts = {
        f"{left}_{right}": len(pool_ids[left] & pool_ids[right])
        for index, left in enumerate(pool_ids)
        for right in list(pool_ids)[index + 1 :]
    }
    return {
        "strategy": strategy,
        "seed": seed,
        "rounds": round_rows,
        "total_feedback_cost": sum(row["feedback_cost"] for row in round_rows),
        "total_queries": sum(row["queried_count"] for row in round_rows),
        "final_selected_data_ids": selected_data_ids,
        "final_selected_data_count": len(selected_data_ids),
        "heldout_knn_accuracy": _knn_accuracy(training, heldout),
        "seed_count": len(seed_candidates),
        "query_pool_count": len(query_pool),
        "selection_pool_count": len(data_pool),
        "heldout_count": len(heldout),
        "pool_ids": {name: sorted(ids) for name, ids in pool_ids.items()},
        "pool_overlap_counts": overlap_counts,
    }


def run_matrix(
    *,
    seeds: Iterable[int] = range(17, 27),
    strategies: Iterable[str] = STRATEGIES,
    rounds: int = 2,
    cost_budget_per_round: float = 12.0,
    selection_budget: int = 12,
    random_reserve_fraction: float = 0.25,
) -> dict[str, Any]:
    seeds = tuple(seeds)
    strategies = tuple(strategies)
    if set(strategies) - set(STRATEGIES):
        raise ValueError(f"unsupported strategies: {strategies}")
    results = []
    for seed in seeds:
        for strategy in strategies:
            results.append(
                run_strategy(
                    seed=seed,
                    strategy=strategy,
                    rounds=rounds,
                    cost_budget_per_round=cost_budget_per_round,
                    selection_budget=selection_budget,
                    random_reserve_fraction=random_reserve_fraction,
                )
            )
    by_strategy = {strategy: [] for strategy in strategies}
    for result in results:
        by_strategy[result["strategy"]].append(result["heldout_knn_accuracy"])
    summary = {
        strategy: {
            "mean_heldout_knn_accuracy": sum(values) / len(values),
            "mean_total_feedback_cost": sum(
                result["total_feedback_cost"]
                for result in results
                if result["strategy"] == strategy
            )
            / len(values),
            "mean_total_queries": sum(
                result["total_queries"]
                for result in results
                if result["strategy"] == strategy
            )
            / len(values),
            "mean_final_selected_data_count": sum(
                result["final_selected_data_count"]
                for result in results
                if result["strategy"] == strategy
            )
            / len(values),
            "mean_cost_utilization": sum(
                result["total_feedback_cost"]
                / max(1e-12, rounds * cost_budget_per_round)
                for result in results
                if result["strategy"] == strategy
            )
            / len(values),
        }
        for strategy, values in by_strategy.items()
    }
    paired_delta_vs_random = {}
    for strategy in strategies:
        if strategy == "random":
            continue
        deltas = []
        for seed in seeds:
            left = next(
                result["heldout_knn_accuracy"]
                for result in results
                if result["strategy"] == strategy and result["seed"] == seed
            )
            right = next(
                result["heldout_knn_accuracy"]
                for result in results
                if result["strategy"] == "random" and result["seed"] == seed
            )
            deltas.append(left - right)
        paired_delta_vs_random[strategy] = {
            "n": len(deltas),
            "mean_accuracy_delta": sum(deltas) / len(deltas),
            "deltas": deltas,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "configuration": {
            "strategies": list(strategies),
            "seeds": list(seeds),
            "rounds": rounds,
            "cost_budget_per_round": cost_budget_per_round,
            "selection_budget": selection_budget,
            "random_reserve_fraction": random_reserve_fraction,
            "selection_uses_hidden_labels": False,
            "feedback_updates_requirement_posterior": True,
            "heldout_labels_used_for_selection": False,
        },
        "results": results,
        "summary": summary,
        "paired_delta_vs_random": paired_delta_vs_random,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("active-feedback-impact-local.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(17, 27)))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cost-budget-per-round", type=float, default=12.0)
    parser.add_argument("--selection-budget", type=int, default=12)
    parser.add_argument("--random-reserve-fraction", type=float, default=0.25)
    args = parser.parse_args()
    report = run_matrix(
        seeds=args.seeds,
        rounds=args.rounds,
        cost_budget_per_round=args.cost_budget_per_round,
        selection_budget=args.selection_budget,
        random_reserve_fraction=args.random_reserve_fraction,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for strategy, values in report["summary"].items():
        print(
            f"{strategy:24s} heldout_knn={values['mean_heldout_knn_accuracy']:.4f} "
            f"feedback_cost={values['mean_total_feedback_cost']:.2f} "
            f"queries={values['mean_total_queries']:.2f} "
            f"utilization={values['mean_cost_utilization']:.3f}"
        )
    for strategy, values in report["paired_delta_vs_random"].items():
        print(
            f"{strategy:24s} minus random delta="
            f"{values['mean_accuracy_delta']:+.4f} n={values['n']}"
        )


if __name__ == "__main__":
    main()
