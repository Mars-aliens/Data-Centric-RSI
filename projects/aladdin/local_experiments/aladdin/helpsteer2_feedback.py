from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core_system.mle_core.models import Budget, TaskRecord
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
    _checkpoint_scores,
)


DATASET_ID = "nvidia/HelpSteer2"
DATASET_REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
ADAPTER_VERSION = "aladdin-helpsteer2-feedback-v1"
LABELS = ("preferred", "rejected", "tie")
STRATEGIES = ("random", "uniform", "disagreement", "novelty", "aladdin")
EXPERIMENTAL_STRATEGIES = ("aladdin-v2",)
SUPPORTED_STRATEGIES = STRATEGIES + EXPERIMENTAL_STRATEGIES
SCORE_FIELDS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


class HelpSteer2AdapterError(ValueError):
    pass


def _stable_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


@dataclass(frozen=True)
class HelpSteer2Group:
    group_id: str
    prompt: str
    split: str
    batch_id: str
    hidden_records: tuple[TaskRecord, ...]
    released_records: tuple[TaskRecord, ...]
    labels: tuple[str, ...]
    score_means: tuple[float, ...]

    @property
    def feedback_size(self) -> int:
        return len(self.hidden_records)

    def release_feedback(self) -> tuple[TaskRecord, ...]:
        """Return labels only after this feedback group is selected."""
        return self.released_records


def _feedback_metadata(record: TaskRecord, *, released: bool) -> dict[str, Any]:
    base = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "adapter_version": ADAPTER_VERSION,
        "feedback_type": "human_multi_attribute_preference",
        "feedback_released": released,
        "feedback_group_id": record.source_group_id,
        "pair_index": record.metadata.get("pair_index"),
        "pair_size": record.metadata.get("pair_size"),
        "batch_id": record.metadata.get("batch_id"),
    }
    if released:
        base["feedback"] = record.metadata.get("feedback")
        base["scores"] = record.metadata.get("scores")
        base["score_mean"] = record.metadata.get("score_mean")
    return base


def _hidden_record(record: TaskRecord) -> TaskRecord:
    return TaskRecord(
        record_id=record.record_id,
        task_id=record.task_id,
        split=record.split,
        input=record.input,
        target=None,
        source_group_id=record.source_group_id,
        metadata=_feedback_metadata(record, released=False),
        parent_record_ids=record.parent_record_ids,
    )


def _released_record(record: TaskRecord) -> TaskRecord:
    if record.target not in LABELS:
        raise HelpSteer2AdapterError(f"unexpected HelpSteer2 target: {record.target!r}")
    return TaskRecord(
        record_id=record.record_id,
        task_id=record.task_id,
        split=record.split,
        input=record.input,
        target=record.target,
        source_group_id=record.source_group_id,
        metadata=_feedback_metadata(record, released=True),
        parent_record_ids=record.parent_record_ids,
    )


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HelpSteer2AdapterError(f"{path}:{line_number} is not an object")
            yield value


def load_helpsteer2_groups(path: Path, *, include_final_test: bool = False) -> list[HelpSteer2Group]:
    grouped: dict[str, list[TaskRecord]] = defaultdict(list)
    for line_number, row in enumerate(_load_jsonl(path), start=1):
        required = {"record_id", "task_id", "split", "input", "target", "source_group_id", "metadata"}
        missing = sorted(required - row.keys())
        if missing:
            raise HelpSteer2AdapterError(f"line {line_number} missing fields: {missing}")
        split = str(row["split"])
        if split == "final_test" and not include_final_test:
            continue
        if split not in {"seed", "held_in", "held_out", "final_test"}:
            raise HelpSteer2AdapterError(f"unsupported split: {split}")
        record = TaskRecord(
            record_id=str(row["record_id"]),
            task_id=str(row["task_id"]),
            split=split,
            input=row["input"],
            target=row["target"],
            source_group_id=str(row["source_group_id"]),
            metadata=dict(row["metadata"]),
            parent_record_ids=tuple(row.get("parent_record_ids", [])),
        )
        if record.target not in LABELS:
            raise HelpSteer2AdapterError(f"line {line_number} has invalid target: {record.target!r}")
        grouped[record.source_group_id].append(record)

    groups: list[HelpSteer2Group] = []
    for group_id in sorted(grouped):
        records = sorted(grouped[group_id], key=lambda item: int(item.metadata.get("pair_index", 0)))
        splits = {record.split for record in records}
        if len(splits) != 1:
            raise HelpSteer2AdapterError(f"group crosses split boundary: {group_id}")
        if len(records) < 2:
            raise HelpSteer2AdapterError(f"feedback group has fewer than two candidates: {group_id}")
        prompts = {str(record.input).split("\nResponse:", 1)[0] for record in records}
        if len(prompts) != 1:
            raise HelpSteer2AdapterError(f"group has inconsistent prompts: {group_id}")
        prompt = next(iter(prompts))
        batch_ids = {str(record.metadata.get("batch_id", "unassigned")) for record in records}
        if len(batch_ids) != 1:
            raise HelpSteer2AdapterError(f"group has inconsistent batch ids: {group_id}")
        groups.append(HelpSteer2Group(
            group_id=group_id,
            prompt=prompt,
            split=records[0].split,
            batch_id=next(iter(batch_ids)),
            hidden_records=tuple(_hidden_record(record) for record in records),
            released_records=tuple(_released_record(record) for record in records),
            labels=tuple(str(record.target) for record in records),
            score_means=tuple(float(record.metadata.get("score_mean", 0.0)) for record in records),
        ))
    return groups


def _train(records: Iterable[TaskRecord], seed: int) -> dict[str, Any]:
    trainer = TfidfLogisticTrainer()
    checkpoint, _ = trainer.train(
        train_records=list(records),
        initial_checkpoint=None,
        config={"ngram_range": (1, 2), "max_iter": 200, "steps": 1},
        seed=seed,
        store=None,
    )
    return checkpoint


def _scores(checkpoint: dict[str, Any], record: TaskRecord) -> dict[str, float]:
    return _checkpoint_scores(checkpoint, str(record.input))


def _group_uncertainty(group: HelpSteer2Group, checkpoint: dict[str, Any]) -> float:
    margins = []
    for record in group.hidden_records:
        values = sorted(_scores(checkpoint, record).values(), reverse=True)
        margins.append(values[0] - values[1] if len(values) > 1 else values[0])
    return 1.0 / (1.0 + sum(abs(value) for value in margins) / len(margins))


def _group_novelty(group: HelpSteer2Group, seed_prompts: list[str]) -> float:
    return _group_novelty_tokens(
        _tokens(group.prompt),
        [_tokens(seed_prompt) for seed_prompt in seed_prompts],
    )


def _group_novelty_tokens(
    prompt_tokens: set[str],
    context_token_sets: list[set[str]],
) -> float:
    if not prompt_tokens or not context_token_sets:
        return 1.0
    overlaps = []
    for context_tokens in context_token_sets:
        union = prompt_tokens | context_tokens
        overlaps.append(len(prompt_tokens & context_tokens) / len(union) if union else 0.0)
    return 1.0 - max(overlaps)


def _rank_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def select_feedback_groups(
    groups: list[HelpSteer2Group],
    *,
    strategy: str,
    budget: int,
    seed: int,
    seed_records: list[TaskRecord],
    aladdin_score: str = "product",
    aladdin_novelty_weight: float = 0.5,
) -> tuple[list[HelpSteer2Group], dict[str, Any]]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unknown HelpSteer2 strategy: {strategy}")
    if budget <= 0 or budget > len(groups):
        raise ValueError("feedback budget must be positive and no larger than the pool")
    if aladdin_score not in {"product", "additive"}:
        raise ValueError(f"unknown Aladdin score: {aladdin_score}")
    if not 0.0 <= aladdin_novelty_weight <= 1.0:
        raise ValueError("Aladdin novelty weight must be between 0 and 1")
    randomizer = random.Random(seed)
    checkpoint = _train(seed_records, seed)
    seed_prompts = [str(record.input).split("\nResponse:", 1)[0] for record in seed_records]
    seed_token_sets = [_tokens(prompt) for prompt in seed_prompts]
    group_token_sets = {group.group_id: _tokens(group.prompt) for group in groups}
    scored = []
    for group in groups:
        uncertainty = _group_uncertainty(group, checkpoint)
        novelty = _group_novelty_tokens(group_token_sets[group.group_id], seed_token_sets)
        impact = (
            uncertainty * (0.5 + novelty)
            if aladdin_score == "product"
            else (1.0 - aladdin_novelty_weight) * uncertainty + aladdin_novelty_weight * novelty
        )
        scored.append((group, uncertainty, novelty, impact))
    if strategy == "random":
        selected = randomizer.sample(groups, budget)
    elif strategy == "uniform":
        by_batch: dict[str, list[HelpSteer2Group]] = defaultdict(list)
        for group in groups:
            by_batch[group.batch_id].append(group)
        ordered = sorted(by_batch.items(), key=lambda item: _stable_hex(item[0]))
        selected = []
        cursor = 0
        while len(selected) < budget:
            batch_groups = ordered[cursor % len(ordered)][1]
            available = [group for group in batch_groups if group not in selected]
            if available:
                selected.append(sorted(available, key=lambda item: item.group_id)[0])
            cursor += 1
    elif strategy == "disagreement":
        selected = [item[0] for item in sorted(scored, key=lambda item: (-item[1], item[0].group_id))[:budget]]
    elif strategy == "novelty":
        selected = [item[0] for item in sorted(scored, key=lambda item: (-item[2], item[0].group_id))[:budget]]
    elif strategy == "aladdin":
        selected = [item[0] for item in sorted(scored, key=lambda item: (-item[3], item[0].group_id))[:budget]]
    else:
        # V2 keeps disagreement as the primary signal and applies diversity
        # greedily against both the seed prompts and already selected prompts.
        uncertainty_ranks = _rank_normalize([item[1] for item in scored])
        ranked_by_id = {
            item[0].group_id: rank
            for item, rank in zip(scored, uncertainty_ranks)
        }
        remaining = list(groups)
        selected = []
        context_token_sets = list(seed_token_sets)
        while remaining and len(selected) < budget:
            candidates = []
            for group in remaining:
                diversity = _group_novelty_tokens(
                    group_token_sets[group.group_id], context_token_sets
                )
                score = 0.75 * ranked_by_id[group.group_id] + 0.25 * diversity
                candidates.append((score, ranked_by_id[group.group_id], diversity, group))
            _, _, _, winner = max(
                candidates,
                key=lambda item: (item[0], item[1], item[3].group_id),
            )
            selected.append(winner)
            remaining.remove(winner)
            context_token_sets.append(group_token_sets[winner.group_id])
    diagnostics = {
        "strategy": strategy,
        "selection_seed": seed,
        "feedback_budget_groups": budget,
        "pool_groups": len(groups),
        "selected_group_ids": [group.group_id for group in selected],
        "selected_uncertainty": {group.group_id: next(item[1] for item in scored if item[0] == group) for group in selected},
        "selected_novelty": {group.group_id: next(item[2] for item in scored if item[0] == group) for group in selected},
        "selection_definition": {
            "random": "seeded random group sampling",
            "uniform": "round-robin over deterministic batch buckets",
            "disagreement": "smallest seed-model margin over candidate records",
            "novelty": "highest prompt novelty relative to seed prompts",
            "aladdin": (
                "uncertainty multiplied by prompt novelty as a local impact proxy"
                if aladdin_score == "product"
                else f"additive score with novelty weight {aladdin_novelty_weight:.3f}"
            ),
            "aladdin-v2": (
                "greedy selection: 75% rank-normalized uncertainty and "
                "25% prompt diversity against seed and selected prompts"
            ),
        }[strategy],
    }
    return selected, diagnostics


def _evaluate(checkpoint: dict[str, Any], records: list[TaskRecord]) -> dict[str, float]:
    metrics = TfidfLogisticEvaluator().evaluate(checkpoint=checkpoint, records=records, config={})[0]
    return {**metrics, **_pairwise_metrics(checkpoint, records)}


def _pairwise_metrics(
    checkpoint: dict[str, Any],
    records: list[TaskRecord],
) -> dict[str, float]:
    """Evaluate whether the model selects the preferred answer in each pair."""
    grouped: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in records:
        if record.source_group_id is not None:
            grouped[record.source_group_id].append(record)

    decisive_groups = 0
    correct_groups = 0
    tie_groups = 0
    for group_records in grouped.values():
        if len(group_records) != 2:
            continue
        labels = [str(record.target) for record in group_records]
        if labels.count("tie") == 2:
            tie_groups += 1
            continue
        if labels.count("preferred") != 1 or labels.count("rejected") != 1:
            continue
        predicted_index = max(
            range(2),
            key=lambda index: (
                _scores(checkpoint, group_records[index]).get("preferred", float("-inf")),
                -index,
            ),
        )
        actual_index = labels.index("preferred")
        decisive_groups += 1
        correct_groups += predicted_index == actual_index
    return {
        "pairwise_accuracy": correct_groups / decisive_groups if decisive_groups else 0.0,
        "pairwise_groups": float(decisive_groups),
        "pairwise_correct": float(correct_groups),
        "tie_groups": float(tie_groups),
    }


def run_helpsteer2_feedback_experiment(
    data_path: Path,
    *,
    strategies: tuple[str, ...] = STRATEGIES,
    seeds: tuple[int, ...] = (17, 18, 19),
    seed_group_limit: int = 64,
    feedback_budget: int = 20,
    pool_group_limit: int = 256,
) -> dict[str, Any]:
    groups = load_helpsteer2_groups(data_path)
    seed_groups = [group for group in groups if group.split == "seed"][:seed_group_limit]
    pool_groups = [group for group in groups if group.split == "held_in"][:pool_group_limit]
    held_out_records = [record for group in groups if group.split == "held_out" for record in group.released_records]
    if len(seed_groups) < 2 or len(pool_groups) < feedback_budget or not held_out_records:
        raise HelpSteer2AdapterError("insufficient seed, feedback-pool, or held-out groups")
    seed_records = [record for group in seed_groups for record in group.release_feedback()]
    results: list[dict[str, Any]] = []
    total_runs = len(seeds) * len(strategies)
    completed_runs = 0
    started_at = time.monotonic()
    for seed in seeds:
        for strategy in strategies:
            selected, diagnostics = select_feedback_groups(
                pool_groups,
                strategy=strategy,
                budget=feedback_budget,
                seed=seed,
                seed_records=seed_records,
            )
            released_records = [record for group in selected for record in group.release_feedback()]
            checkpoint = _train([*seed_records, *released_records], seed)
            results.append({
                "strategy": strategy,
                "seed": seed,
                "metrics": _evaluate(checkpoint, held_out_records),
                "budget": {
                    "feedback_requests": len(selected),
                    "feedback_records_released": len(released_records),
                    "training_steps": 1,
                    "llm_calls": 0,
                    "evaluator_calls": 1,
                },
                "diagnostics": diagnostics,
                "model": "tfidf-logistic-regression-v1",
                "llm_required": False,
            })
            completed_runs += 1
            elapsed = time.monotonic() - started_at
            rate = elapsed / completed_runs
            eta = rate * (total_runs - completed_runs)
            print(
                f"[{completed_runs}/{total_runs}] strategy={strategy} seed={seed} "
                f"pairwise={results[-1]['metrics']['pairwise_accuracy']:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    summary: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        strategy_results = [item for item in results if item["strategy"] == strategy]
        values = [item["metrics"]["accuracy"] for item in strategy_results]
        pairwise_values = [item["metrics"]["pairwise_accuracy"] for item in strategy_results]
        summary[strategy] = {
            "mean_held_out_accuracy": sum(values) / len(values),
            "min_held_out_accuracy": min(values),
            "max_held_out_accuracy": max(values),
            "mean_pairwise_accuracy": sum(pairwise_values) / len(pairwise_values),
            "min_pairwise_accuracy": min(pairwise_values),
            "max_pairwise_accuracy": max(pairwise_values),
        }
    return {
        "experiment": "aladdin-helpsteer2-technique-1-feedback-selection",
        "status": "local-mechanism-experiment",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "adapter_version": ADAPTER_VERSION,
        "data_path": str(data_path),
        "split_protocol": {
            "seed": f"first {seed_group_limit} groups from HelpSteer2 train seed split",
            "feedback_pool": f"first {pool_group_limit} groups from HelpSteer2 train held_in split",
            "held_out": "HelpSteer2 train held_out split",
            "final_test": "not loaded",
        },
        "configuration": {
            "seeds": list(seeds),
            "feedback_budget_groups": feedback_budget,
            "seed_groups": len(seed_groups),
            "feedback_pool_groups": len(pool_groups),
            "held_out_records": len(held_out_records),
            "strategies": list(strategies),
            "backbone_llm": "not used in local mechanism experiment",
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "The local Aladdin score uses a deterministic TF-IDF seed model and a novelty-based impact proxy.",
            "No LLM generation, LLM evaluator, or new human annotation is used.",
            "This tests feedback hiding, release, budget equality, and selection protocol; it is not a paper-level effectiveness claim.",
            "HelpSteer2 validation/final_test records are not loaded by this experiment.",
            "Pairwise accuracy excludes tie groups and evaluates whether the higher preferred-class score is assigned to the labeled preferred candidate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Aladdin HelpSteer2 feedback-selection experiment.")
    parser.add_argument("--data", type=Path, default=Path("data/helpsteer2-train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("local_experiments/aladdin/helpsteer2-technique-1-report.json"))
    parser.add_argument("--feedback-budget", type=int, default=20)
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--pool-group-limit", type=int, default=256)
    args = parser.parse_args()
    report = run_helpsteer2_feedback_experiment(
        args.data,
        feedback_budget=args.feedback_budget,
        seed_group_limit=args.seed_group_limit,
        pool_group_limit=args.pool_group_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
