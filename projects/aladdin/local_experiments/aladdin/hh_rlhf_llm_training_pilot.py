from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from local_experiments.aladdin.hh_rlhf_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    STRATEGIES,
    HHRLHFGroup,
    _evaluate,
    _groups_for_split_seed,
    _train,
    load_hh_rlhf_groups,
    select_feedback_groups,
)
from local_experiments.aladdin.hh_rlhf_llm_matrix import _response
from local_experiments.aladdin.hh_rlhf_llm_smoke import (
    SMOKE_VERSION,
    _build_prompt,
    _call_llm,
)
from core_system.mle_core.models import TaskRecord


PILOT_VERSION = "aladdin-hh-rlhf-real-llm-training-pilot-v1"
SEED_ONLY_STRATEGY = "seed-only"


def _ordered_candidates(seed: int, group_id: str, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Use pair identity for A/B order so selection order cannot affect labels."""
    order_seed = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16)
    result = list(candidates)
    random.Random(order_seed).shuffle(result)
    return result


def _relabel(records: tuple[TaskRecord, ...], preferred_candidate: str) -> list[TaskRecord]:
    """Convert one provider judgment into the labels consumed by the task model."""
    result: list[TaskRecord] = []
    for record in records:
        candidate = str(record.metadata.get("candidate", ""))
        if not candidate:
            candidate = record.record_id.rsplit("-", 1)[-1]
        target = "preferred" if candidate == preferred_candidate else "rejected"
        result.append(replace(record, target=target))
    return result


def _judgment(
    *,
    group: HHRLHFGroup,
    seed: int,
    group_index: int,
    endpoint: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    released = group.release_feedback()
    if len(released) != 2:
        raise ValueError(f"feedback group must contain two records: {group.group_id}")
    candidates = _ordered_candidates(
        seed,
        group.group_id,
        [("chosen", _response(released[0])), ("rejected", _response(released[1]))],
    )
    label_a, candidate_a = candidates[0]
    label_b, candidate_b = candidates[1]
    system_prompt, user_prompt = _build_prompt(group.prompt, candidate_a, candidate_b)
    choice, usage = _call_llm(
        url=endpoint,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=timeout_seconds,
    )
    predicted_label = label_a if choice == "A" else label_b
    return {
        "group_id": group.group_id,
        "source_line": released[0].metadata.get("source_line"),
        "candidate_order": [label_a, label_b],
        "predicted_label": predicted_label,
        "ground_truth_label": "chosen",
        "correct": predicted_label == "chosen",
        "prompt_hash": hashlib.sha256(group.prompt.encode("utf-8")).hexdigest(),
        "usage": usage,
    }


def _llm_records(group: HHRLHFGroup, judgment: dict[str, Any]) -> list[TaskRecord]:
    preferred = str(judgment["predicted_label"])
    return _relabel(group.release_feedback(), preferred)


def run_training_pilot(
    data_path: Path,
    *,
    output_path: Path,
    endpoint: str,
    model: str,
    model_revision: str,
    strategies: tuple[str, ...] = STRATEGIES,
    seeds: tuple[int, ...] = (17, 18),
    split_seed: int = 17,
    seed_group_limit: int = 32,
    feedback_budget: int = 5,
    pool_group_limit: int = 64,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    if not strategies or not seeds:
        raise ValueError("strategies and seeds must not be empty")
    if feedback_budget <= 0:
        raise ValueError("feedback_budget must be positive")

    groups = load_hh_rlhf_groups(data_path, split_seed=split_seed)
    results: list[dict[str, Any]] = []
    total_runs = len(strategies) * len(seeds)
    total_calls = (len(strategies) - strategies.count(SEED_ONLY_STRATEGY)) * len(seeds) * feedback_budget
    completed_calls = 0
    started_at = time.monotonic()

    for offset, seed in enumerate(seeds):
        run_split_seed = split_seed + offset
        run_groups = _groups_for_split_seed(groups, run_split_seed)
        seed_groups = [g for g in run_groups if g.split == "seed"][:seed_group_limit]
        pool_groups = [g for g in run_groups if g.split == "held_in"][:pool_group_limit]
        held_out = [
            record
            for group in run_groups
            if group.split == "held_out"
            for record in group.release_feedback()
        ]
        if len(seed_groups) < 2 or len(pool_groups) < feedback_budget or not held_out:
            raise ValueError(f"insufficient groups for seed={seed}, split_seed={run_split_seed}")
        seed_records = [record for group in seed_groups for record in group.release_feedback()]

        for strategy in strategies:
            if strategy == SEED_ONLY_STRATEGY:
                selected = []
                diagnostics = {
                    "strategy": strategy,
                    "selection_seed": seed,
                    "feedback_budget_groups": 0,
                    "pool_groups": len(pool_groups),
                    "selected_group_ids": [],
                    "selection_definition": "train on the fixed seed records without additional feedback",
                }
            else:
                selected, diagnostics = select_feedback_groups(
                    pool_groups,
                    strategy=strategy,
                    budget=feedback_budget,
                    seed=seed,
                    seed_records=seed_records,
                )
            judgments: list[dict[str, Any]] = []
            llm_records: list[TaskRecord] = []
            oracle_records: list[TaskRecord] = []
            for group_index, group in enumerate(selected):
                judgment = _judgment(
                    group=group,
                    seed=seed,
                    group_index=group_index,
                    endpoint=endpoint,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )
                judgments.append(judgment)
                llm_records.extend(_llm_records(group, judgment))
                oracle_records.extend(group.release_feedback())
                completed_calls += 1
                elapsed = time.monotonic() - started_at
                eta = elapsed / completed_calls * (total_calls - completed_calls)
                print(
                    f"[{completed_calls}/{total_calls}] pilot strategy={strategy} seed={seed} "
                    f"correct={judgment['correct']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

            llm_checkpoint = _train([*seed_records, *llm_records], seed)
            oracle_checkpoint = _train([*seed_records, *oracle_records], seed)
            llm_metrics = _evaluate(llm_checkpoint, held_out)
            oracle_metrics = _evaluate(oracle_checkpoint, held_out)
            correct = sum(bool(item["correct"]) for item in judgments)
            results.append({
                "strategy": strategy,
                "seed": seed,
                "split_seed": run_split_seed,
                "metrics": {
                    "feedback_accuracy": correct / len(judgments) if judgments else None,
                    "feedback_correct": correct,
                    "feedback_groups": len(judgments),
                    "llm_feedback_task_model": llm_metrics,
                    "oracle_feedback_task_model": oracle_metrics,
                    "llm_minus_oracle_pairwise": (
                        llm_metrics["pairwise_accuracy"] - oracle_metrics["pairwise_accuracy"]
                    ),
                },
                "budget": {
                    "feedback_requests": len(selected),
                    "llm_calls": len(judgments),
                    "llm_tokens": sum(item["usage"]["total_tokens"] for item in judgments),
                    "training_steps": 2,
                    "evaluator_calls": 2,
                },
                "diagnostics": diagnostics,
                "model": {"id": model, "revision": model_revision},
                "judgments": judgments,
            })

    summary: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        values = [item for item in results if item["strategy"] == strategy]
        feedback_values = [
            item["metrics"]["feedback_accuracy"]
            for item in values
            if item["metrics"]["feedback_accuracy"] is not None
        ]
        summary[strategy] = {
            "mean_feedback_accuracy": (
                sum(feedback_values) / len(feedback_values) if feedback_values else None
            ),
            "mean_llm_pairwise_accuracy": sum(
                item["metrics"]["llm_feedback_task_model"]["pairwise_accuracy"] for item in values
            ) / len(values),
            "mean_oracle_pairwise_accuracy": sum(
                item["metrics"]["oracle_feedback_task_model"]["pairwise_accuracy"] for item in values
            ) / len(values),
            "mean_llm_minus_oracle_pairwise": sum(
                item["metrics"]["llm_minus_oracle_pairwise"] for item in values
            ) / len(values),
        }
    return {
        "experiment": "aladdin-hh-rlhf-real-llm-training-pilot",
        "status": "completed",
        "pilot_version": PILOT_VERSION,
        "smoke_protocol_version": SMOKE_VERSION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_path": str(data_path),
        "final_test": "not loaded",
        "model": {"id": model, "revision": model_revision, "endpoint": endpoint},
        "configuration": {
            "strategies": list(strategies),
            "seeds": list(seeds),
            "seed_group_limit": seed_group_limit,
            "pool_group_limit": pool_group_limit,
            "feedback_budget_groups": feedback_budget,
            "total_runs": total_runs,
            "total_llm_calls": total_calls,
            "timeout_seconds": timeout_seconds,
            "task_model": "tfidf-logistic-regression-v1",
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "The feedback provider is a real LLM, but the task model is still the local TF-IDF pilot.",
            "Oracle feedback is an upper-bound control using the official chosen/rejected labels.",
            "This pilot evaluates held-out only; official final_test remains sealed.",
            "LLM feedback task-model results are not a Qwen fine-tuning or DPO result.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Aladdin real-LLM feedback training pilot.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[*STRATEGIES, SEED_ONLY_STRATEGY],
        help="Strategies to compare; seed-only is the no-feedback control.",
    )
    parser.add_argument("--feedback-budget", type=int, default=5)
    parser.add_argument("--seed-group-limit", type=int, default=32)
    parser.add_argument("--pool-group-limit", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    report = run_training_pilot(
        args.data,
        output_path=args.output,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=args.model_revision,
        strategies=tuple(args.strategies),
        seeds=tuple(args.seeds),
        feedback_budget=args.feedback_budget,
        seed_group_limit=args.seed_group_limit,
        pool_group_limit=args.pool_group_limit,
        split_seed=args.split_seed,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
