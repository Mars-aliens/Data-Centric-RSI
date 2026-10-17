from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from local_experiments.aladdin.hh_rlhf_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    STRATEGIES,
    _groups_for_split_seed,
    load_hh_rlhf_groups,
    select_feedback_groups,
)
from local_experiments.aladdin.hh_rlhf_llm_smoke import (
    SMOKE_VERSION,
    _build_prompt,
    _call_llm,
)


MATRIX_VERSION = "aladdin-hh-rlhf-real-llm-feedback-matrix-v1"


def _response(record: Any) -> str:
    value = str(record.input)
    marker = "\n\nResponse:"
    if marker not in value:
        raise ValueError(f"record {record.record_id} has no response marker")
    return value.rsplit(marker, 1)[1].strip()


def _run_one(
    *,
    group: Any,
    seed: int,
    strategy: str,
    group_index: int,
    endpoint: str,
    model: str,
    model_revision: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    released = group.release_feedback()
    if len(released) != 2:
        raise ValueError(f"feedback group must contain two records: {group.group_id}")
    candidates = [("chosen", _response(released[0])), ("rejected", _response(released[1]))]
    random.Random(seed * 1000003 + group_index).shuffle(candidates)
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
        "batch_id": group.batch_id,
        "candidate_order": [label_a, label_b],
        "predicted_label": predicted_label,
        "ground_truth_label": "chosen",
        "correct": predicted_label == "chosen",
        "source_line": released[0].metadata.get("source_line"),
        "prompt_hash": hashlib.sha256(group.prompt.encode("utf-8")).hexdigest(),
        "system_prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_hash": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        "usage": usage,
    }


def run_llm_feedback_matrix(
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
    total_calls = total_runs * feedback_budget
    completed_runs = 0
    completed_calls = 0
    started_at = time.monotonic()
    split_seed_by_seed: dict[str, int] = {}
    for offset, seed in enumerate(seeds):
        run_split_seed = split_seed + offset
        split_seed_by_seed[str(seed)] = run_split_seed
        run_groups = _groups_for_split_seed(groups, run_split_seed)
        seed_groups = [group for group in run_groups if group.split == "seed"][:seed_group_limit]
        pool_groups = [group for group in run_groups if group.split == "held_in"][:pool_group_limit]
        if len(seed_groups) < 2 or len(pool_groups) < feedback_budget:
            raise ValueError(f"insufficient groups for seed={seed}, split_seed={run_split_seed}")
        seed_records = [record for group in seed_groups for record in group.release_feedback()]
        for strategy in strategies:
            selected, diagnostics = select_feedback_groups(
                pool_groups,
                strategy=strategy,
                budget=feedback_budget,
                seed=seed,
                seed_records=seed_records,
            )
            judgments: list[dict[str, Any]] = []
            for group_index, group in enumerate(selected):
                judgment = _run_one(
                    group=group,
                    seed=seed,
                    strategy=strategy,
                    group_index=group_index,
                    endpoint=endpoint,
                    model=model,
                    model_revision=model_revision,
                    timeout_seconds=timeout_seconds,
                )
                judgments.append(judgment)
                completed_calls += 1
                print(
                    f"[{completed_calls}/{total_calls}] llm-feedback strategy={strategy} seed={seed} "
                    f"correct={judgment['correct']} model={model}",
                    flush=True,
                )
            correct = sum(bool(item["correct"]) for item in judgments)
            run_result = {
                "strategy": strategy,
                "seed": seed,
                "split_seed": run_split_seed,
                "metrics": {
                    "feedback_accuracy": correct / len(judgments),
                    "feedback_correct": correct,
                    "feedback_groups": len(judgments),
                },
                "budget": {
                    "feedback_requests": len(selected),
                    "llm_calls": len(judgments),
                    "llm_tokens": sum(item["usage"]["total_tokens"] for item in judgments),
                    "training_steps": 0,
                    "evaluator_calls": 0,
                },
                "diagnostics": diagnostics,
                "judgments": judgments,
                "model": {"id": model, "revision": model_revision},
            }
            results.append(run_result)
            completed_runs += 1
            elapsed = time.monotonic() - started_at
            eta = elapsed / completed_calls * (total_calls - completed_calls) if completed_calls else 0.0
            print(
                f"run [{completed_runs}/{total_runs}] strategy={strategy} seed={seed} "
                f"feedback_accuracy={run_result['metrics']['feedback_accuracy']:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    summary: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        values = [item["metrics"]["feedback_accuracy"] for item in results if item["strategy"] == strategy]
        summary[strategy] = {
            "mean_feedback_accuracy": sum(values) / len(values),
            "min_feedback_accuracy": min(values),
            "max_feedback_accuracy": max(values),
        }
    return {
        "experiment": "aladdin-hh-rlhf-real-llm-feedback-matrix",
        "status": "completed",
        "matrix_version": MATRIX_VERSION,
        "smoke_protocol_version": SMOKE_VERSION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_path": str(data_path),
        "final_test": "not loaded",
        "model": {"id": model, "revision": model_revision, "endpoint": endpoint},
        "split_protocol": {
            "base_split_seed": split_seed,
            "split_seed_by_seed": split_seed_by_seed,
            "final_test": "sealed and not loaded",
        },
        "configuration": {
            "strategies": list(strategies),
            "seeds": list(seeds),
            "seed_group_limit": seed_group_limit,
            "pool_group_limit": pool_group_limit,
            "feedback_budget_groups": feedback_budget,
            "total_runs": total_runs,
            "total_llm_calls": total_calls,
            "timeout_seconds": timeout_seconds,
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "This matrix uses the real LLM as a feedback provider, not as a fine-tuned task model.",
            "The chosen/rejected labels are hidden from the LLM prompt and used only for audit accuracy.",
            "No model checkpoint is trained in this stage; training integration is the next stage.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small real-LLM Aladdin feedback matrix.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--feedback-budget", type=int, default=5)
    parser.add_argument("--seed-group-limit", type=int, default=32)
    parser.add_argument("--pool-group-limit", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    report = run_llm_feedback_matrix(
        args.data,
        output_path=args.output,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=args.model_revision,
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
