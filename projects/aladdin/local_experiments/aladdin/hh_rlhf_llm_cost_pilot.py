from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from core_system.mle_core.models import TaskRecord
from local_experiments.aladdin.hh_rlhf_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    STRATEGIES,
    _evaluate,
    _groups_for_split_seed,
    _train,
    load_hh_rlhf_groups,
    select_feedback_groups,
)
from local_experiments.aladdin.helpsteer2_feedback import SUPPORTED_STRATEGIES
from local_experiments.aladdin.hh_rlhf_llm_matrix import _response
from local_experiments.aladdin.hh_rlhf_llm_smoke import _build_prompt, _call_llm, SMOKE_VERSION


COST_PILOT_VERSION = "aladdin-hh-rlhf-real-llm-cost-pilot-v1"
SEED_ONLY_STRATEGY = "seed-only"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _progress_line(completed: int, total: int, *, width: int = 24) -> str:
    ratio = completed / total if total else 1.0
    filled = min(width, int(width * ratio))
    return f"[{'#' * filled}{'.' * (width - filled)}] {completed}/{total}"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_resume_results(path: Path, *, strategies: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from invalid report: {path}") from exc
    if report.get("experiment") != "aladdin-hh-rlhf-real-llm-cost-pilot":
        raise ValueError(f"resume report has a different experiment: {path}")
    results = report.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"resume report has invalid results: {path}")
    allowed = {(strategy, seed) for strategy in strategies for seed in seeds}
    resumed: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"resume report contains an invalid result: {path}")
        key = (str(result.get("strategy")), int(result.get("seed")))
        if key not in allowed:
            raise ValueError(f"resume report configuration differs at {key}")
        if key in seen:
            raise ValueError(f"resume report contains duplicate result: {key}")
        seen.add(key)
        resumed.append(result)
    return resumed


def _candidate_order(group: Any, seed: int, group_index: int) -> list[tuple[str, str]]:
    released = group.release_feedback()
    candidates = [("chosen", _response(released[0])), ("rejected", _response(released[1]))]
    random.Random(seed * 1000003 + group_index).shuffle(candidates)
    return candidates


def _estimate_prompt_tokens(group: Any, seed: int, group_index: int) -> int:
    """Conservative ASCII prompt estimate used before spending the next request."""
    candidates = _candidate_order(group, seed, group_index)
    _, user_prompt = _build_prompt(group.prompt, candidates[0][1], candidates[1][1])
    # HH-RLHF is English-heavy; three characters per token is intentionally conservative.
    return max(1, math.ceil(len(user_prompt) / 3))


def _relabel(records: tuple[TaskRecord, ...], preferred_candidate: str) -> list[TaskRecord]:
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
    group: Any,
    seed: int,
    group_index: int,
    endpoint: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    released = group.release_feedback()
    candidates = _candidate_order(group, seed, group_index)
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
        "estimated_prompt_tokens": _estimate_prompt_tokens(group, seed, group_index),
        "usage": usage,
    }


def _records_from_judgment(group: Any, judgment: dict[str, Any]) -> list[TaskRecord]:
    return _relabel(group.release_feedback(), str(judgment["predicted_label"]))


def _run_judgments(
    selected: list[Any],
    *,
    seed: int,
    endpoint: str,
    model: str,
    timeout_seconds: float,
    token_budget: int,
    llm_workers: int,
    strategy: str,
    completed_runs: int,
    total_runs: int,
    started_at: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run independent feedback requests concurrently after deterministic budget admission."""
    admitted: list[tuple[int, Any, int]] = []
    estimated_tokens = 0
    skipped_groups = 0
    for group_index, group in enumerate(selected):
        estimate = _estimate_prompt_tokens(group, seed, group_index)
        if estimated_tokens + estimate > token_budget:
            skipped_groups = len(selected) - group_index
            break
        admitted.append((group_index, group, estimate))
        estimated_tokens += estimate

    if not admitted:
        return [], skipped_groups, estimated_tokens

    workers = min(llm_workers, len(admitted))
    judgments_by_index: dict[int, dict[str, Any]] = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _judgment,
                group=group,
                seed=seed,
                group_index=group_index,
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout_seconds,
            ): group_index
            for group_index, group, _ in admitted
        }
        for future in concurrent.futures.as_completed(futures):
            group_index = futures[future]
            judgment = future.result()
            judgments_by_index[group_index] = judgment
            completed += 1
            elapsed = time.monotonic() - started_at
            # The run-level estimate is intentionally based on completed runs, so it
            # remains meaningful even when token budgets yield different group counts.
            effective_completed = completed_runs + completed / max(len(admitted), 1)
            rate = elapsed / effective_completed if effective_completed else 0.0
            eta = rate * max(total_runs - effective_completed, 0)
            print(
                f"feedback [{completed}/{len(admitted)}] strategy={strategy} seed={seed} "
                f"correct={judgment['correct']} workers={workers} "
                f"overall={_progress_line(completed_runs, total_runs)} "
                f"elapsed={_format_duration(elapsed)} eta~{_format_duration(eta)}",
                flush=True,
            )

    judgments = [judgments_by_index[group_index] for group_index, _, _ in admitted]
    return judgments, skipped_groups, estimated_tokens


def run_cost_pilot(
    data_path: Path,
    *,
    output_path: Path,
    endpoint: str,
    model: str,
    model_revision: str,
    strategies: tuple[str, ...] = (*STRATEGIES, SEED_ONLY_STRATEGY),
    seeds: tuple[int, ...] = (17, 18, 19, 20, 21),
    split_seed: int = 17,
    seed_group_limit: int = 64,
    candidate_budget: int = 64,
    token_budget: int = 8000,
    pool_group_limit: int = 256,
    timeout_seconds: float = 300.0,
    llm_workers: int = 1,
    resume: bool = False,
    aladdin_score: str = "product",
    aladdin_novelty_weight: float = 0.5,
) -> dict[str, Any]:
    if not strategies or not seeds:
        raise ValueError("strategies and seeds must not be empty")
    allowed_strategies = {*SUPPORTED_STRATEGIES, SEED_ONLY_STRATEGY}
    invalid_strategies = set(strategies) - allowed_strategies
    if invalid_strategies:
        raise ValueError(f"invalid strategies: {sorted(invalid_strategies)}")
    if candidate_budget <= 0 or token_budget <= 0:
        raise ValueError("candidate_budget and token_budget must be positive")
    if llm_workers <= 0:
        raise ValueError("llm_workers must be positive")

    groups = load_hh_rlhf_groups(data_path, split_seed=split_seed)
    results = _load_resume_results(output_path, strategies=strategies, seeds=seeds) if resume else []
    completed_keys = {(str(item["strategy"]), int(item["seed"])) for item in results}
    total_runs = len(strategies) * len(seeds)
    completed_runs = len(results)
    started_at = time.monotonic()
    print(
        f"cost-pilot start {_progress_line(completed_runs, total_runs)} "
        f"runs={total_runs} llm_workers={llm_workers} token_budget={token_budget} "
        f"resume={resume}",
        flush=True,
    )

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
        if len(seed_groups) < 2 or len(pool_groups) < candidate_budget or not held_out:
            raise ValueError(f"insufficient groups for seed={seed}, split_seed={run_split_seed}")
        seed_records = [record for group in seed_groups for record in group.release_feedback()]

        for strategy in strategies:
            if (strategy, seed) in completed_keys:
                print(
                    f"[resume {completed_runs}/{total_runs}] skip strategy={strategy} seed={seed}",
                    flush=True,
                )
                continue
            if strategy == SEED_ONLY_STRATEGY:
                selected: list[Any] = []
                diagnostics = {
                    "strategy": strategy,
                    "selection_seed": seed,
                    "candidate_budget_groups": 0,
                    "pool_groups": len(pool_groups),
                    "selected_group_ids": [],
                    "selection_definition": "train on fixed seed records without additional feedback",
                }
            else:
                selected, diagnostics = select_feedback_groups(
                    pool_groups,
                    strategy=strategy,
                    budget=candidate_budget,
                    seed=seed,
                    seed_records=seed_records,
                    aladdin_score=aladdin_score,
                    aladdin_novelty_weight=aladdin_novelty_weight,
                )

            judgments: list[dict[str, Any]] = []
            feedback_records: list[TaskRecord] = []
            judgments, skipped_groups, estimated_tokens = _run_judgments(
                selected,
                seed=seed,
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout_seconds,
                token_budget=token_budget,
                llm_workers=llm_workers,
                strategy=strategy,
                completed_runs=completed_runs,
                total_runs=total_runs,
                started_at=started_at,
            )
            selected_by_id = {group.group_id: group for group in selected}
            for judgment in judgments:
                feedback_records.extend(
                    _records_from_judgment(selected_by_id[judgment["group_id"]], judgment)
                )
            actual_tokens = sum(int(item["usage"]["total_tokens"]) for item in judgments)

            llm_checkpoint = _train([*seed_records, *feedback_records], seed)
            oracle_records = [record for group in selected[:len(judgments)] for record in group.release_feedback()]
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
                    "llm_minus_oracle_pairwise": llm_metrics["pairwise_accuracy"] - oracle_metrics["pairwise_accuracy"],
                },
                "budget": {
                    "token_budget": token_budget,
                    "candidate_budget_groups": candidate_budget,
                    "feedback_requests": len(judgments),
                    "skipped_groups": skipped_groups,
                    "llm_calls": len(judgments),
                    "llm_tokens": actual_tokens,
                    "estimated_prompt_tokens": estimated_tokens,
                    "llm_workers": min(llm_workers, len(judgments)) if judgments else 0,
                    "token_budget_exceeded": actual_tokens > token_budget,
                    "training_steps": 2,
                    "evaluator_calls": 2,
                },
                "diagnostics": diagnostics,
                "model": {"id": model, "revision": model_revision},
                "judgments": judgments,
            })
            completed_keys.add((strategy, seed))
            _write_json_atomic(
                output_path,
                {
                    "experiment": "aladdin-hh-rlhf-real-llm-cost-pilot",
                    "status": "running",
                    "cost_pilot_version": COST_PILOT_VERSION,
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
                        "candidate_budget_groups": candidate_budget,
                        "token_budget_per_run": token_budget,
                        "llm_workers": llm_workers,
                        "resume": True,
                        "task_model": "tfidf-logistic-regression-v1",
                        "aladdin_score": aladdin_score,
                        "aladdin_novelty_weight": aladdin_novelty_weight,
                    },
                    "completed_runs": len(results),
                    "total_runs": total_runs,
                    "summary": {},
                    "results": results,
                },
            )
            completed_runs += 1
            elapsed = time.monotonic() - started_at
            eta = elapsed / completed_runs * (total_runs - completed_runs)
            print(
                f"{_progress_line(completed_runs, total_runs)} cost-pilot "
                f"strategy={strategy} seed={seed} "
                f"groups={len(judgments)} tokens={actual_tokens}/{token_budget} "
                f"pairwise={llm_metrics['pairwise_accuracy']:.4f} "
                f"elapsed={_format_duration(elapsed)} eta~{_format_duration(eta)}",
                flush=True,
            )

    summary: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        values = [item for item in results if item["strategy"] == strategy]
        feedback_values = [item["metrics"]["feedback_accuracy"] for item in values if item["metrics"]["feedback_accuracy"] is not None]
        summary[strategy] = {
            "mean_feedback_accuracy": sum(feedback_values) / len(feedback_values) if feedback_values else None,
            "mean_feedback_groups": sum(item["metrics"]["feedback_groups"] for item in values) / len(values),
            "mean_llm_pairwise_accuracy": sum(item["metrics"]["llm_feedback_task_model"]["pairwise_accuracy"] for item in values) / len(values),
            "mean_oracle_pairwise_accuracy": sum(item["metrics"]["oracle_feedback_task_model"]["pairwise_accuracy"] for item in values) / len(values),
            "mean_llm_tokens": sum(item["budget"]["llm_tokens"] for item in values) / len(values),
        }

    report = {
        "experiment": "aladdin-hh-rlhf-real-llm-cost-pilot",
        "status": "completed",
        "cost_pilot_version": COST_PILOT_VERSION,
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
            "candidate_budget_groups": candidate_budget,
            "token_budget_per_run": token_budget,
            "llm_workers": llm_workers,
            "task_model": "tfidf-logistic-regression-v1",
            "aladdin_score": aladdin_score,
            "aladdin_novelty_weight": aladdin_novelty_weight,
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "The pre-call token guard uses a conservative character-based estimate; actual usage is recorded separately.",
            "A fixed token budget can produce different numbers of feedback groups across strategies.",
            "The task model remains TF-IDF and this is not a Qwen fine-tuning or DPO result.",
            "Official final_test remains sealed.",
        ],
    }
    _write_json_atomic(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aladdin under a fixed LLM token budget.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--strategies", nargs="+", default=[*STRATEGIES, SEED_ONLY_STRATEGY])
    parser.add_argument("--candidate-budget", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=8000)
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--pool-group-limit", type=int, default=256)
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=1,
        help="Concurrent independent feedback requests; use 8 on the server after a small smoke test.",
    )
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--aladdin-score", choices=("product", "additive"), default="product")
    parser.add_argument("--aladdin-novelty-weight", type=float, default=0.5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed strategy/seed runs from the output JSON checkpoint.",
    )
    args = parser.parse_args()
    report = run_cost_pilot(
        args.data,
        output_path=args.output,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=args.model_revision,
        strategies=tuple(args.strategies),
        seeds=tuple(args.seeds),
        candidate_budget=args.candidate_budget,
        token_budget=args.token_budget,
        seed_group_limit=args.seed_group_limit,
        pool_group_limit=args.pool_group_limit,
        llm_workers=args.llm_workers,
        split_seed=args.split_seed,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
        aladdin_score=args.aladdin_score,
        aladdin_novelty_weight=args.aladdin_novelty_weight,
    )
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
