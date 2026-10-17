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
from local_experiments.aladdin.helpsteer2_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    STRATEGIES,
    SUPPORTED_STRATEGIES,
    _evaluate,
    _train,
    load_helpsteer2_groups,
    select_feedback_groups,
)
from local_experiments.aladdin.hh_rlhf_llm_smoke import (
    SMOKE_VERSION,
    _build_prompt,
    _call_llm,
)


SEED_ONLY_STRATEGY = "seed-only"
VERSION = "aladdin-helpsteer2-real-llm-cost-pilot-v1"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def _progress(done: int, total: int, width: int = 24) -> str:
    ratio = done / total if total else 1.0
    filled = min(width, int(width * ratio))
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _response(record: TaskRecord) -> str:
    value = str(record.input)
    marker = "\nResponse:"
    if marker not in value:
        raise ValueError(f"record {record.record_id} has no response marker")
    return value.rsplit(marker, 1)[1].strip()


def _pairwise_groups(groups: list[Any]) -> list[Any]:
    result = []
    for group in groups:
        labels = [str(record.target) for record in group.released_records]
        if len(labels) == 2 and labels.count("preferred") == 1 and labels.count("rejected") == 1:
            result.append(group)
    return result


def _groups_for_run(
    groups: list[Any],
    *,
    split: str,
    limit: int,
    seed: int,
    salt: int,
    independent_run_splits: bool,
) -> list[Any]:
    candidates = sorted(
        (group for group in groups if group.split == split),
        key=lambda group: group.group_id,
    )
    if limit <= 0:
        raise ValueError(f"{split} group limit must be positive")
    if len(candidates) < limit:
        raise ValueError(f"insufficient {split} groups: {len(candidates)} < {limit}")
    if not independent_run_splits:
        return candidates[:limit]
    randomizer = random.Random(seed * 1000003 + salt)
    return sorted(randomizer.sample(candidates, limit), key=lambda group: group.group_id)


def _candidate_order(group: Any, seed: int, index: int) -> list[tuple[int, str]]:
    import random

    records = group.release_feedback()
    candidates = [(position, _response(record)) for position, record in enumerate(records)]
    random.Random(seed * 1000003 + index).shuffle(candidates)
    return candidates


def _estimate(group: Any, seed: int, index: int) -> int:
    candidates = _candidate_order(group, seed, index)
    _, first = candidates[0]
    _, second = candidates[1]
    _, prompt = _build_prompt(group.prompt, first, second)
    return max(1, math.ceil(len(prompt) / 3))


def _judgment(*, group: Any, seed: int, index: int, endpoint: str, model: str, timeout: float) -> dict[str, Any]:
    records = group.release_feedback()
    candidates = _candidate_order(group, seed, index)
    first_index, first_response = candidates[0]
    second_index, second_response = candidates[1]
    system_prompt, user_prompt = _build_prompt(group.prompt, first_response, second_response)
    choice, usage = _call_llm(
        url=endpoint,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=timeout,
    )
    predicted_index = first_index if choice == "A" else second_index
    truth_index = next(i for i, record in enumerate(records) if str(record.target) == "preferred")
    return {
        "group_id": group.group_id,
        "source_line": records[0].metadata.get("source_line"),
        "candidate_order": [first_index, second_index],
        "predicted_index": predicted_index,
        "ground_truth_index": truth_index,
        "correct": predicted_index == truth_index,
        "prompt_hash": hashlib.sha256(group.prompt.encode("utf-8")).hexdigest(),
        "estimated_prompt_tokens": _estimate(group, seed, index),
        "usage": usage,
    }


def _relabel(group: Any, judgment: dict[str, Any]) -> list[TaskRecord]:
    preferred = int(judgment["predicted_index"])
    return [
        replace(record, target="preferred" if index == preferred else "rejected")
        for index, record in enumerate(group.release_feedback())
    ]


def _run_judgments(selected: list[Any], *, seed: int, endpoint: str, model: str, timeout: float,
                   token_budget: int, workers: int, strategy: str, completed_runs: int,
                   total_runs: int, started_at: float) -> tuple[list[dict[str, Any]], int, int]:
    admitted: list[tuple[int, Any, int]] = []
    estimated = 0
    skipped = 0
    for index, group in enumerate(selected):
        estimate = _estimate(group, seed, index)
        if estimated + estimate > token_budget:
            skipped = len(selected) - index
            break
        admitted.append((index, group, estimate))
        estimated += estimate
    if not admitted:
        return [], skipped, estimated

    judgments_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(admitted))) as executor:
        futures = {
            executor.submit(
                _judgment,
                group=group,
                seed=seed,
                index=index,
                endpoint=endpoint,
                model=model,
                timeout=timeout,
            ): index
            for index, group, _ in admitted
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index = futures[future]
            judgment = future.result()
            judgments_by_index[index] = judgment
            elapsed = time.monotonic() - started_at
            effective = completed_runs + completed / len(admitted)
            eta = elapsed / effective * max(total_runs - effective, 0)
            print(
                f"feedback [{completed}/{len(admitted)}] strategy={strategy} seed={seed} "
                f"correct={judgment['correct']} workers={min(workers, len(admitted))} "
                f"overall={_progress(completed_runs, total_runs)} "
                f"elapsed={_duration(elapsed)} eta~{_duration(eta)}",
                flush=True,
            )
    return [judgments_by_index[index] for index, _, _ in admitted], skipped, estimated


def _load_resume(path: Path, strategies: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("experiment") != "aladdin-helpsteer2-real-llm-cost-pilot":
        raise ValueError(f"cannot resume a different experiment: {path}")
    allowed = {(strategy, seed) for strategy in strategies for seed in seeds}
    result = []
    seen = set()
    for item in report.get("results", []):
        key = (str(item["strategy"]), int(item["seed"]))
        if key not in allowed or key in seen:
            raise ValueError(f"invalid or duplicate resume result: {key}")
        seen.add(key)
        result.append(item)
    return result


def run_cost_pilot(data_path: Path, *, output_path: Path, endpoint: str, model: str,
                   model_revision: str, seeds: tuple[int, ...], candidate_budget: int,
                   token_budget: int, seed_group_limit: int, pool_group_limit: int,
                   workers: int, timeout: float, resume: bool,
                   independent_run_splits: bool, strategies: tuple[str, ...] | None = None,
                   aladdin_score: str = "product", aladdin_novelty_weight: float = 0.5) -> dict[str, Any]:
    all_groups = load_helpsteer2_groups(data_path)
    usable = _pairwise_groups(all_groups)
    seed_pool = [group for group in usable if group.split == "seed"]
    feedback_pool = [group for group in usable if group.split == "held_in"]
    held_out_groups = [group for group in usable if group.split == "held_out"]
    if len(seed_pool) < seed_group_limit or len(feedback_pool) < max(pool_group_limit, candidate_budget) or not held_out_groups:
        raise ValueError("insufficient pairwise HelpSteer2 groups for requested configuration")

    strategies = tuple(strategies or (*STRATEGIES, SEED_ONLY_STRATEGY))
    allowed_strategies = {*SUPPORTED_STRATEGIES, SEED_ONLY_STRATEGY}
    invalid_strategies = set(strategies) - allowed_strategies
    if invalid_strategies or not strategies:
        raise ValueError(f"invalid strategies: {sorted(invalid_strategies)}")
    results = _load_resume(output_path, strategies, seeds) if resume else []
    completed_keys = {(str(item["strategy"]), int(item["seed"])) for item in results}
    total_runs = len(strategies) * len(seeds)
    completed_runs = len(results)
    started_at = time.monotonic()
    print(
        f"helpsteer2 cost-pilot start {_progress(completed_runs, total_runs)} "
        f"workers={workers} token_budget={token_budget} usable_groups={len(usable)} resume={resume}",
        flush=True,
    )

    for seed in seeds:
        seed_groups = _groups_for_run(
            usable,
            split="seed",
            limit=seed_group_limit,
            seed=seed,
            salt=17,
            independent_run_splits=independent_run_splits,
        )
        pool_groups = _groups_for_run(
            usable,
            split="held_in",
            limit=pool_group_limit,
            seed=seed,
            salt=31,
            independent_run_splits=independent_run_splits,
        )
        held_out = [
            record
            for group in held_out_groups
            for record in group.release_feedback()
        ]
        if len(seed_groups) < 2 or len(pool_groups) < candidate_budget:
            raise ValueError(f"insufficient groups for seed={seed}")
        seed_records = [record for group in seed_groups for record in group.release_feedback()]
        for strategy in strategies:
            if (strategy, seed) in completed_keys:
                print(f"resume skip strategy={strategy} seed={seed}", flush=True)
                continue
            if strategy == SEED_ONLY_STRATEGY:
                selected = []
                diagnostics = {"strategy": strategy, "selected_group_ids": [], "selection_definition": "seed-only"}
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
            judgments, skipped, estimated = _run_judgments(
                selected,
                seed=seed,
                endpoint=endpoint,
                model=model,
                timeout=timeout,
                token_budget=token_budget,
                workers=workers,
                strategy=strategy,
                completed_runs=completed_runs,
                total_runs=total_runs,
                started_at=started_at,
            )
            selected_by_id = {group.group_id: group for group in selected}
            feedback_records = [record for judgment in judgments for record in _relabel(selected_by_id[judgment["group_id"]], judgment)]
            llm_checkpoint = _train([*seed_records, *feedback_records], seed)
            oracle_records = [record for judgment in judgments for record in selected_by_id[judgment["group_id"]].release_feedback()]
            oracle_checkpoint = _train([*seed_records, *oracle_records], seed)
            llm_metrics = _evaluate(llm_checkpoint, held_out)
            oracle_metrics = _evaluate(oracle_checkpoint, held_out)
            feedback_accuracy = (
                sum(bool(item["correct"]) for item in judgments) / len(judgments)
                if judgments else None
            )
            results.append({
                "strategy": strategy,
                "seed": seed,
                "metrics": {
                    "feedback_accuracy": feedback_accuracy,
                    "feedback_groups": len(judgments),
                    "llm_feedback_task_model": llm_metrics,
                    "oracle_feedback_task_model": oracle_metrics,
                    "llm_minus_oracle_pairwise": llm_metrics["pairwise_accuracy"] - oracle_metrics["pairwise_accuracy"],
                },
                "budget": {
                    "candidate_budget_groups": candidate_budget,
                    "token_budget": token_budget,
                    "feedback_requests": len(judgments),
                    "skipped_groups": skipped,
                    "llm_calls": len(judgments),
                    "llm_tokens": sum(int(item["usage"]["total_tokens"]) for item in judgments),
                    "estimated_prompt_tokens": estimated,
                    "workers": min(workers, len(judgments)) if judgments else 0,
                },
                "diagnostics": diagnostics,
                "run_split": {
                    "independent_run_splits": independent_run_splits,
                    "seed_groups": len(seed_groups),
                    "feedback_pool_groups": len(pool_groups),
                    "held_out_groups": len(held_out_groups),
                    "seed_group_ids_hash": hashlib.sha256(
                        "\n".join(group.group_id for group in seed_groups).encode("utf-8")
                    ).hexdigest(),
                    "feedback_pool_group_ids_hash": hashlib.sha256(
                        "\n".join(group.group_id for group in pool_groups).encode("utf-8")
                    ).hexdigest(),
                },
                "model": {"id": model, "revision": model_revision},
                "judgments": judgments,
            })
            completed_keys.add((strategy, seed))
            completed_runs += 1
            _write_atomic(output_path, {
                "experiment": "aladdin-helpsteer2-real-llm-cost-pilot",
                "status": "running",
                "version": VERSION,
                "dataset": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "configuration": {
                    "seeds": list(seeds),
                    "strategies": list(strategies),
                    "candidate_budget_groups": candidate_budget,
                    "token_budget_per_run": token_budget,
                    "seed_group_limit": seed_group_limit,
                    "pool_group_limit": pool_group_limit,
                    "usable_pairwise_groups": len(usable),
                    "excluded_non_pairwise_groups": len(all_groups) - len(usable),
                    "workers": workers,
                    "independent_run_splits": independent_run_splits,
                    "aladdin_score": aladdin_score,
                    "aladdin_novelty_weight": aladdin_novelty_weight,
                },
                "completed_runs": completed_runs,
                "total_runs": total_runs,
                "results": results,
            })
            elapsed = time.monotonic() - started_at
            eta = elapsed / completed_runs * max(total_runs - completed_runs, 0)
            print(
                f"{_progress(completed_runs, total_runs)} strategy={strategy} seed={seed} "
                f"groups={len(judgments)} tokens={sum(int(item['usage']['total_tokens']) for item in judgments)} "
                f"pairwise={llm_metrics['pairwise_accuracy']:.4f} elapsed={_duration(elapsed)} eta~{_duration(eta)}",
                flush=True,
            )

    summary = {}
    for strategy in strategies:
        values = [item for item in results if item["strategy"] == strategy]
        feedback = [item["metrics"]["feedback_accuracy"] for item in values if item["metrics"]["feedback_accuracy"] is not None]
        summary[strategy] = {
            "mean_feedback_accuracy": sum(feedback) / len(feedback) if feedback else None,
            "mean_feedback_groups": sum(item["metrics"]["feedback_groups"] for item in values) / len(values),
            "mean_llm_pairwise_accuracy": sum(item["metrics"]["llm_feedback_task_model"]["pairwise_accuracy"] for item in values) / len(values),
            "mean_oracle_pairwise_accuracy": sum(item["metrics"]["oracle_feedback_task_model"]["pairwise_accuracy"] for item in values) / len(values),
            "mean_llm_tokens": sum(item["budget"]["llm_tokens"] for item in values) / len(values),
        }
    report = {
        "experiment": "aladdin-helpsteer2-real-llm-cost-pilot",
        "status": "completed",
        "version": VERSION,
        "smoke_protocol_version": SMOKE_VERSION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_path": str(data_path),
        "final_test": "not loaded",
        "model": {"id": model, "revision": model_revision, "endpoint": endpoint},
        "configuration": {
            "seeds": list(seeds),
            "strategies": list(strategies),
            "candidate_budget_groups": candidate_budget,
            "token_budget_per_run": token_budget,
            "seed_group_limit": seed_group_limit,
            "pool_group_limit": pool_group_limit,
            "usable_pairwise_groups": len(usable),
            "excluded_non_pairwise_groups": len(all_groups) - len(usable),
            "workers": workers,
            "independent_run_splits": independent_run_splits,
            "task_model": "tfidf-logistic-regression-v1",
            "aladdin_score": aladdin_score,
            "aladdin_novelty_weight": aladdin_novelty_weight,
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "Only standard two-candidate groups with one preferred and one rejected label are included.",
            "The task model is TF-IDF; this is not Qwen fine-tuning or DPO.",
            "HelpSteer2 human scores are hidden from selection and released only through selected groups.",
            "Official validation/final_test records are not loaded.",
        ],
    }
    _write_atomic(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aladdin HelpSteer2 with a real LLM feedback provider.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=32000)
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--pool-group-limit", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--aladdin-score", choices=("product", "additive"), default="product")
    parser.add_argument("--aladdin-novelty-weight", type=float, default=0.5)
    parser.add_argument(
        "--independent-run-splits",
        action="store_true",
        help="resample seed and held-in groups independently for each run seed",
    )
    args = parser.parse_args()
    report = run_cost_pilot(
        args.data,
        output_path=args.output,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=args.model_revision,
        seeds=tuple(args.seeds),
        candidate_budget=args.candidate_budget,
        token_budget=args.token_budget,
        seed_group_limit=args.seed_group_limit,
        pool_group_limit=args.pool_group_limit,
        workers=args.workers,
        timeout=args.timeout,
        resume=args.resume,
        independent_run_splits=args.independent_run_splits,
        strategies=tuple(args.strategies) if args.strategies else None,
        aladdin_score=args.aladdin_score,
        aladdin_novelty_weight=args.aladdin_novelty_weight,
    )
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
