from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from core_system.mle_core.models import TaskRecord
from local_experiments.aladdin.helpsteer2_feedback import (
    STRATEGIES,
    _pairwise_metrics,
    select_feedback_groups,
)


DATASET_ID = "Anthropic/hh-rlhf"
DATASET_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
ADAPTER_VERSION = "aladdin-hh-rlhf-feedback-v3"
SPLITS = ("seed", "held_in", "held_out", "final_test")
SPLIT_UNITS = ("pair", "prompt")


class HHRLHFAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class HHRLHFGroup:
    group_id: str
    prompt: str
    split: str
    batch_id: str
    hidden_records: tuple[TaskRecord, ...]
    released_records: tuple[TaskRecord, ...]
    labels: tuple[str, ...]

    def release_feedback(self) -> tuple[TaskRecord, ...]:
        """Release the pairwise preference only after group selection."""
        return self.released_records


def _stable_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HHRLHFAdapterError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise HHRLHFAdapterError(f"{path}:{line_number} is not an object")
            yield value


def _split_last_turn(conversation: str) -> tuple[str, str]:
    """Split one conversation when its final answer contains no ambiguous turn marker."""
    human_marker = "\n\nHuman:"
    assistant_marker = "\n\nAssistant:"
    human_index = conversation.rfind(human_marker)
    if human_index < 0 and conversation.startswith("Human:"):
        human_index = 0
    if human_index < 0:
        raise HHRLHFAdapterError("HH-RLHF row has no final Human turn")
    marker_index = conversation.find(assistant_marker, human_index)
    if marker_index < 0:
        raise HHRLHFAdapterError("HH-RLHF row has no final Assistant turn")
    history = conversation[:marker_index].strip()
    response = conversation[marker_index + len(assistant_marker):].strip()
    if not history:
        raise HHRLHFAdapterError("HH-RLHF row has an empty conversation history")
    if not (history.startswith("Human:") or "\n\nHuman:" in history):
        raise HHRLHFAdapterError("HH-RLHF history has no Human turn")
    return history, response


def _split_pair_turns(chosen: str, rejected: str, *, source_line: int) -> tuple[str, str, str]:
    """Use the pair's common prefix to avoid treating answer text as dialogue turns."""
    common_length = 0
    for chosen_char, rejected_char in zip(chosen, rejected):
        if chosen_char != rejected_char:
            break
        common_length += 1
    common_prefix = chosen[:common_length]
    assistant_marker = "\n\nAssistant:"
    marker_index = common_prefix.rfind(assistant_marker)
    if marker_index < 0:
        raise HHRLHFAdapterError(
            f"{source_line}: chosen/rejected have no shared final Assistant boundary"
        )
    history = chosen[:marker_index].strip()
    chosen_response = chosen[marker_index + len(assistant_marker):].strip()
    rejected_response = rejected[marker_index + len(assistant_marker):].strip()
    if not history:
        raise HHRLHFAdapterError(f"{source_line}: chosen/rejected have an empty history")
    if not (history.startswith("Human:") or "\n\nHuman:" in history):
        raise HHRLHFAdapterError(f"{source_line}: shared history has no Human turn")
    return history, chosen_response, rejected_response


def _validate_split_unit(split_unit: str) -> str:
    if split_unit not in SPLIT_UNITS:
        raise HHRLHFAdapterError(
            f"split_unit must be one of {SPLIT_UNITS}, got {split_unit!r}"
        )
    return split_unit


def _split_for_key(key: str, seed: int) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8],
        "big",
    ) % 100
    if bucket < 20:
        return "seed"
    if bucket < 60:
        return "held_in"
    return "held_out"


def _split_for_group(group_id: str, seed: int) -> str:
    """Legacy pair-level split retained for historical experiment reports."""
    return _split_for_key(group_id, seed)


def _split_for_prompt(prompt: str, seed: int) -> str:
    return _split_for_key("prompt:" + prompt.strip(), seed)


def _metadata(
    *,
    group_id: str,
    released: bool,
    source_line: int,
    empty_response: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "adapter_version": ADAPTER_VERSION,
        "feedback_type": "human_pairwise_preference",
        "feedback_group_id": group_id,
        "feedback_released": released,
        "source_line": source_line,
        "empty_response": empty_response,
    }
    if released:
        metadata["feedback"] = {"label": "pairwise_preference", "source": "human_preference"}
    return metadata


def _record(
    *,
    group_id: str,
    split: str,
    prompt: str,
    response: str,
    target: str | None,
    candidate: str,
    source_line: int,
    released: bool,
) -> TaskRecord:
    return TaskRecord(
        record_id=f"{group_id}-{candidate}",
        task_id="hh-rlhf-response-preference",
        split=split,
        input=f"{prompt}\n\nResponse: {response}",
        target=target,
        source_group_id=group_id,
        metadata=_metadata(
            group_id=group_id,
            released=released,
            source_line=source_line,
            empty_response=response == "",
        ),
    )


def load_hh_rlhf_groups(
    path: Path,
    *,
    split_seed: int = 17,
    source_revision: str = DATASET_REVISION,
    split_unit: str = "pair",
) -> list[HHRLHFGroup]:
    """Load official HH-RLHF train rows and hide labels from the task model."""
    _validate_split_unit(split_unit)
    grouped: dict[str, HHRLHFGroup] = {}
    for source_line, row in enumerate(_load_jsonl(path), start=1):
        if set(row) != {"chosen", "rejected"} and not {"chosen", "rejected"}.issubset(row):
            raise HHRLHFAdapterError(
                f"{path}:{source_line} must contain chosen and rejected fields"
            )
        chosen = row["chosen"]
        rejected = row["rejected"]
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            raise HHRLHFAdapterError(f"{path}:{source_line} chosen/rejected must be strings")
        try:
            shared_prompt, chosen_response, rejected_response = _split_pair_turns(
                chosen, rejected, source_line=source_line
            )
        except HHRLHFAdapterError as exc:
            raise HHRLHFAdapterError(f"{path}:{source_line} {exc}") from exc
        pair_key = chosen + "\0" + rejected
        group_id = "hh-rlhf-" + _stable_hex(pair_key)[:24]
        if group_id in grouped:
            raise HHRLHFAdapterError(f"duplicate HH-RLHF pair at {path}:{source_line}")
        split = (
            _split_for_group(group_id, split_seed)
            if split_unit == "pair"
            else _split_for_prompt(shared_prompt, split_seed)
        )
        hidden = (
            _record(
                group_id=group_id,
                split=split,
                prompt=shared_prompt,
                response=chosen_response,
                target=None,
                candidate="chosen",
                source_line=source_line,
                released=False,
            ),
            _record(
                group_id=group_id,
                split=split,
                prompt=shared_prompt,
                response=rejected_response,
                target=None,
                candidate="rejected",
                source_line=source_line,
                released=False,
            ),
        )
        released = (
            _record(
                group_id=group_id,
                split=split,
                prompt=shared_prompt,
                response=chosen_response,
                target="preferred",
                candidate="chosen",
                source_line=source_line,
                released=True,
            ),
            _record(
                group_id=group_id,
                split=split,
                prompt=shared_prompt,
                response=rejected_response,
                target="rejected",
                candidate="rejected",
                source_line=source_line,
                released=True,
            ),
        )
        # Keep the revision in every record's provenance even though the raw row has no revision field.
        hidden = tuple(
            TaskRecord(
                record_id=record.record_id,
                task_id=record.task_id,
                split=record.split,
                input=record.input,
                target=record.target,
                source_group_id=record.source_group_id,
                metadata={**record.metadata, "source_revision": source_revision},
            )
            for record in hidden
        )
        released = tuple(
            TaskRecord(
                record_id=record.record_id,
                task_id=record.task_id,
                split=record.split,
                input=record.input,
                target=record.target,
                source_group_id=record.source_group_id,
                metadata={**record.metadata, "source_revision": source_revision},
            )
            for record in released
        )
        grouped[group_id] = HHRLHFGroup(
            group_id=group_id,
            prompt=shared_prompt,
            split=split,
            batch_id="hash-bucket-" + str(int(group_id[-2:], 16) % 16),
            hidden_records=hidden,
            released_records=released,
            labels=("preferred", "rejected"),
        )
    return [grouped[group_id] for group_id in sorted(grouped)]


def _train(records: Iterable[TaskRecord], seed: int) -> dict[str, Any]:
    from core_system.mle_core.sklearn_components import TfidfLogisticTrainer

    checkpoint, _ = TfidfLogisticTrainer().train(
        train_records=list(records),
        initial_checkpoint=None,
        config={"ngram_range": (1, 2), "max_iter": 200, "steps": 1},
        seed=seed,
        store=None,
    )
    return checkpoint


def _evaluate(checkpoint: dict[str, Any], records: list[TaskRecord]) -> dict[str, float]:
    from core_system.mle_core.sklearn_components import TfidfLogisticEvaluator

    metrics = TfidfLogisticEvaluator().evaluate(checkpoint=checkpoint, records=records, config={})[0]
    return {**metrics, **_pairwise_metrics(checkpoint, records)}


_WORKER_POOL_GROUPS: list[HHRLHFGroup] = []
_WORKER_SEED_RECORDS: list[TaskRecord] = []
_WORKER_HELD_OUT_RECORDS: list[TaskRecord] = []
_WORKER_FEEDBACK_BUDGET = 0
_WORKER_CONTEXTS: dict[int, tuple[list[HHRLHFGroup], list[TaskRecord], list[TaskRecord]]] = {}


def _set_worker_context(
    pool_groups: list[HHRLHFGroup],
    seed_records: list[TaskRecord],
    held_out_records: list[TaskRecord],
    feedback_budget: int,
) -> None:
    global _WORKER_POOL_GROUPS, _WORKER_SEED_RECORDS, _WORKER_HELD_OUT_RECORDS
    global _WORKER_FEEDBACK_BUDGET
    _WORKER_POOL_GROUPS = pool_groups
    _WORKER_SEED_RECORDS = seed_records
    _WORKER_HELD_OUT_RECORDS = held_out_records
    _WORKER_FEEDBACK_BUDGET = feedback_budget


def _set_worker_contexts(
    contexts: dict[int, tuple[list[HHRLHFGroup], list[TaskRecord], list[TaskRecord]]],
    feedback_budget: int,
) -> None:
    global _WORKER_CONTEXTS, _WORKER_FEEDBACK_BUDGET
    _WORKER_CONTEXTS = contexts
    _WORKER_FEEDBACK_BUDGET = feedback_budget


def _init_spawn_worker(
    pool_groups: list[HHRLHFGroup],
    seed_records: list[TaskRecord],
    held_out_records: list[TaskRecord],
    feedback_budget: int,
) -> None:
    _set_worker_context(pool_groups, seed_records, held_out_records, feedback_budget)


def _init_spawn_workers(
    contexts: dict[int, tuple[list[HHRLHFGroup], list[TaskRecord], list[TaskRecord]]],
    feedback_budget: int,
) -> None:
    _set_worker_contexts(contexts, feedback_budget)


def _run_one_hh_rlhf(
    task: tuple[int, int, str],
    *,
    pool_groups: list[HHRLHFGroup],
    seed_records: list[TaskRecord],
    held_out_records: list[TaskRecord],
    feedback_budget: int,
) -> dict[str, Any]:
    _, seed, strategy = task
    selected, diagnostics = select_feedback_groups(
        pool_groups,
        strategy=strategy,
        budget=feedback_budget,
        seed=seed,
        seed_records=seed_records,
    )
    released_records = [record for group in selected for record in group.release_feedback()]
    checkpoint = _train([*seed_records, *released_records], seed)
    return {
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
    }


def _run_worker_task(task: tuple[int, int, str, int]) -> dict[str, Any]:
    index, seed, strategy, split_seed = task
    pool_groups, seed_records, held_out_records = _WORKER_CONTEXTS[seed]
    result = _run_one_hh_rlhf(
        (index, seed, strategy),
        pool_groups=pool_groups,
        seed_records=seed_records,
        held_out_records=held_out_records,
        feedback_budget=_WORKER_FEEDBACK_BUDGET,
    )
    result["split_seed"] = split_seed
    result["split_counts"] = {
        "seed_groups": len(seed_records) // 2,
        "feedback_pool_groups": len(pool_groups),
        "held_out_records": len(held_out_records),
    }
    return result


def _copy_group_split(group: HHRLHFGroup, split: str) -> HHRLHFGroup:
    def update(record: TaskRecord) -> TaskRecord:
        return TaskRecord(
            record_id=record.record_id,
            task_id=record.task_id,
            split=split,
            input=record.input,
            target=record.target,
            source_group_id=record.source_group_id,
            metadata=record.metadata,
            parent_record_ids=record.parent_record_ids,
        )

    return HHRLHFGroup(
        group_id=group.group_id,
        prompt=group.prompt,
        split=split,
        batch_id=group.batch_id,
        hidden_records=tuple(update(record) for record in group.hidden_records),
        released_records=tuple(update(record) for record in group.released_records),
        labels=group.labels,
    )


def _groups_for_split_seed(
    groups: list[HHRLHFGroup], split_seed: int, split_unit: str = "pair"
) -> list[HHRLHFGroup]:
    _validate_split_unit(split_unit)
    return [
        _copy_group_split(
            group,
            (
                _split_for_group(group.group_id, split_seed)
                if split_unit == "pair"
                else _split_for_prompt(group.prompt, split_seed)
            ),
        )
        for group in groups
    ]


def split_audit(groups: list[HHRLHFGroup]) -> dict[str, Any]:
    """Summarize whether identical prompts cross the train/eval split boundary."""
    prompt_splits: dict[str, set[str]] = {}
    for group in groups:
        prompt_splits.setdefault(group.prompt.strip(), set()).add(group.split)
    overlapping = [
        prompt for prompt, splits in prompt_splits.items() if len(splits) > 1
    ]
    group_counts = {split: 0 for split in SPLITS}
    prompt_counts = {split: 0 for split in SPLITS}
    for group in groups:
        group_counts[group.split] = group_counts.get(group.split, 0) + 1
    for splits in prompt_splits.values():
        for split in splits:
            prompt_counts[split] = prompt_counts.get(split, 0) + 1
    return {
        "groups": len(groups),
        "unique_prompts": len(prompt_splits),
        "prompts_with_multiple_splits": len(overlapping),
        "max_splits_per_prompt": max(
            (len(splits) for splits in prompt_splits.values()), default=0
        ),
        "group_counts": group_counts,
        "prompt_counts": prompt_counts,
    }


def run_hh_rlhf_feedback_experiment(
    data_path: Path,
    *,
    strategies: tuple[str, ...] = STRATEGIES,
    seeds: tuple[int, ...] = (17, 18, 19, 20, 21),
    split_seed: int = 17,
    seed_group_limit: int = 128,
    feedback_budget: int = 20,
    pool_group_limit: int = 512,
    final_test_path: Path | None = None,
    workers: int = 1,
    split_unit: str = "pair",
    fail_on_split_overlap: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    _validate_split_unit(split_unit)
    groups = load_hh_rlhf_groups(data_path, split_seed=split_seed, split_unit=split_unit)
    if not seeds:
        raise ValueError("seeds must not be empty")
    contexts: dict[int, tuple[list[HHRLHFGroup], list[TaskRecord], list[TaskRecord]]] = {}
    split_seed_by_seed: dict[str, int] = {}
    split_counts_by_seed: dict[str, dict[str, Any]] = {}
    for offset, seed in enumerate(seeds):
        run_split_seed = split_seed + offset
        split_seed_by_seed[str(seed)] = run_split_seed
        run_groups = _groups_for_split_seed(groups, run_split_seed, split_unit)
        audit = split_audit(run_groups)
        if fail_on_split_overlap and audit["prompts_with_multiple_splits"]:
            raise HHRLHFAdapterError(
                f"prompt split overlap detected for split_seed={run_split_seed}: {audit}"
            )
        split_counts_by_seed[str(seed)] = audit
        seed_groups = [group for group in run_groups if group.split == "seed"][:seed_group_limit]
        pool_groups = [group for group in run_groups if group.split == "held_in"][:pool_group_limit]
        held_out_records = [
            record for group in run_groups
            if group.split == "held_out"
            for record in group.released_records
        ]
        if len(seed_groups) < 2 or len(pool_groups) < feedback_budget or not held_out_records:
            raise HHRLHFAdapterError(
                f"insufficient groups for seed={seed} split_seed={run_split_seed}"
            )
        seed_records = [record for group in seed_groups for record in group.release_feedback()]
        contexts[seed] = (pool_groups, seed_records, held_out_records)
        split_counts_by_seed[str(seed)].update({
            "seed_groups": len(seed_groups),
            "feedback_pool_groups": len(pool_groups),
            "held_out_records": len(held_out_records),
        })
    task_specs: list[tuple[int, int, str, int]] = []
    for seed in seeds:
        for strategy in strategies:
            task_specs.append((len(task_specs), seed, strategy, split_seed_by_seed[str(seed)]))
    total_runs = len(task_specs)
    workers = min(workers, total_runs)
    results_by_index: dict[int, dict[str, Any]] = {}
    completed_runs = 0
    started_at = time.monotonic()

    def record_result(index: int, result: dict[str, Any]) -> None:
        nonlocal completed_runs
        results_by_index[index] = result
        completed_runs += 1
        elapsed = time.monotonic() - started_at
        eta = elapsed / completed_runs * (total_runs - completed_runs)
        print(
            f"[{completed_runs}/{total_runs}] dataset=hh-rlhf strategy={result['strategy']} seed={result['seed']} "
            f"pairwise={result['metrics']['pairwise_accuracy']:.4f} "
            f"workers={workers} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    if workers == 1:
        for task in task_specs:
            _, seed, _, run_split_seed = task
            pool_groups, seed_records, held_out_records = contexts[seed]
            result = _run_one_hh_rlhf(
                (task[0], task[1], task[2]),
                pool_groups=pool_groups,
                seed_records=seed_records,
                held_out_records=held_out_records,
                feedback_budget=feedback_budget,
            )
            result["split_seed"] = run_split_seed
            result["split_counts"] = split_counts_by_seed[str(seed)]
            record_result(
                task[0],
                result,
            )
    else:
        _set_worker_contexts(contexts, feedback_budget)
        try:
            mp_context = multiprocessing.get_context("fork")
            executor_kwargs: dict[str, Any] = {"mp_context": mp_context}
        except ValueError:
            executor_kwargs = {
                "initializer": _init_spawn_workers,
                "initargs": (contexts, feedback_budget),
            }
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, **executor_kwargs) as executor:
            future_to_index = {
                executor.submit(_run_worker_task, task): task[0]
                for task in task_specs
            }
            for future in concurrent.futures.as_completed(future_to_index):
                record_result(future_to_index[future], future.result())

    results = [results_by_index[index] for index, _, _, _ in task_specs]
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
        "experiment": "aladdin-hh-rlhf-technique-1-feedback-selection",
        "status": "server-replay-mechanism-experiment",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "adapter_version": ADAPTER_VERSION,
        "data_path": str(data_path),
        "final_test_path": str(final_test_path) if final_test_path else None,
        "split_protocol": {
            "split_unit": split_unit,
            "base_split_seed": split_seed,
            "split_seed_by_seed": split_seed_by_seed,
            "seed": f"stable hash bucket 0-19 per run split seed, limit {seed_group_limit}",
            "feedback_pool": f"stable hash bucket 20-59 per run split seed, limit {pool_group_limit}",
            "held_out": "stable hash bucket 60-99 per run split seed",
            "final_test": "official Anthropic/hh-rlhf test split; path recorded only and not loaded",
            "fail_on_split_overlap": fail_on_split_overlap,
        },
        "configuration": {
            "seeds": list(seeds),
            "feedback_budget_groups": feedback_budget,
            "split_counts_by_seed": split_counts_by_seed,
            "strategies": list(strategies),
            "workers": workers,
            "backbone_llm": "not used in local mechanism experiment",
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "HH-RLHF chosen/rejected labels are replayed as hidden human preference feedback; no new annotation is requested.",
            "The local task model is deterministic TF-IDF and is not an LLM backbone or an RLHF trainer.",
            "Official test is sealed and is not loaded by this command.",
            "Pairwise accuracy measures preference selection in the held-out replay set, not general helpfulness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aladdin's HH-RLHF pairwise feedback replay.")
    parser.add_argument("--data", type=Path, required=True, help="Official train JSONL with chosen/rejected fields")
    parser.add_argument("--test-data", type=Path, default=None, help="Sealed test path; recorded but never opened")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback-budget", type=int, default=20)
    parser.add_argument("--seed-group-limit", type=int, default=128)
    parser.add_argument("--pool-group-limit", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--workers", type=int, default=1, help="Parallel independent runs; use 4 on the server first")
    args = parser.parse_args()
    report = run_hh_rlhf_feedback_experiment(
        args.data,
        feedback_budget=args.feedback_budget,
        seed_group_limit=args.seed_group_limit,
        pool_group_limit=args.pool_group_limit,
        split_seed=args.split_seed,
        seeds=tuple(args.seeds),
        final_test_path=args.test_data,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
