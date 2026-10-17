from __future__ import annotations

"""Paired HH-RLHF diagnostic separating selection quality from label noise."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from local_experiments.aladdin.hh_rlhf_qwen_lora_pilot import (
    SEED_ONLY,
    _pairwise_accuracy,
    _pairwise_uncertainty,
    _payload,
)
from local_experiments.aladdin.hh_rlhf_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    HHRLHFGroup,
    SPLIT_UNITS,
    _groups_for_split_seed,
    load_hh_rlhf_groups,
    select_feedback_groups,
    split_audit,
)
from local_experiments.aladdin.helpsteer2_feedback import (
    _group_novelty_tokens,
    _tokens,
)
from local_experiments.aladdin.hh_rlhf_llm_training_pilot import _judgment, _llm_records
from local_experiments.aladdin.hh_rlhf_llm_smoke import SMOKE_VERSION
from local_experiments.aladdin.qwen_preference import train_checkpoint


DEFAULT_STRATEGIES = (SEED_ONLY, "random", "aladdin-v2")
QWEN_ALADDIN = "aladdin-qwen"
QWEN_ALADDIN_RELIABLE = "aladdin-qwen-reliable"
QWEN_STRATEGIES = {QWEN_ALADDIN, QWEN_ALADDIN_RELIABLE}
SUPPORTED_STRATEGIES = {*DEFAULT_STRATEGIES, *QWEN_STRATEGIES}
FORMAL_PROTOCOL_VERSION = "formal-hh-rlhf-qwen-v1"
FORMAL_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
FORMAL_CONFIGURATION = {
    "seeds": tuple(range(17, 27)),
    "strategies": DEFAULT_STRATEGIES,
    "split_seed": 17,
    "split_unit": "prompt",
    "fail_on_split_overlap": True,
    "vary_split_by_seed": False,
    "candidate_budget": 32,
    "seed_group_limit": 64,
    "pool_group_limit": 256,
    "eval_group_limit": 512,
    "eval_batch_size": 8,
    "max_steps": 100,
    "seed_only_max_steps": None,
    "feedback_max_steps": None,
    "training_budget_mode": "equal-epochs",
    "batch_size": 2,
    "max_length": 1024,
    "learning_rate": 2e-4,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "max_gpu_memory_gib": 60.0,
    "timeout_seconds": 300.0,
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_revision": FORMAL_MODEL_REVISION,
}


def _sample_groups(
    groups: list[HHRLHFGroup], limit: int, seed: int
) -> list[HHRLHFGroup]:
    if limit <= 0 or limit >= len(groups):
        return list(groups)
    return sorted(random.Random(seed).sample(groups, limit), key=lambda group: group.group_id)


def _rank_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _select_qwen_feedback_groups(
    groups: list[HHRLHFGroup],
    *,
    budget: int,
    seed_records: list[Any],
    selector_checkpoint: dict[str, Any],
    eval_batch_size: int,
    strategy: str = QWEN_ALADDIN,
) -> tuple[list[HHRLHFGroup], dict[str, Any]]:
    """Select feedback with the seed-trained Qwen selector.

    The original strategy ranks extreme uncertainty highest. The reliable variant
    keeps informative middle-uncertainty groups and penalizes the most difficult
    cases, whose feedback accuracy was lower in the diagnostic audit.
    """
    if strategy not in QWEN_STRATEGIES:
        raise ValueError(f"unsupported Qwen selector strategy: {strategy}")
    uncertainty = _pairwise_uncertainty(
        selector_checkpoint,
        groups,
        batch_size=eval_batch_size,
    )
    seed_token_sets = [
        _tokens(str(record.input).split("\nResponse:", 1)[0])
        for record in seed_records
    ]
    group_token_sets = {group.group_id: _tokens(group.prompt) for group in groups}
    uncertainty_ranks = _rank_normalize([uncertainty[group.group_id] for group in groups])
    ranked_uncertainty = {
        group.group_id: rank for group, rank in zip(groups, uncertainty_ranks)
    }
    target_uncertainty_rank = 0.35

    def acquisition_value(group_id: str) -> float:
        rank = ranked_uncertainty[group_id]
        if strategy == QWEN_ALADDIN:
            return rank
        # Prefer informative but not extreme uncertainty. The target is fixed
        # before the follow-up run and is independent of held-in labels.
        return max(
            0.0,
            1.0 - abs(rank - target_uncertainty_rank) / (1.0 - target_uncertainty_rank),
        )

    remaining = list(groups)
    selected: list[HHRLHFGroup] = []
    context_token_sets = list(seed_token_sets)
    selected_novelty: dict[str, float] = {}
    selected_scores: dict[str, float] = {}
    while remaining and len(selected) < budget:
        candidates = []
        for group in remaining:
            novelty = _group_novelty_tokens(
                group_token_sets[group.group_id], context_token_sets
            )
            acquisition = acquisition_value(group.group_id)
            score = 0.75 * acquisition + 0.25 * novelty
            candidates.append((score, acquisition, novelty, group))
        score, _, novelty, winner = max(
            candidates,
            key=lambda item: (item[0], item[1], item[3].group_id),
        )
        selected.append(winner)
        remaining.remove(winner)
        context_token_sets.append(group_token_sets[winner.group_id])
        selected_novelty[winner.group_id] = novelty
        selected_scores[winner.group_id] = score

    return selected, {
        "strategy": strategy,
        "feedback_budget_groups": budget,
        "pool_groups": len(groups),
        "selected_group_ids": [group.group_id for group in selected],
        "selected_qwen_uncertainty": {
            group.group_id: uncertainty[group.group_id] for group in selected
        },
        "selected_novelty": selected_novelty,
        "selected_score": selected_scores,
        "selection_definition": (
            (
                "greedy selection using 75% rank-normalized Qwen pairwise uncertainty "
                "and 25% prompt diversity; Qwen selector is trained on seed records"
                if strategy == QWEN_ALADDIN
                else "greedy selection using 75% target-centered Qwen uncertainty value "
                "and 25% prompt diversity; target uncertainty rank=0.35; "
                "Qwen selector is trained on seed records"
            )
        ),
        "target_uncertainty_rank": target_uncertainty_rank if strategy == QWEN_ALADDIN_RELIABLE else None,
        "selector_checkpoint": {
            "adapter_hash": selector_checkpoint["adapter_hash"],
            "training_steps": selector_checkpoint["training_steps"],
            "seed": selector_checkpoint["seed"],
        },
    }


def _save(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary(results: list[dict[str, Any]], strategies: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in strategies:
        rows = [row for row in results if row["strategy"] == strategy]
        if not rows:
            continue
        summary[strategy] = {
            "runs": len(rows),
            "mean_feedback_accuracy": _mean(
                [row["feedback_accuracy"] for row in rows if row["feedback_accuracy"] is not None]
            ),
            "mean_llm_pairwise_accuracy": _mean(
                [row["metrics"]["llm_feedback"]["pairwise_accuracy"] for row in rows]
            ),
            "mean_oracle_pairwise_accuracy": _mean(
                [row["metrics"]["oracle_feedback"]["pairwise_accuracy"] for row in rows]
            ),
        }
    return summary


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_formal_protocol(args: argparse.Namespace) -> None:
    if args.protocol_version != FORMAL_PROTOCOL_VERSION:
        return
    mismatches = []
    for name, expected in FORMAL_CONFIGURATION.items():
        actual = tuple(getattr(args, name)) if name in {"seeds", "strategies"} else getattr(args, name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (required {expected!r})")
    base_model_revision = Path(args.base_model.rstrip("/\\")).name
    if base_model_revision != FORMAL_MODEL_REVISION:
        mismatches.append(
            f"base-model snapshot={base_model_revision!r} "
            f"(required {FORMAL_MODEL_REVISION!r})"
        )
    if mismatches:
        raise ValueError(
            "formal protocol parameters are frozen; mismatches: " + "; ".join(mismatches)
        )


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "peft", "vllm", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _training_steps(
    args: argparse.Namespace,
    strategy: str,
    *,
    record_count: int | None = None,
    reference_record_count: int | None = None,
) -> int:
    """Return the configured optimization budget for one strategy family."""
    configured = args.max_steps
    if strategy == SEED_ONLY and args.seed_only_max_steps is not None:
        configured = args.seed_only_max_steps
    elif strategy != SEED_ONLY and args.feedback_max_steps is not None:
        configured = args.feedback_max_steps
    if getattr(args, "training_budget_mode", "fixed-steps") == "equal-epochs":
        if not record_count or not reference_record_count:
            raise ValueError("equal-epochs requires training record counts")
        configured = (configured * record_count + reference_record_count - 1) // reference_record_count
    return max(1, configured)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_formal_protocol(args)
    strategies = tuple(args.strategies)
    seeds = tuple(args.seeds)
    if not strategies or not seeds:
        raise ValueError("strategies and seeds must not be empty")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if any(strategy not in SUPPORTED_STRATEGIES for strategy in strategies):
        raise ValueError(f"supported strategies are {tuple(sorted(SUPPORTED_STRATEGIES))}")
    if args.candidate_budget <= 0:
        raise ValueError("candidate budget must be positive")
    for name in (
        "max_steps",
        "seed_only_max_steps",
        "feedback_max_steps",
        "eval_batch_size",
        "batch_size",
        "max_length",
        "lora_r",
        "lora_alpha",
        "max_gpu_memory_gib",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")

    groups = load_hh_rlhf_groups(
        Path(args.data), split_seed=args.split_seed, split_unit=args.split_unit
    )
    output = Path(args.output)
    adapters_root = output.parent / (output.stem + "-adapters")
    config = {
        "base_model_path": args.base_model,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "per_device_batch_size": args.batch_size,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_gpu_memory_gib": args.max_gpu_memory_gib,
        "deterministic": True,
    }
    report: dict[str, Any] = {
        "experiment": "aladdin-hh-rlhf-qwen-lora-label-diagnostic-v2",
        "protocol_version": args.protocol_version,
        "status": "running",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": _sha256_file(Path(args.data)),
        "base_model_path": str(Path(args.base_model).resolve()),
        "base_model_revision": Path(args.base_model.rstrip("/\\")).name,
        "diagnostic": "Same selected pairs train one Qwen adapter with LLM labels and one with dataset oracle labels.",
        "oracle_note": "Oracle labels are diagnostic only and are not an available feedback method.",
        "inference_scope": (
            "Seeds are paired stochastic repeats on one fixed prompt-level split. "
            "Intervals describe between-seed variation conditional on this split and "
            "do not capture dataset/split sampling uncertainty."
        ),
        "configuration": {
            "protocol_version": args.protocol_version,
            "strategies": list(strategies),
            "seeds": list(seeds),
            "split_seed": args.split_seed,
            "split_unit": args.split_unit,
            "fail_on_split_overlap": args.fail_on_split_overlap,
            "vary_split_by_seed": args.vary_split_by_seed,
            "candidate_budget": args.candidate_budget,
            "seed_group_limit": args.seed_group_limit,
            "pool_group_limit": args.pool_group_limit,
            "eval_group_limit": args.eval_group_limit,
            "pool_sampling": "uniform sample from all held-in groups, shared across strategies per seed",
            "seed_sampling": "uniform sample from all seed groups, shared across strategies per seed",
            "eval_sampling": "uniform sample from all held-out groups, shared across strategies per seed",
            "llm_endpoint": args.endpoint,
            "llm_model": args.model,
            "llm_model_revision": args.model_revision,
            "llm_prompt_protocol": SMOKE_VERSION,
            "runtime_versions": _runtime_versions(),
            "training": config,
            "training_schedule": {
                "default_max_steps": args.max_steps,
                "seed_only_max_steps": args.seed_only_max_steps,
                "feedback_strategy_max_steps": args.feedback_max_steps,
                "budget_mode": args.training_budget_mode,
                "exposure_definition": (
                    "Optimizer steps are scaled by training record count from the "
                    "seed-only reference; exact consumed records are recorded per branch."
                ),
            },
            "reproducibility": {
                "deterministic_algorithms": True,
                "same_seed_reused_for_llm_and_oracle_branches": True,
            },
            "split_audit_by_seed": {},
        },
        "results": [],
        "summary": {},
    }
    _save(output, report)

    total = len(strategies) * len(seeds)
    for seed_index, seed in enumerate(seeds):
        run_split_seed = (
            args.split_seed + seed_index if args.vary_split_by_seed else args.split_seed
        )
        run_groups = _groups_for_split_seed(groups, run_split_seed, args.split_unit)
        audit = split_audit(run_groups)
        if args.fail_on_split_overlap and audit["prompts_with_multiple_splits"]:
            raise ValueError(
                f"prompt split overlap detected for seed={seed}, split_seed={run_split_seed}: {audit}"
            )
        report["configuration"]["split_audit_by_seed"][str(seed)] = {
            "split_seed": run_split_seed,
            **audit,
        }
        _save(output, report)
        seed_groups = _sample_groups(
            [group for group in run_groups if group.split == "seed"],
            args.seed_group_limit,
            run_split_seed + 1103,
        )
        pool_groups = _sample_groups(
            [group for group in run_groups if group.split == "held_in"],
            args.pool_group_limit,
            run_split_seed + 2207,
        )
        held_out_groups = _sample_groups(
            [group for group in run_groups if group.split == "held_out"],
            args.eval_group_limit,
            run_split_seed + 3301,
        )
        if len(seed_groups) < 2 or len(pool_groups) < args.candidate_budget or not held_out_groups:
            raise ValueError(f"insufficient groups for seed={seed} split_seed={run_split_seed}")
        seed_records = [record for group in seed_groups for record in group.release_feedback()]

        qwen_selector_checkpoint: dict[str, Any] | None = None
        qwen_selector_cost: dict[str, Any] | None = None
        if QWEN_STRATEGIES.intersection(strategies):
            selector_dir = adapters_root / f"{QWEN_ALADDIN}-selector-seed{seed}"
            qwen_selector_checkpoint, qwen_selector_cost = train_checkpoint(
                [_payload(record) for record in seed_records],
                config=config,
                seed=seed,
                output_dir=selector_dir,
            )

        for strategy in strategies:
            started = time.perf_counter()
            judgments: list[dict[str, Any]] = []
            if strategy == SEED_ONLY:
                selected: list[HHRLHFGroup] = []
                diagnostics: dict[str, Any] = {
                    "strategy": SEED_ONLY,
                    "selection_definition": "fixed seed records without additional feedback",
                    "selected_group_ids": [],
                }
                llm_records: list[Any] = []
                oracle_records: list[Any] = []
                feedback_accuracy = None
            elif strategy in QWEN_STRATEGIES:
                if qwen_selector_checkpoint is None:
                    raise RuntimeError("Qwen selector checkpoint was not initialized")
                selected, diagnostics = _select_qwen_feedback_groups(
                    pool_groups,
                    budget=args.candidate_budget,
                    seed_records=seed_records,
                    selector_checkpoint=qwen_selector_checkpoint,
                    eval_batch_size=args.eval_batch_size,
                    strategy=strategy,
                )
                llm_records = []
                oracle_records = []
                feedback_accuracy = None
                for index, group in enumerate(selected):
                    judgment = _judgment(
                        group=group,
                        seed=seed,
                        group_index=index,
                        endpoint=args.endpoint,
                        model=args.model,
                        timeout_seconds=args.timeout_seconds,
                    )
                    judgments.append(judgment)
                llm_records = [
                    record
                    for group, judgment in zip(selected, judgments)
                    for record in _llm_records(group, judgment)
                ]
                oracle_records = [
                    record for group in selected for record in group.release_feedback()
                ]
                feedback_accuracy = (
                    sum(bool(item["correct"]) for item in judgments) / len(judgments)
                    if judgments
                    else None
                )
            else:
                selected, diagnostics = select_feedback_groups(
                    pool_groups,
                    strategy=strategy,
                    budget=args.candidate_budget,
                    seed=seed,
                    seed_records=seed_records,
                )
                for index, group in enumerate(selected):
                    judgment = _judgment(
                        group=group,
                        seed=seed,
                        group_index=index,
                        endpoint=args.endpoint,
                        model=args.model,
                        timeout_seconds=args.timeout_seconds,
                    )
                    judgments.append(judgment)
                llm_records = [
                    record
                    for group, judgment in zip(selected, judgments)
                    for record in _llm_records(group, judgment)
                ]
                oracle_records = [
                    record for group in selected for record in group.release_feedback()
                ]
                feedback_accuracy = (
                    sum(bool(item["correct"]) for item in judgments) / len(judgments)
                    if judgments
                    else None
                )

            branch_records = {
                "llm_feedback": [*seed_records, *llm_records],
                "oracle_feedback": [*seed_records, *oracle_records],
            }
            training_record_count = len(branch_records["llm_feedback"])
            strategy_max_steps = _training_steps(
                args,
                strategy,
                record_count=training_record_count,
                reference_record_count=len(seed_records),
            )
            strategy_config = {**config, "max_steps": strategy_max_steps}
            branch_metrics: dict[str, Any] = {}
            branch_costs: dict[str, Any] = {}
            branch_checkpoints: dict[str, dict[str, Any]] = {}
            for channel, records in branch_records.items():
                adapter_dir = adapters_root / f"{strategy}-{channel}-seed{seed}"
                checkpoint, cost = train_checkpoint(
                    [_payload(record) for record in records],
                    config=strategy_config,
                    seed=seed,
                    output_dir=adapter_dir,
                )
                branch_metrics[channel] = _pairwise_accuracy(
                    checkpoint,
                    held_out_groups,
                    batch_size=args.eval_batch_size,
                )
                branch_costs[channel] = cost
                branch_checkpoints[channel] = {
                    "adapter_hash": checkpoint["adapter_hash"],
                    "training_steps": checkpoint["training_steps"],
                    "seed": checkpoint["seed"],
                }

            row = {
                "strategy": strategy,
                "seed": seed,
                "split_seed": run_split_seed,
                "feedback_accuracy": feedback_accuracy,
                "selected_groups": len(selected),
                "selected_group_ids": [group.group_id for group in selected],
                "training_records": training_record_count,
                "training_max_steps": strategy_max_steps,
                "training_records_seen": {
                    channel: branch_costs[channel].get("training_records_seen")
                    for channel in branch_costs
                },
                "training_record_passes": {
                    channel: (
                        branch_costs[channel]["training_records_seen"] / len(records)
                        if branch_costs[channel].get("training_records_seen") is not None
                        and records
                        else None
                    )
                    for channel, records in branch_records.items()
                },
                "diagnostics": diagnostics,
                "judgments": judgments,
                "metrics": branch_metrics,
                "reproducibility": {
                    "same_training_records": strategy == SEED_ONLY,
                    "branch_checkpoints": branch_checkpoints,
                    "identical_seed_only_adapters": (
                        strategy == SEED_ONLY
                        and branch_checkpoints["llm_feedback"]["adapter_hash"]
                        == branch_checkpoints["oracle_feedback"]["adapter_hash"]
                    ),
                    "seed_only_metric_delta": (
                        branch_metrics["llm_feedback"]["pairwise_accuracy"]
                        - branch_metrics["oracle_feedback"]["pairwise_accuracy"]
                        if strategy == SEED_ONLY
                        else None
                    ),
                },
                "cost": {
                    "branches": branch_costs,
                    "wall_seconds": time.perf_counter() - started,
                    "llm_calls": len(judgments),
                    "llm_tokens": sum(item["usage"]["total_tokens"] for item in judgments),
                    "qwen_selector": qwen_selector_cost if strategy in QWEN_STRATEGIES else None,
                },
            }
            report["results"].append(row)
            report["summary"] = _summary(report["results"], strategies)
            _save(output, report)
            completed = len(report["results"])
            print(
                f"completed {completed}/{total} strategy={strategy} seed={seed} "
                f"llm={branch_metrics['llm_feedback']['pairwise_accuracy']:.4f} "
                f"oracle={branch_metrics['oracle_feedback']['pairwise_accuracy']:.4f}",
                flush=True,
            )

    report["status"] = "completed"
    report["summary"] = _summary(report["results"], strategies)
    _save(output, report)
    return report


def _mark_failed(output: Path, error: BaseException) -> None:
    """Make an interrupted report explicit without hiding the original error."""
    if not output.exists():
        return
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("status") == "completed":
            return
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        _save(output, report)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        # Preserve the original exception if the report itself is unreadable.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=FORMAL_MODEL_REVISION)
    parser.add_argument(
        "--protocol-version",
        choices=(FORMAL_PROTOCOL_VERSION, "custom-diagnostic-v1"),
        default=FORMAL_PROTOCOL_VERSION,
        help="formal protocol validates every frozen parameter and model snapshot",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(17, 27)))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--pool-group-limit", type=int, default=256)
    parser.add_argument("--eval-group-limit", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--split-unit",
        choices=SPLIT_UNITS,
        default="prompt",
        help="hash pairs for legacy reports or prompts for leakage-safe formal runs",
    )
    parser.add_argument(
        "--fail-on-split-overlap",
        action="store_true",
        default=True,
        help="abort if one prompt occurs in more than one split",
    )
    parser.add_argument(
        "--vary-split-by-seed",
        action="store_true",
        help="legacy mode: use split_seed + seed index instead of one fixed split",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="default training steps when no strategy-specific override is set",
    )
    parser.add_argument(
        "--seed-only-max-steps",
        type=int,
        default=None,
        help="training steps for seed-only; defaults to --max-steps",
    )
    parser.add_argument(
        "--feedback-max-steps",
        type=int,
        default=None,
        help="training steps for random and aladdin-v2; defaults to --max-steps",
    )
    parser.add_argument(
        "--training-budget-mode",
        choices=("fixed-steps", "equal-epochs"),
        default="equal-epochs",
        help="fixed optimizer steps or equalize approximate record exposure across strategies",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=60.0)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        _mark_failed(Path(args.output), exc)
        raise


if __name__ == "__main__":
    main()
