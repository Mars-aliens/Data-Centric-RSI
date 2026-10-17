from __future__ import annotations

"""HH-RLHF downstream Qwen LoRA pilot for Aladdin feedback selection."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _preload_legacy_sklearn() -> None:
    """Load sklearn from the legacy environment without shadowing HF packages."""
    site_path = os.environ.get("ALADDIN_SKLEARN_SITE")
    if not site_path or "sklearn" in sys.modules:
        return
    sys.path.insert(0, site_path)
    try:
        import sklearn  # noqa: F401
    finally:
        sys.path.remove(site_path)


_preload_legacy_sklearn()

from local_experiments.aladdin.hh_rlhf_feedback import (
    HHRLHFGroup,
    _groups_for_split_seed,
    load_hh_rlhf_groups,
    select_feedback_groups,
)
from local_experiments.aladdin.hh_rlhf_llm_training_pilot import (
    _judgment,
    _llm_records,
)
from local_experiments.aladdin.qwen_preference import (
    load_adapter_checkpoint,
    prompt_for,
    train_checkpoint,
)


SEED_ONLY = "seed-only"
DEFAULT_STRATEGIES = (SEED_ONLY, "random", "aladdin-v2")


def _payload(record: Any) -> dict[str, Any]:
    return {
        "input": record.input,
        "target": record.target,
        "source_group_id": record.source_group_id,
        "record_id": record.record_id,
    }


def _candidate(record: Any) -> str:
    return str(record.record_id).rsplit("-", 1)[-1]


def _pairwise_accuracy(
    checkpoint: dict[str, Any],
    groups: list[HHRLHFGroup],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Compare Qwen's preferred-vs-rejected margin within each held-out pair."""
    margins = _record_preference_margins(
        checkpoint,
        [record for group in groups for record in group.release_feedback()],
        batch_size=batch_size,
    )
    correct = 0
    total = 0
    for group in groups:
        chosen = next(record for record in group.release_feedback() if _candidate(record) == "chosen")
        rejected = next(record for record in group.release_feedback() if _candidate(record) == "rejected")
        correct += margins[chosen.record_id] > margins[rejected.record_id]
        total += 1
    return {"pairwise_accuracy": correct / total if total else 0.0, "pairwise_groups": float(total)}


def _record_preference_margins(
    checkpoint: dict[str, Any],
    records: list[Any],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Return Qwen's preferred-minus-rejected logit for each response record."""
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, label_ids = load_adapter_checkpoint(checkpoint, target)
    rows = list(records)
    margins: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            encoded = tokenizer(
                [prompt_for(_payload(record)) for record in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(checkpoint.get("max_length", 1024)),
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            logits = model(**encoded).logits[:, -1, :]
            margin = logits[:, int(label_ids["A"])] - logits[:, int(label_ids["B"])]
            for record, value in zip(batch, margin.detach().float().cpu().tolist()):
                margins[record.record_id] = float(value)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return margins


def _pairwise_uncertainty(
    checkpoint: dict[str, Any],
    groups: list[HHRLHFGroup],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Score groups by Qwen's uncertainty about the preferred response."""
    margins = _record_preference_margins(
        checkpoint,
        [record for group in groups for record in group.hidden_records],
        batch_size=batch_size,
    )
    return {
        group.group_id: 1.0 / (
            1.0
            + abs(
                margins[next(record for record in group.hidden_records if _candidate(record) == "chosen").record_id]
                - margins[next(record for record in group.hidden_records if _candidate(record) == "rejected").record_id]
            )
        )
        for group in groups
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
            "mean_feedback_accuracy": (
                sum(row["feedback_accuracy"] for row in rows if row["feedback_accuracy"] is not None)
                / sum(row["feedback_accuracy"] is not None for row in rows)
                if any(row["feedback_accuracy"] is not None for row in rows)
                else None
            ),
            "mean_pairwise_accuracy": sum(row["metrics"]["pairwise_accuracy"] for row in rows) / len(rows),
            "mean_training_seconds": sum(row["cost"]["training_seconds"] for row in rows) / len(rows),
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    strategies = tuple(args.strategies)
    seeds = tuple(args.seeds)
    output = Path(args.output)
    existing: dict[str, Any] = {}
    if args.resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
    results = list(existing.get("results", []))
    completed = {(row["strategy"], int(row["seed"])) for row in results}
    groups = load_hh_rlhf_groups(Path(args.data), split_seed=args.split_seed)
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
    }
    report: dict[str, Any] = {
        "experiment": "aladdin-hh-rlhf-qwen-lora-pilot-v1",
        "status": "running",
        "dataset": "Anthropic/hh-rlhf",
        "data_path": str(Path(args.data).resolve()),
        "base_model_path": str(Path(args.base_model).resolve()),
        "configuration": {
            "strategies": list(strategies),
            "seeds": list(seeds),
            "split_seed": args.split_seed,
            "candidate_budget": args.candidate_budget,
            "seed_group_limit": args.seed_group_limit,
            "pool_group_limit": args.pool_group_limit,
            "eval_group_limit": args.eval_group_limit,
            "eval_batch_size": args.eval_batch_size,
            "llm_endpoint": args.endpoint,
            "llm_model": args.model,
            "training": config,
        },
        "results": results,
        "summary": _summary(results, strategies),
        "limitations": [
            "This is a downstream Qwen LoRA pilot, not a final-test result.",
            "The held-out pairwise metric compares preferred-vs-rejected logits.",
            "Use the same base model, split seeds, training steps, and optimizer settings for all strategies.",
        ],
    }
    _save(output, report)

    total = len(strategies) * len(seeds)
    for seed in seeds:
        run_split_seed = args.split_seed + (seed - seeds[0])
        run_groups = _groups_for_split_seed(groups, run_split_seed)
        seed_groups = [group for group in run_groups if group.split == "seed"][: args.seed_group_limit]
        pool_groups = [group for group in run_groups if group.split == "held_in"][: args.pool_group_limit]
        held_out_groups = [group for group in run_groups if group.split == "held_out"]
        if args.eval_group_limit > 0:
            held_out_groups = held_out_groups[: args.eval_group_limit]
        seed_records = [record for group in seed_groups for record in group.release_feedback()]
        if len(seed_groups) < 2 or len(pool_groups) < args.candidate_budget or not held_out_groups:
            raise ValueError(f"insufficient groups for seed={seed} split_seed={run_split_seed}")

        for strategy in strategies:
            if (strategy, seed) in completed:
                print(f"skip completed strategy={strategy} seed={seed}", flush=True)
                continue
            started = time.perf_counter()
            selected: list[HHRLHFGroup] = []
            judgments: list[dict[str, Any]] = []
            train_records = list(seed_records)
            feedback_accuracy = None
            if strategy != SEED_ONLY:
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
                    train_records.extend(_llm_records(group, judgment))
                feedback_accuracy = sum(bool(item["correct"]) for item in judgments) / len(judgments)
            else:
                diagnostics = {
                    "strategy": SEED_ONLY,
                    "selection_definition": "fixed seed records without additional feedback",
                    "selected_group_ids": [],
                }

            adapter_dir = adapters_root / f"{strategy}-seed{seed}"
            checkpoint, cost = train_checkpoint(
                [_payload(record) for record in train_records],
                config=config,
                seed=seed,
                output_dir=adapter_dir,
            )
            metrics = _pairwise_accuracy(
                checkpoint,
                held_out_groups,
                batch_size=args.eval_batch_size,
            )
            row = {
                "strategy": strategy,
                "seed": seed,
                "split_seed": run_split_seed,
                "feedback_accuracy": feedback_accuracy,
                "selected_groups": len(selected),
                "diagnostics": diagnostics,
                "judgments": judgments,
                "metrics": metrics,
                "cost": {
                    **cost,
                    "wall_seconds": time.perf_counter() - started,
                    "llm_calls": len(judgments),
                    "llm_tokens": sum(item["usage"]["total_tokens"] for item in judgments),
                },
                "checkpoint": checkpoint,
            }
            results.append(row)
            completed.add((strategy, seed))
            report["results"] = results
            report["summary"] = _summary(results, strategies)
            _save(output, report)
            print(
                f"completed {len(results)}/{total} strategy={strategy} seed={seed} "
                f"pairwise={metrics['pairwise_accuracy']:.4f}",
                flush=True,
            )
    report["status"] = "completed"
    report["summary"] = _summary(results, strategies)
    _save(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--candidate-budget", type=int, default=32)
    parser.add_argument("--seed-group-limit", type=int, default=64)
    parser.add_argument("--pool-group-limit", type=int, default=256)
    parser.add_argument(
        "--eval-group-limit",
        type=int,
        default=512,
        help="Held-out groups per seed; use 0 for all groups.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=86.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
