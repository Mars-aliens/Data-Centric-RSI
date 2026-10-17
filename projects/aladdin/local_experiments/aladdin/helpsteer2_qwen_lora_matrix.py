"""Qwen LoRA downstream validation for a frozen HelpSteer2 feedback matrix.

The matrix fixes feedback allocation and selected groups. This runner only
replaces the TF-IDF downstream proxy with the shared Qwen LoRA trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _preload_legacy_sklearn() -> None:
    """Make the CPU sklearn components available in the Qwen environment."""
    site_path = os.environ.get("ALADDIN_SKLEARN_SITE")
    if not site_path or "sklearn" in sys.modules:
        return
    sys.path.insert(0, site_path)
    try:
        import sklearn  # noqa: F401
    finally:
        sys.path.remove(site_path)


_preload_legacy_sklearn()

from local_experiments.aladdin.active_feedback_helpsteer2 import (
    PROFILES,
    _labeled_records,
    _oracle_preference,
    _posterior_labels,
    _predict_groups,
    _file_sha256,
    _seed_attribute_models,
)
from local_experiments.aladdin.helpsteer2_feedback import (
    DATASET_ID,
    DATASET_REVISION,
    load_helpsteer2_groups,
)


PROTOCOL_VERSION = "helpsteer2-qwen-lora-frozen-matrix-v1"
STRATEGIES = ("seed-only", "random", "disagreement-only", "impact-aware-random")
QWEN_LABEL_TO_TEXT = {"A": "preferred", "B": "rejected", "C": "tie"}


def train_checkpoint(*args: Any, **kwargs: Any):
    # Keep protocol and mock tests usable in CPU-only environments.
    from local_experiments.aladdin.qwen_preference import train_checkpoint as impl

    return impl(*args, **kwargs)


def predict_scores(*args: Any, **kwargs: Any):
    from local_experiments.aladdin.qwen_preference import predict_scores as impl

    return impl(*args, **kwargs)


def _payload(record: Any) -> dict[str, Any]:
    return {
        "input": record.input,
        "target": record.target,
        "source_group_id": record.source_group_id,
        "record_id": record.record_id,
    }


def _pairwise_accuracy(scores: list[dict[str, float]], groups: list[Any],
                       labels: dict[str, str]) -> dict[str, float]:
    correct = 0
    total = 0
    cursor = 0
    for group in groups:
        relation = labels[group.group_id]
        if relation == "tie":
            cursor += len(group.hidden_records)
            continue
        if len(group.hidden_records) != 2:
            raise ValueError(f"expected pairwise group: {group.group_id}")
        first, second = scores[cursor : cursor + 2]
        cursor += 2
        predicted_index = max(
            range(2),
            key=lambda index: (first if index == 0 else second).get("A", float("-inf")),
        )
        actual_index = 0 if relation == "preferred" else 1
        correct += int(predicted_index == actual_index)
        total += 1
    return {
        "pairwise_accuracy": correct / total if total else 0.0,
        "pairwise_groups": float(total),
    }


def _normalize_prediction(label: str) -> str:
    """Accept the Qwen evaluator's A/B/C keys and text labels in tests."""
    return QWEN_LABEL_TO_TEXT.get(str(label), str(label))


def _matrix_rows(matrix: dict[str, Any], *, split_seeds: tuple[int, ...],
                 policy_seeds: tuple[int, ...], profile: str) -> list[dict[str, Any]]:
    rows = [
        row for row in matrix.get("results", [])
        if row.get("strategy") in STRATEGIES
        and int(row["split_seed"]) in split_seeds
        and int(row["policy_seed"]) in policy_seeds
        and row.get("target_profile_simulator") == profile
    ]
    expected = len(STRATEGIES) * len(split_seeds) * len(policy_seeds)
    if len(rows) != expected:
        raise ValueError(f"frozen matrix has {len(rows)} rows; expected {expected}")
    return rows


def run_matrix(
    data_path: Path,
    matrix_path: Path,
    output_path: Path,
    *,
    base_model: str,
    split_seeds: tuple[int, ...] = (17, 23, 29, 37),
    policy_seeds: tuple[int, ...] = (17, 18, 19),
    target_profile: str = "helpfulness-only",
    max_steps: int = 150,
    batch_size: int = 2,
    eval_batch_size: int = 16,
    max_length: int = 1024,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_gpu_memory_gib: float = 86.0,
    resume: bool = True,
) -> dict[str, Any]:
    if target_profile not in PROFILES:
        raise ValueError(f"unknown target profile: {target_profile}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("final_test_accessed") or matrix.get("configuration", {}).get("final_test_loaded"):
        raise ValueError("frozen matrix accessed final_test")
    rows = _matrix_rows(matrix, split_seeds=split_seeds, policy_seeds=policy_seeds,
                        profile=target_profile)
    groups = load_helpsteer2_groups(data_path)
    by_id = {group.group_id: group for group in groups}
    selected_keys = {(row["split_seed"], row["policy_seed"], row["strategy"]) for row in rows}
    if len(selected_keys) != len(rows):
        raise ValueError("duplicate frozen matrix condition")

    existing = json.loads(output_path.read_text(encoding="utf-8")) if resume and output_path.exists() else {}
    results = list(existing.get("results", []))
    completed = {(int(row["split_seed"]), int(row["policy_seed"]), row["strategy"]) for row in results}
    adapter_root = output_path.parent / (output_path.stem + "-adapters")
    config = {
        "base_model_path": base_model,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "per_device_batch_size": batch_size,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_gpu_memory_gib": max_gpu_memory_gib,
        "eval_batch_size": eval_batch_size,
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "data_sha256": _file_sha256(data_path),
        "frozen_matrix_path": str(matrix_path.resolve()),
        "frozen_matrix_sha256": _file_sha256(matrix_path),
        "target_profile": target_profile,
        "configuration": {
            "split_seeds": list(split_seeds),
            "policy_seeds": list(policy_seeds),
            "strategies": list(STRATEGIES),
            "training": config,
            "selection_source": "frozen_multiround_matrix",
            "final_test_accessed": False,
        },
        "results": results,
        "limitations": [
            "This validates Qwen LoRA downstream training on a frozen simulated-feedback matrix.",
            "It is not new human feedback and does not validate an independent dataset.",
        ],
    }

    def save() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = output_path.with_suffix(output_path.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(output_path)

    save()
    total = len(rows)
    for row in rows:
        key = (int(row["split_seed"]), int(row["policy_seed"]), row["strategy"])
        if key in completed:
            print(f"skip completed split={key[0]} policy={key[1]} strategy={key[2]}", flush=True)
            continue
        split_seed, policy_seed, strategy = key
        seed_groups = [by_id[item] for item in row["pool_ids"]["seed"]]
        heldout_groups = [by_id[item] for item in row["pool_ids"]["heldout"]]
        selected_groups = [by_id[item] for item in row["selected_data_ids"]]
        models = _seed_attribute_models(seed_groups, split_seed)
        prediction_groups = [*seed_groups, *selected_groups, *heldout_groups]
        query_groups = [
            type("Query", (), {"group_id": group.group_id, "records": group.hidden_records})()
            for group in prediction_groups
        ]
        _, predicted_values = _predict_groups(query_groups, models, tie_margin=0.0)
        posterior = row["rounds"][-1]["posterior"]
        selected_labels = _posterior_labels(
            selected_groups, predicted_values, posterior, tie_margin=0.0
        ) if selected_groups else {}
        seed_labels = {
            group.group_id: _oracle_preference(group, PROFILES[target_profile], tie_margin=0.0)
            for group in seed_groups
        }
        heldout_labels = {
            group.group_id: _oracle_preference(group, PROFILES[target_profile], tie_margin=0.0)
            for group in heldout_groups
        }
        train_records = _labeled_records(seed_groups, seed_labels, "seed")
        train_records.extend(_labeled_records(selected_groups, selected_labels, "held_in"))
        adapter_dir = adapter_root / f"split{split_seed}-policy{policy_seed}-{strategy}"
        started = time.perf_counter()
        checkpoint, cost = train_checkpoint(
            [_payload(record) for record in train_records],
            config=config,
            seed=policy_seed,
            output_dir=adapter_dir,
        )
        predicted = predict_scores(
            checkpoint,
            [_payload(record) for group in heldout_groups for record in group.hidden_records],
            batch_size=eval_batch_size,
        )
        heldout_records = [
            record for group in heldout_groups for record in group.hidden_records
        ]
        if len(predicted) != len(heldout_records):
            raise ValueError(
                f"Qwen evaluator returned {len(predicted)} scores for "
                f"{len(heldout_records)} records"
            )
        metrics = _pairwise_accuracy(predicted, heldout_groups, heldout_labels)
        results.append({
            "split_seed": split_seed,
            "policy_seed": policy_seed,
            "strategy": strategy,
            "target_profile": target_profile,
            "selected_group_count": len(selected_groups),
            "train_group_count": len(seed_groups) + len(selected_groups),
            "metrics": metrics,
            "cost": {**cost, "wall_seconds": time.perf_counter() - started},
            "checkpoint": checkpoint,
            "frozen_selected_data_ids": row["selected_data_ids"],
        })
        completed.add(key)
        report["results"] = results
        save()
        print(f"[{len(results)}/{total}] split={split_seed} policy={policy_seed} "
              f"strategy={strategy} pairwise={metrics['pairwise_accuracy']:.4f}", flush=True)

    report["status"] = "completed"
    report["results"] = results
    report["summary"] = {
        strategy: {
            "runs": len(values),
            "mean_pairwise_accuracy": sum(item["metrics"]["pairwise_accuracy"] for item in values) / len(values),
        }
        for strategy in STRATEGIES
        for values in [[item for item in results if item["strategy"] == strategy]]
        if values
    }
    save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[17, 23, 29, 37])
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--target-profile", choices=tuple(PROFILES), default="helpfulness-only")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=86.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_matrix(
        args.data, args.matrix, args.output, base_model=args.base_model,
        split_seeds=tuple(args.split_seeds), policy_seeds=tuple(args.policy_seeds),
        target_profile=args.target_profile, max_steps=args.max_steps,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        max_length=args.max_length, learning_rate=args.learning_rate,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, max_gpu_memory_gib=args.max_gpu_memory_gib,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
