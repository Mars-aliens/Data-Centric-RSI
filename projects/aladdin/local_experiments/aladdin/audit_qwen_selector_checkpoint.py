from __future__ import annotations

"""Audit train/held-out preference signals for saved HH-RLHF Qwen adapters."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# Import the diagnostic module first; it preloads sklearn from the legacy site path.
from local_experiments.aladdin.hh_rlhf_qwen_lora_label_diagnostic import (
    _llm_records,
    _sample_groups,
)
from local_experiments.aladdin.hh_rlhf_feedback import (
    _groups_for_split_seed,
    load_hh_rlhf_groups,
)
from local_experiments.aladdin.hh_rlhf_qwen_lora_pilot import _payload
from local_experiments.aladdin.qwen_preference import (
    _configure_memory_limit,
    device,
    label_ids,
    load_adapter_checkpoint,
    load_tokenizer,
    prompt_for,
)


def _score(
    checkpoint: dict[str, Any], records: list[Any], *, batch_size: int = 8
) -> tuple[list[str], dict[str, float]]:
    target = device()
    _configure_memory_limit(target)
    tokenizer, model, ids = load_adapter_checkpoint(checkpoint, target)
    labels = ("A", "B")
    token_ids = torch.tensor([ids[label] for label in labels], device=target)
    predictions: list[str] = []
    margins: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            encoded = tokenizer(
                [prompt_for(_payload(record)) for record in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(checkpoint.get("max_length", 1024)),
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            logits = model(**encoded).logits[:, -1, :]
            scores = logits.index_select(1, token_ids)
            predictions.extend(labels[index] for index in scores.argmax(dim=1).tolist())
            ab_margin = scores[:, 0] - scores[:, 1]
            margins.update(
                (record.record_id, float(value))
                for record, value in zip(batch, ab_margin.float().cpu().tolist())
            )
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, margins


def _metrics(
    records: list[Any], predictions: list[str], margins: dict[str, float]
) -> dict[str, Any]:
    target_to_label = {"preferred": "A", "rejected": "B"}
    pointwise = sum(
        prediction == target_to_label[record.target]
        for record, prediction in zip(records, predictions)
    ) / len(records)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        if record.source_group_id:
            grouped[record.source_group_id].append(record)
    correct = 0
    total = 0
    for pair in grouped.values():
        preferred = next((record for record in pair if record.target == "preferred"), None)
        rejected = next((record for record in pair if record.target == "rejected"), None)
        if preferred is None or rejected is None:
            continue
        correct += margins[preferred.record_id] > margins[rejected.record_id]
        total += 1
    margin_means: dict[str, float | None] = {}
    for label in ("preferred", "rejected"):
        values = [margins[r.record_id] for r in records if r.target == label]
        margin_means[label] = sum(values) / len(values) if values else None
    return {
        "record_accuracy": pointwise,
        "pairwise_accuracy": correct / total if total else None,
        "pair_count": total,
        "margin_means": margin_means,
    }


def audit(report_path: Path, data_path: Path, *, seed: int) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split_seed = int(report["configuration"]["split_seed"])
    groups = load_hh_rlhf_groups(
        data_path, split_seed=split_seed, split_unit="prompt"
    )
    groups = _groups_for_split_seed(groups, split_seed, "prompt")
    seed_groups = _sample_groups(
        [group for group in groups if group.split == "seed"],
        int(report["configuration"]["seed_group_limit"]),
        split_seed + 1103,
    )
    pool_groups = _sample_groups(
        [group for group in groups if group.split == "held_in"],
        int(report["configuration"]["pool_group_limit"]),
        split_seed + 2207,
    )
    heldout_groups = _sample_groups(
        [group for group in groups if group.split == "held_out"],
        int(report["configuration"]["eval_group_limit"]),
        split_seed + 3301,
    )
    seed_records = [record for group in seed_groups for record in group.release_feedback()]
    pool_by_id = {group.group_id: group for group in pool_groups}
    adapter_root = report_path.parent / f"{report_path.stem}-adapters"
    base_model_path = report["base_model_path"]
    seed_rows = [
        row for row in report["results"] if int(row["seed"]) == seed
    ]
    if not seed_rows:
        raise ValueError(f"report has no results for seed={seed}")

    for strategy in ("seed-only", "random", "aladdin-qwen"):
        row = next((item for item in seed_rows if item["strategy"] == strategy), None)
        if row is None:
            raise ValueError(f"report has no {strategy} result for seed={seed}")
        judgments = {item["group_id"]: item for item in row["judgments"]}
        selected = [pool_by_id[group_id] for group_id in row["selected_group_ids"]]
        train_records = list(seed_records)
        for group in selected:
            train_records.extend(_llm_records(group, judgments[group.group_id]))
        heldout_records = [
            record for group in heldout_groups for record in group.release_feedback()
        ]
        adapter_path = adapter_root / f"{strategy}-llm_feedback-seed{seed}"
        tokenizer = load_tokenizer(str(adapter_path))
        checkpoint = {
            "base_model_path": base_model_path,
            "adapter_path": str(adapter_path),
            "label_ids": label_ids(tokenizer),
            "max_length": int(report["configuration"]["training"]["max_length"]),
        }
        del tokenizer
        for split_name, records in (
            ("train", train_records),
            ("heldout", heldout_records),
        ):
            predictions, margins = _score(checkpoint, records)
            result = _metrics(records, predictions, margins)
            print(
                f"{strategy:12s} {split_name:7s} "
                f"record_acc={result['record_accuracy']:.4f} "
                f"pairwise={result['pairwise_accuracy']:.4f} "
                f"pairs={result['pair_count']} "
                f"margin_means={result['margin_means']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    audit(args.report, args.data, seed=args.seed)


if __name__ == "__main__":
    main()
