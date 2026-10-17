from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

from core_system.mle_core.models import TaskRecord

from .protocol import DataProtocol


SCORE_FIELDS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
DATASET_REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"


def _stable_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_for_group(group_key: str, source_split: str) -> str:
    if source_split == "validation":
        return "final_test"
    if source_split != "train":
        raise ValueError(f"Unsupported HelpSteer2 source split: {source_split}")
    bucket = int(_stable_hex(group_key)[:8], 16) % 100
    if bucket < 70:
        return "seed"
    if bucket < 90:
        return "held_in"
    return "held_out"


def _batch_for_group(group_key: str) -> int:
    return int(_stable_hex(group_key)[8:16], 16) % 1000


def normalize_helpsteer2_rows(
    rows: Iterable[dict[str, Any]],
    *,
    task_id: str = "helpsteer2-preference",
    feedback_delay: int = 2,
    source_split: str = "train",
) -> list[TaskRecord]:
    """Convert HelpSteer2 rows into pair-grouped, provenance-preserving records.

    The target is derived only within a prompt group: responses with the
    highest mean score are `preferred`, the lowest are `rejected`, and any
    strict middle scores are `tie`. Raw rubric scores stay in metadata so
    downstream experiments can choose a stricter evaluator without losing the
    original supervision.
    """
    if feedback_delay not in {0, 2, 5}:
        raise ValueError("HelpSteer2 feedback_delay must be one of 0, 2, 5")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt = str(row["prompt"])
        grouped[prompt].append(row)

    records: list[TaskRecord] = []
    for prompt in sorted(grouped, key=_stable_hex):
        pair = grouped[prompt]
        if len(pair) < 2:
            continue
        split = _split_for_group(prompt, source_split)
        batch = _batch_for_group(prompt)
        means = [
            sum(float(row[field]) for field in SCORE_FIELDS) / len(SCORE_FIELDS)
            for row in pair
        ]
        highest = max(means)
        lowest = min(means)
        if highest == lowest:
            labels = tuple("tie" for _ in means)
        else:
            labels = tuple(
                "preferred" if score == highest else "rejected" if score == lowest else "tie"
                for score in means
            )
        group_id = f"prompt-{_stable_hex(prompt)[:16]}"
        for index, (row, label, mean_score) in enumerate(zip(pair, labels, means, strict=True)):
            response = str(row["response"])
            record_key = f"{prompt}\n{response}\n{index}"
            record_id = f"helpsteer2-{_stable_hex(record_key)[:20]}"
            scores = {field: int(row[field]) for field in SCORE_FIELDS}
            records.append(
                TaskRecord(
                    record_id=record_id,
                    task_id=task_id,
                    split=split,
                    input=f"Prompt: {prompt}\nResponse: {response}",
                    target=label,
                    source_group_id=group_id,
                    metadata={
                        "dataset": "nvidia/HelpSteer2",
                        "pair_index": index,
                        "pair_size": len(pair),
                        "group_size": len(pair),
                        "batch_id": f"stream-{batch:04d}",
                        "score_mean": mean_score,
                        "scores": scores,
                        "feedback": {
                            "decision": label,
                            "scores": scores,
                            "source_batch": batch,
                            "release_batch": batch + feedback_delay,
                        },
                    },
                )
            )
    return records


def load_helpsteer2_records(
    *,
    split: str = "train",
    task_id: str = "helpsteer2-preference",
    feedback_delay: int = 2,
) -> list[TaskRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HelpSteer2 loading requires the optional `datasets` package"
        ) from exc
    dataset = load_dataset(
        "nvidia/HelpSteer2",
        split=split,
        revision=DATASET_REVISION,
        streaming=True,
    )
    return normalize_helpsteer2_rows(
        dataset,
        task_id=task_id,
        feedback_delay=feedback_delay,
        source_split=split,
    )


def write_jsonl(records: Iterable[TaskRecord], path: str) -> int:
    import json
    from pathlib import Path

    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "task_id": record.task_id,
                        "split": record.split,
                        "input": record.input,
                        "target": record.target,
                        "source_group_id": record.source_group_id,
                        "metadata": record.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count

