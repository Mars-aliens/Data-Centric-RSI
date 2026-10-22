from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from core_system.mle_core.models import TaskRecord


DATASET_REVISION = "a9ec82149036374768f70e132a2369a750e06bbe"
DEFAULT_TASK_ID = "prism-alignment-conversation"
ALLOWED_DELAYS = (0, 2, 5)


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def _user_split(user_id: str) -> str:
    """Assign every conversation from a participant to one split."""
    bucket = _stable_int(user_id) % 100
    if bucket < 50:
        return "seed"
    if bucket < 70:
        return "held_in"
    if bucket < 90:
        return "held_out"
    return "final_test"


def _conversation_order(row: dict[str, Any]) -> tuple[str, str]:
    timestamp = str(row.get("generated_datetime") or "")
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        normalized = parsed.isoformat()
    except ValueError:
        normalized = timestamp
    return normalized, str(row["conversation_id"])


def normalize_prism_rows(
    conversations: Iterable[dict[str, Any]],
    utterances: Iterable[dict[str, Any]],
    *,
    task_id: str = DEFAULT_TASK_ID,
    feedback_delay: int = 2,
    batch_size: int = 50,
) -> list[TaskRecord]:
    """Create provenance-preserving records from the official PRISM release.

    PRISM scores are retained as feedback metadata rather than converted into a
    hard target. The source has human ratings, but it does not define a target
    for a LEGO data-construction task.
    """
    if feedback_delay not in ALLOWED_DELAYS:
        raise ValueError(f"feedback_delay must be one of {ALLOWED_DELAYS}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    conversation_rows = list(conversations)
    conversation_by_id = {str(row["conversation_id"]): row for row in conversation_rows}
    if len(conversation_by_id) != len(conversation_rows):
        raise ValueError("conversation_id values must be unique")
    ordered_ids = sorted(conversation_by_id, key=lambda key: _conversation_order(conversation_by_id[key]))
    batch_by_id = {conversation_id: index // batch_size for index, conversation_id in enumerate(ordered_ids)}

    records: list[TaskRecord] = []
    for row in utterances:
        conversation_id = str(row["conversation_id"])
        conversation = conversation_by_id.get(conversation_id)
        if conversation is None:
            raise ValueError(f"Utterance references unknown conversation {conversation_id}")
        user_id = str(row["user_id"])
        if user_id != str(conversation["user_id"]):
            raise ValueError(f"User mismatch for conversation {conversation_id}")
        source_batch = batch_by_id[conversation_id]
        split = _user_split(user_id)
        release_batch = source_batch + feedback_delay
        score = row.get("score")
        record_id = f"prism-{row['utterance_id']}"
        records.append(
            TaskRecord(
                record_id=record_id,
                task_id=task_id,
                split=split,
                input={
                    "user_prompt": row.get("user_prompt"),
                    "model_response": row.get("model_response"),
                },
                target=None,
                source_group_id=f"conversation-{conversation_id}",
                metadata={
                    "dataset": "HannahRoseKirk/prism-alignment",
                    "source_revision": DATASET_REVISION,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "interaction_id": str(row["interaction_id"]),
                    "turn": int(row["turn"]),
                    "within_turn_id": int(row["within_turn_id"]),
                    "conversation_type": row.get("conversation_type"),
                    "model_name": row.get("model_name"),
                    "model_provider": row.get("model_provider"),
                    "batch_id": f"stream-{source_batch:04d}",
                    "feedback": {
                        "kind": "native_human_rating",
                        "score": score,
                        "if_chosen": row.get("if_chosen"),
                        "source_batch": source_batch,
                        "release_batch": release_batch,
                        "delay_is_imposed": True,
                    },
                },
            )
        )
    return records


def load_prism_records(
    raw_dir: str | Path,
    *,
    task_id: str = DEFAULT_TASK_ID,
    feedback_delay: int = 2,
    batch_size: int = 50,
) -> list[TaskRecord]:
    root = Path(raw_dir)
    return normalize_prism_rows(
        _read_jsonl(root / "conversations.jsonl"),
        _read_jsonl(root / "utterances.jsonl"),
        task_id=task_id,
        feedback_delay=feedback_delay,
        batch_size=batch_size,
    )


def write_prism_records(records: Iterable[TaskRecord], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
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
