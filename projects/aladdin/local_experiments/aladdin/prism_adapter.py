from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from core_system.mle_core.models import TaskRecord, TaskSpec
from core_system.mle_core.registry import TaskRegistry


PRISM_DATASET_ID = "HannahRoseKirk/prism-alignment"
PRISM_TASK_ID = "prism-response-preference"
PRISM_ADAPTER_VERSION = "prism-adapter-v1"
SPLITS = ("seed", "development", "held_out", "final_test")
SPLIT_BUCKETS = (("seed", 20), ("development", 30), ("held_out", 30), ("final_test", 20))
REQUIRED_UTTERANCE_FIELDS = {
    "utterance_id", "conversation_id", "user_id", "turn", "within_turn_id",
    "user_prompt", "model_response", "score", "if_chosen",
}
REQUIRED_CONVERSATION_FIELDS = {
    "conversation_id", "user_id", "conversation_type", "choice_attributes",
    "performance_attributes", "open_feedback",
}


class PrismAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class PrismRecord:
    record_id: str
    task_id: str
    split: str
    user_id: str
    conversation_id: str
    interaction_id: str
    turn: int
    candidate_id: int
    prompt: str
    response: str
    score: int
    if_chosen: bool
    source_group_id: str
    feedback: dict[str, Any]
    metadata: dict[str, Any]

    def to_task_record(self, *, release_feedback: bool = False) -> TaskRecord:
        metadata = {
            "dataset": PRISM_DATASET_ID,
            "adapter_version": PRISM_ADAPTER_VERSION,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "interaction_id": self.interaction_id,
            "turn": self.turn,
            "candidate_id": self.candidate_id,
            "feedback_type": "human_rating_and_choice",
            "feedback_artifact_version": self.feedback["feedback_artifact_version"],
            "feedback_released": release_feedback,
            **self.metadata,
        }
        if release_feedback:
            metadata["feedback"] = self.feedback
            metadata["conversation_feedback"] = self.metadata["conversation_feedback"]
            metadata["if_chosen"] = self.if_chosen
        else:
            metadata.pop("conversation_feedback", None)
        return TaskRecord(
            record_id=self.record_id,
            task_id=self.task_id,
            split=self.split,
            input={"prompt": self.prompt, "response": self.response},
            target=self.score if release_feedback else None,
            source_group_id=self.source_group_id,
            metadata=metadata,
        )

    def release_feedback(self) -> TaskRecord:
        """Materialize the human label only after the core releases feedback."""
        return self.to_task_record(release_feedback=True)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrismAdapterError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise PrismAdapterError(f"Expected an object in {path}:{line_number}")
            yield value


def assign_user_splits(user_ids: Iterable[str], *, seed: int = 17) -> dict[str, str]:
    """Assign each user to one split using a stable hash, without user leakage."""
    assignments: dict[str, str] = {}
    for user_id in sorted(set(user_ids)):
        digest = hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        lower = 0
        for split, width in SPLIT_BUCKETS:
            if lower <= bucket < lower + width:
                assignments[user_id] = split
                break
            lower += width
        else:
            raise AssertionError(f"unassigned split bucket: {bucket}")
    if len(assignments) >= 20 and set(assignments.values()) != set(SPLITS):
        raise PrismAdapterError("stable split assignment did not produce all four splits")
    return assignments


def _require_fields(row: dict[str, Any], required: set[str], *, source: str) -> None:
    missing = sorted(required - row.keys())
    if missing:
        raise PrismAdapterError(f"{source} is missing fields: {missing}")


def validate_conversations(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    conversations: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        _require_fields(row, REQUIRED_CONVERSATION_FIELDS, source=f"conversation {row_number}")
        conversation_id = str(row["conversation_id"])
        if not conversation_id or conversation_id in conversations:
            raise PrismAdapterError(f"duplicate or empty conversation_id: {conversation_id!r}")
        conversations[conversation_id] = row
    return conversations


def normalize_utterances(
    utterances: Iterable[dict[str, Any]],
    conversations: dict[str, dict[str, Any]],
    *,
    split_by_user: dict[str, str],
    data_version: str,
    source_revision: str,
) -> list[PrismRecord]:
    records: list[PrismRecord] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(utterances, start=1):
        _require_fields(row, REQUIRED_UTTERANCE_FIELDS, source=f"utterance {row_number}")
        utterance_id = str(row["utterance_id"])
        conversation_id = str(row["conversation_id"])
        user_id = str(row["user_id"])
        if not utterance_id or utterance_id in seen_ids:
            raise PrismAdapterError(f"duplicate or empty utterance_id: {utterance_id!r}")
        seen_ids.add(utterance_id)
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise PrismAdapterError(f"utterance {utterance_id} references unknown conversation {conversation_id}")
        if str(conversation["user_id"]) != user_id:
            raise PrismAdapterError(f"user mismatch for conversation {conversation_id}")
        split = split_by_user.get(user_id)
        if split not in SPLITS:
            raise PrismAdapterError(f"user {user_id} has no valid split assignment")
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or int(score) != score or not 1 <= int(score) <= 100:
            raise PrismAdapterError(f"score must be an integer in [1, 100]: {utterance_id}")
        if not isinstance(row["if_chosen"], bool):
            raise PrismAdapterError(f"if_chosen must be boolean: {utterance_id}")
        conversation_feedback = {
            "choice_attributes": conversation.get("choice_attributes"),
            "performance_attributes": conversation.get("performance_attributes"),
            "open_feedback": conversation.get("open_feedback"),
            "feedback_scope": "conversation",
        }
        records.append(
            PrismRecord(
                record_id=f"prism-{utterance_id}",
                task_id=PRISM_TASK_ID,
                split=split,
                user_id=user_id,
                conversation_id=conversation_id,
                interaction_id=str(row.get("interaction_id", "")),
                turn=int(row["turn"]),
                candidate_id=int(row["within_turn_id"]),
                prompt=str(row["user_prompt"]),
                response=str(row["model_response"]),
                score=int(score),
                if_chosen=row["if_chosen"],
                source_group_id=f"{user_id}:{conversation_id}",
                feedback={
                    "score": int(score),
                    "if_chosen": row["if_chosen"],
                    "feedback_source": "human_participant",
                    "feedback_scope": "candidate",
                    "feedback_artifact_version": data_version,
                },
                metadata={
                    "data_version": data_version,
                    "source_revision": source_revision,
                    "conversation_type": conversation.get("conversation_type"),
                    "model_name": row.get("model_name"),
                    "model_provider": row.get("model_provider"),
                    "included_in_balanced_subset": row.get("included_in_balanced_subset"),
                    "conversation_feedback": conversation_feedback,
                },
            )
        )
    return records


def load_prism_records(
    raw_dir: Path,
    *,
    seed: int = 17,
    data_version: str = "prism-local-v1",
    source_revision: str = "unknown",
    include_final_test: bool = False,
) -> tuple[list[PrismRecord], dict[str, Any]]:
    conversation_path = raw_dir / "conversations.jsonl"
    utterance_path = raw_dir / "utterances.jsonl"
    if not conversation_path.exists() or not utterance_path.exists():
        raise FileNotFoundError("PRISM raw directory must contain conversations.jsonl and utterances.jsonl")
    conversations = validate_conversations(iter_jsonl(conversation_path))
    user_ids = {str(row["user_id"]) for row in conversations.values()}
    split_by_user = assign_user_splits(user_ids, seed=seed)
    records = normalize_utterances(
        iter_jsonl(utterance_path),
        conversations,
        split_by_user=split_by_user,
        data_version=data_version,
        source_revision=source_revision,
    )
    if not include_final_test:
        records = [record for record in records if record.split != "final_test"]
    report = {
        "dataset_id": PRISM_DATASET_ID,
        "task_id": PRISM_TASK_ID,
        "adapter_version": PRISM_ADAPTER_VERSION,
        "data_version": data_version,
        "source_revision": source_revision,
        "split_seed": seed,
        "split_rule": "stable sha256 user assignment; all conversations from a user stay together",
        "sealed_split": "final_test",
        "include_final_test": include_final_test,
        "feedback_release_policy": "scores and open feedback are hidden from TaskRecord until PrismRecord.release_feedback()",
        "survey_sensitive_fields_loaded": False,
        "conversation_count": len(conversations),
        "user_count": len(user_ids),
        "utterance_count": len(records),
        "user_counts_by_split": dict(Counter(split_by_user.values())),
        "record_counts_by_split": dict(Counter(record.split for record in records)),
        "source_group_count": len({record.source_group_id for record in records}),
    }
    return records, report


def build_task_registry(records: Iterable[PrismRecord]) -> TaskRegistry:
    registry = TaskRegistry()
    registry.register_task(TaskSpec(
        task_id=PRISM_TASK_ID,
        name="PRISM candidate response preference",
        input_schema="prompt and candidate response",
        target_schema="human score 1-100",
    ))
    registry.add_records(record.to_task_record() for record in records)
    return registry


def validate_raw_directory(raw_dir: Path, output_path: Path, *, seed: int = 17) -> dict[str, Any]:
    records, report = load_prism_records(
        raw_dir,
        seed=seed,
        data_version="prism-local-v1",
        source_revision="a9ec82149036374768f70e132a2369a750e06bbe",
        include_final_test=True,
    )
    report["records_available_to_tuning"] = sum(record.split != "final_test" for record in records)
    report["final_test_records_sealed"] = sum(record.split == "final_test" for record in records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local PRISM raw files without downloading data.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(validate_raw_directory(args.raw_dir, args.report, seed=args.seed), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
