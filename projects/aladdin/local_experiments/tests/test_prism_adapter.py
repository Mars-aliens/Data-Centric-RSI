from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from local_experiments.aladdin.prism_adapter import (
    PRISM_TASK_ID,
    PrismAdapterError,
    assign_user_splits,
    build_task_registry,
    normalize_utterances,
    load_prism_records,
)


def conversation(conversation_id: str, user_id: str) -> dict[str, object]:
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "conversation_type": "unguided",
        "choice_attributes": {"helpfulness": 80},
        "performance_attributes": {"fluency": 75},
        "open_feedback": "Useful answer.",
    }


def utterance(utterance_id: str, conversation_id: str, user_id: str, score: int = 80) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "interaction_id": "interaction-1",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "turn": 0,
        "within_turn_id": 0,
        "user_prompt": "How do I test this?",
        "model_response": "Run the test suite.",
        "model_name": "test-model",
        "model_provider": "test-provider",
        "score": score,
        "if_chosen": True,
        "included_in_balanced_subset": False,
    }


class PrismAdapterTests(unittest.TestCase):
    def test_split_assignment_keeps_each_user_in_one_split(self) -> None:
        assignments = assign_user_splits([f"user-{index}" for index in range(100)], seed=17)
        self.assertEqual(set(assignments), {f"user-{index}" for index in range(100)})
        self.assertEqual(set(assignments.values()), {"seed", "development", "held_out", "final_test"})

    def test_normalization_preserves_feedback_and_provenance(self) -> None:
        records = normalize_utterances(
            [utterance("ut-1", "conversation-a", "user-a")],
            {"conversation-a": conversation("conversation-a", "user-a")},
            split_by_user={"user-a": "development"},
            data_version="prism-fixture-v1",
            source_revision="fixture-revision",
        )
        record = records[0]
        self.assertEqual(record.task_id, PRISM_TASK_ID)
        self.assertEqual(record.source_group_id, "user-a:conversation-a")
        self.assertEqual(record.feedback["feedback_source"], "human_participant")
        self.assertEqual(record.metadata["source_revision"], "fixture-revision")
        self.assertEqual(record.metadata["conversation_feedback"]["open_feedback"], "Useful answer.")
        task_record = record.to_task_record()
        self.assertIsNone(task_record.target)
        self.assertNotIn("feedback", task_record.metadata)
        self.assertNotIn("conversation_feedback", task_record.metadata)
        self.assertNotIn("if_chosen", task_record.metadata)
        released = record.release_feedback()
        self.assertEqual(released.target, 80)
        self.assertTrue(released.metadata["feedback_released"])
        self.assertEqual(released.metadata["feedback"]["score"], 80)
        self.assertTrue(released.metadata["if_chosen"])

    def test_unknown_conversation_and_invalid_score_are_rejected(self) -> None:
        with self.assertRaises(PrismAdapterError):
            normalize_utterances(
                [utterance("ut-1", "missing", "user-a")], {},
                split_by_user={"user-a": "development"}, data_version="v1", source_revision="rev",
            )
        with self.assertRaises(PrismAdapterError):
            normalize_utterances(
                [utterance("ut-1", "conversation-a", "user-a", score=101)],
                {"conversation-a": conversation("conversation-a", "user-a")},
                split_by_user={"user-a": "development"}, data_version="v1", source_revision="rev",
            )

    def test_user_mismatch_and_final_test_are_auditable(self) -> None:
        with self.assertRaises(PrismAdapterError):
            normalize_utterances(
                [utterance("ut-1", "conversation-a", "user-b")],
                {"conversation-a": conversation("conversation-a", "user-a")},
                split_by_user={"user-b": "development"}, data_version="v1", source_revision="rev",
            )
        records = normalize_utterances(
            [utterance("ut-2", "conversation-a", "user-a")],
            {"conversation-a": conversation("conversation-a", "user-a")},
            split_by_user={"user-a": "final_test"}, data_version="v1", source_revision="rev",
        )
        registry = build_task_registry(records)
        with self.assertRaises(ValueError):
            registry.records_for(PRISM_TASK_ID, split="final_test")
        self.assertEqual(len(registry.records_for(PRISM_TASK_ID, split="final_test", include_final_test=True)), 1)
        self.assertIsNone(registry.records["prism-ut-2"].target)

    def test_raw_loader_hides_final_test_by_default(self) -> None:
        user_id = next(user for user in (f"user-{index}" for index in range(100))
                       if assign_user_splits([user], seed=17)[user] == "final_test")
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "conversations.jsonl").write_text(
                json.dumps(conversation("conversation-a", user_id)) + "\n", encoding="utf-8"
            )
            (raw_dir / "utterances.jsonl").write_text(
                json.dumps(utterance("ut-1", "conversation-a", user_id)) + "\n", encoding="utf-8"
            )
            hidden, hidden_report = load_prism_records(raw_dir)
            visible, visible_report = load_prism_records(raw_dir, include_final_test=True)
        self.assertEqual(hidden, [])
        self.assertEqual(hidden_report["include_final_test"], False)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible_report["sealed_split"], "final_test")


if __name__ == "__main__":
    unittest.main()
