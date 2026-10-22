from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from .long_horizon_task import (
    LongHorizonTaskConfig,
    TaskIntegrityError,
    prepare_helpsteer2_long_horizon_task,
    write_agent_preflight_bundle,
)


def rows_for(prefix: str, count: int, *, source_offset: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(source_offset, source_offset + count):
        prompt = f"{prefix} prompt {index}"
        for suffix, score in (("high", 4), ("low", 0)):
            rows.append(
                {
                    "prompt": prompt,
                    "response": f"{suffix} response",
                    "helpfulness": score,
                    "correctness": score,
                    "coherence": score,
                    "complexity": score,
                    "verbosity": score,
                }
            )
    return rows


class LongHorizonTaskTests(unittest.TestCase):
    def _task(self, delay: int = 2):
        return prepare_helpsteer2_long_horizon_task(
            rows_for("train", 800),
            rows_for("validation", 30),
            config=LongHorizonTaskConfig(
                stream_seed=23,
                groups_per_batch=20,
                feedback_delay_batches=delay,
            ),
        )

    def test_split_and_stream_manifest_are_reproducible(self) -> None:
        left = self._task().agent_view.manifest
        right = self._task().agent_view.manifest
        self.assertEqual(left, right)
        self.assertEqual(left["task_definition_version"], "mle-lego-long-horizon-task-v1")
        self.assertGreater(left["split_group_counts"]["development"], 0)
        self.assertEqual(left["split_group_counts"]["final_test"], 30)
        self.assertEqual(left["groups_per_batch"], 20)

    def test_max_batches_records_a_stream_prefix_and_preserves_feedback_timing(self) -> None:
        task = prepare_helpsteer2_long_horizon_task(
            rows_for("train", 800),
            rows_for("validation", 30),
            config=LongHorizonTaskConfig(
                stream_seed=23,
                groups_per_batch=20,
                feedback_delay_batches=2,
                max_batches=4,
            ),
        )
        manifest = task.agent_view.manifest
        self.assertEqual(task.scheduler.batch_count, 4)
        self.assertGreater(manifest["available_development_batch_count"], 4)
        self.assertEqual(manifest["development_batch_count"], 4)
        self.assertEqual(manifest["scheduled_development_batch_count"], 4)
        self.assertEqual(manifest["max_batches"], 4)
        self.assertTrue(manifest["stream_prefix_limited"])

        for expected_index in range(4):
            current = task.next_agent_input()
            self.assertEqual(current.batch.batch_index, expected_index)
            if expected_index < 3:
                self.assertFalse(current.released_feedback)
            else:
                self.assertTrue(
                    any(item.source_batch_index == 0 for item in current.released_feedback)
                )
            task.feedback_oracle.submit_predictions(
                source_batch_index=expected_index,
                model_version_id="model-v1",
                predictions={
                    record.record_id: "preferred" for record in current.batch.records
                },
            )

        with self.assertRaises(StopIteration):
            task.next_agent_input()

    def test_agent_view_has_seed_labels_but_no_stream_gold_or_scores(self) -> None:
        task = self._task()
        current = task.next_agent_input().to_dict()
        payload = json.dumps(
            {"agent_view": task.agent_view.to_dict(), "current_batch": current},
            ensure_ascii=False,
        )
        for forbidden in ('"scores"', '"target"', '"score_mean"'):
            # Seed targets are allowed; only check the current stream payload below.
            self.assertNotIn(forbidden, json.dumps(current, ensure_ascii=False))
        self.assertIn('"target"', payload)
        self.assertNotIn('"scores"', payload)
        self.assertNotIn('"score_mean"', payload)

    def test_feedback_and_prequential_metrics_wait_for_due_batch(self) -> None:
        task = self._task(delay=2)
        first = task.next_agent_input()
        self.assertEqual(first.batch.batch_index, 0)
        self.assertEqual(first.released_feedback, ())
        self.assertEqual(first.released_metrics, ())
        predictions = {record.record_id: "preferred" for record in first.batch.records}
        task.feedback_oracle.submit_predictions(
            source_batch_index=0,
            model_version_id="model-v1",
            predictions=predictions,
        )

        for expected_index in (1, 2):
            current = task.next_agent_input()
            self.assertEqual(current.batch.batch_index, expected_index)
            self.assertEqual(current.released_feedback, ())
            self.assertEqual(current.released_metrics, ())
            task.feedback_oracle.submit_predictions(
                source_batch_index=expected_index,
                model_version_id="model-v1",
                predictions={record.record_id: "preferred" for record in current.batch.records},
            )

        third = task.next_agent_input()
        feedback = [item for item in third.released_feedback if item.source_batch_index == 0]
        metrics = [item for item in third.released_metrics if item.source_batch_index == 0]
        self.assertEqual(len(feedback), len(first.batch.records))
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].available_from_batch, 3)
        self.assertEqual(metrics[0].model_version_id, "model-v1")
        self.assertEqual(metrics[0].num_records, len(first.batch.records))

    def test_zero_delay_is_available_only_at_the_next_batch_boundary(self) -> None:
        task = self._task(delay=0)
        first = task.next_agent_input()
        task.feedback_oracle.submit_predictions(
            source_batch_index=0,
            model_version_id="model-v1",
            predictions={record.record_id: "rejected" for record in first.batch.records},
        )
        second = task.next_agent_input()
        self.assertTrue(second.released_feedback)
        self.assertTrue(second.released_metrics)
        self.assertEqual(second.released_metrics[0].available_from_batch, 1)

    def test_feedback_state_can_resume_without_duplicate_release(self) -> None:
        task = self._task(delay=0)
        first = task.next_agent_input()
        task.feedback_oracle.submit_predictions(
            source_batch_index=0,
            model_version_id="model-v1",
            predictions={record.record_id: "preferred" for record in first.batch.records},
        )
        second = task.next_agent_input()
        self.assertEqual(len(second.released_feedback), len(first.batch.records))
        state = task.state_dict()

        resumed = self._task(delay=0)
        resumed.restore_state(state)
        third = resumed.next_agent_input()
        self.assertFalse(
            any(item.source_batch_index == 0 for item in third.released_feedback)
        )
        self.assertFalse(
            any(item.source_batch_index == 0 for item in third.released_metrics)
        )

    def test_inconsistent_resume_snapshot_is_rejected(self) -> None:
        task = self._task(delay=0)
        task.next_agent_input()
        state = task.state_dict()
        state["scheduler"]["next_batch_index"] = 0
        with self.assertRaisesRegex(ValueError, "snapshots are inconsistent"):
            self._task(delay=0).restore_state(state)

    def test_max_batches_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_batches must be positive"):
            LongHorizonTaskConfig(max_batches=0)

    def test_terminal_metrics_are_reported_without_agent_feedback(self) -> None:
        task = self._task(delay=0)
        observed = []
        while True:
            try:
                current = task.next_agent_input()
            except StopIteration:
                break
            observed.extend(current.released_metrics)
            task.feedback_oracle.submit_predictions(
                source_batch_index=current.batch.batch_index,
                model_version_id="model-v1",
                predictions={record.record_id: "preferred" for record in current.batch.records},
            )
        terminal = task.feedback_oracle.finalize_prequential_metrics()
        combined = observed + list(terminal)
        source_batches = {item.source_batch_index for item in combined}
        self.assertEqual(source_batches, set(range(task.scheduler.batch_count)))
        self.assertEqual(len(combined), task.scheduler.batch_count)

    def test_terminal_metrics_leave_not_yet_due_feedback_pending(self) -> None:
        task = self._task(delay=5)
        while True:
            try:
                current = task.next_agent_input()
            except StopIteration:
                break
            task.feedback_oracle.submit_predictions(
                source_batch_index=current.batch.batch_index,
                model_version_id="model-v1",
                predictions={record.record_id: "preferred" for record in current.batch.records},
            )
        terminal = task.feedback_oracle.finalize_prequential_metrics()
        self.assertLess(len(terminal), task.scheduler.batch_count)
        with self.assertRaisesRegex(ValueError, "finalized"):
            task.feedback_oracle.release_for_batch(task.scheduler.batch_count)

    def test_finalization_is_only_allowed_after_stream_end(self) -> None:
        task = self._task(delay=0)
        task.next_agent_input()
        with self.assertRaisesRegex(ValueError, "must be complete"):
            task.feedback_oracle.finalize_prequential_metrics()

    def test_preflight_bundle_contains_no_development_gold_labels(self) -> None:
        task = self._task(delay=2)
        with TemporaryDirectory() as directory:
            manifest = write_agent_preflight_bundle(task, Path(directory) / "bundle")
            root = Path(directory) / "bundle"
            batch_text = (root / "runner-only" / "development-batches.jsonl").read_text(
                encoding="utf-8"
            )
            schedule_text = (root / "runner-only" / "feedback-schedule.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn('"target"', batch_text)
            self.assertNotIn('"scores"', batch_text)
            self.assertNotIn('"target"', schedule_text)
            self.assertNotIn('"scores"', schedule_text)
            self.assertEqual(manifest["agent_bundle"]["preflight_only"], True)
            self.assertTrue((root / "task_manifest.json").is_file())
            self.assertTrue((root / "agent-visible" / "seed.jsonl").is_file())
            self.assertTrue((root / "agent-visible" / "task_manifest.json").is_file())
            self.assertFalse(
                (root / "agent-visible" / "development-batches.jsonl").exists()
            )
            self.assertFalse(
                (root / "agent-visible" / "feedback-schedule.jsonl").exists()
            )

    def test_predictions_must_be_complete_and_submitted_for_open_batch(self) -> None:
        task = self._task()
        current = task.next_agent_input()
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            task.feedback_oracle.submit_predictions(
                source_batch_index=0,
                model_version_id="model-v1",
                predictions={},
            )
        with self.assertRaisesRegex(ValueError, "source batch is active"):
            task.feedback_oracle.submit_predictions(
                source_batch_index=1,
                model_version_id="model-v1",
                predictions={record.record_id: "preferred" for record in current.batch.records},
            )

    def test_prompt_group_cannot_cross_official_source_splits(self) -> None:
        train = rows_for("train", 20)
        validation = rows_for("validation", 20)
        validation[0]["prompt"] = train[0]["prompt"]
        validation[1]["prompt"] = train[0]["prompt"]
        with self.assertRaisesRegex(TaskIntegrityError, "cross official"):
            prepare_helpsteer2_long_horizon_task(train, validation)

        singleton_validation = rows_for("validation", 20)[2:]
        singleton_validation.append(
            {
                "prompt": train[0]["prompt"],
                "response": "single overlapping response",
                "helpfulness": 2,
                "correctness": 2,
                "coherence": 2,
                "complexity": 2,
                "verbosity": 2,
            }
        )
        with self.assertRaisesRegex(TaskIntegrityError, "cross official"):
            prepare_helpsteer2_long_horizon_task(train, singleton_validation)

    def test_invalid_scores_and_delay_are_rejected(self) -> None:
        invalid = rows_for("train", 20)
        invalid[0]["helpfulness"] = 9
        with self.assertRaisesRegex(TaskIntegrityError, "must be in \\[0, 4\\]"):
            prepare_helpsteer2_long_horizon_task(invalid, rows_for("validation", 20))
        with self.assertRaisesRegex(ValueError, "one of 0, 2, 5"):
            LongHorizonTaskConfig(feedback_delay_batches=1)


if __name__ == "__main__":
    unittest.main()
