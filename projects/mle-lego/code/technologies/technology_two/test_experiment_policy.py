from __future__ import annotations

import unittest

from technologies.technology_two.experiment_policy import ExperienceTrainingPolicy


def _records(labels: list[str]) -> list[dict[str, str]]:
    return [
        {"record_id": f"r{index}", "input": f"input {index}", "target": label}
        for index, label in enumerate(labels)
    ]


class ExperienceTrainingPolicyTests(unittest.TestCase):
    manifest = {"dataset_revision": "fixed-revision"}

    def test_reflection_arm_records_advice_without_changing_training_data(self) -> None:
        records = _records(["preferred"] * 8 + ["rejected"] * 2)
        selected, audit, state = ExperienceTrainingPolicy("reflection_only").prepare(
            records,
            newly_released=records,
            released_metrics=[],
            batch_index=0,
            state={},
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-0",
        )
        self.assertEqual(selected, records)
        self.assertEqual(audit["records_dropped"], 0)
        self.assertEqual(state["reflections"][0]["action"], "none")

    def test_governed_experience_caps_replay_and_records_audit(self) -> None:
        records = _records(["preferred"] * 8 + ["rejected"] * 2)
        selected, audit, state = ExperienceTrainingPolicy("lifecycle_governed").prepare(
            records,
            newly_released=records,
            released_metrics=[],
            batch_index=0,
            state={},
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-0",
        )
        self.assertEqual(selected, records)
        self.assertFalse(audit["experience_applied"])
        self.assertEqual(state["experience"]["status"], "candidate")

        selected, audit, state = ExperienceTrainingPolicy("lifecycle_governed").prepare(
            records,
            newly_released=records,
            released_metrics=[],
            batch_index=1,
            state=state,
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-1",
        )
        self.assertEqual(len(selected), 6)
        self.assertTrue(audit["experience_applied"])
        self.assertEqual(state["experience"]["status"], "verified")
        self.assertEqual(state["experience"]["source_version_id"], "model-v1")
        self.assertEqual(audit["experience_status_before"], "candidate")
        self.assertEqual(audit["experience_source_version_id"], "model-v1")
        self.assertEqual(audit["record_ids_selected"], [item["record_id"] for item in selected])
        self.assertEqual(len(audit["record_ids_selected_sha256"]), 64)

    def test_context_drift_requires_revalidation_before_reuse(self) -> None:
        policy = ExperienceTrainingPolicy("lifecycle_governed")
        first = _records(["preferred"] * 8 + ["rejected"] * 2)
        _, _, state = policy.prepare(
            first,
            newly_released=first,
            released_metrics=[],
            batch_index=0,
            state={},
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-0",
        )
        changed = _records(["tie"] * 8 + ["rejected"] * 2)
        selected, audit, state = policy.prepare(
            changed,
            newly_released=changed,
            released_metrics=[],
            batch_index=1,
            state=state,
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-1",
        )
        self.assertFalse(audit["experience_applied"])
        self.assertEqual(audit["experience_status"], "needs_validation")
        self.assertTrue(audit["context_changed"])
        self.assertEqual(audit["conditions_observed"]["dominant_label"], "tie")
        self.assertEqual(audit["experience_status_before"], "candidate")
        self.assertEqual(selected, changed)
        _, audit, state = policy.prepare(
            changed,
            newly_released=changed,
            released_metrics=[],
            batch_index=2,
            state=state,
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-2",
        )
        self.assertTrue(audit["experience_applied"])
        self.assertEqual(state["experience"]["conditions"]["dominant_label"], "tie")

    def test_unmanaged_experience_keeps_running_after_context_drift(self) -> None:
        policy = ExperienceTrainingPolicy("executable_unmanaged")
        first = _records(["preferred"] * 8 + ["rejected"] * 2)
        _, _, state = policy.prepare(
            first,
            newly_released=first,
            released_metrics=[],
            batch_index=0,
            state={},
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-0",
        )
        changed = _records(["tie"] * 8 + ["rejected"] * 2)
        _, audit, _ = policy.prepare(
            changed,
            newly_released=changed,
            released_metrics=[],
            batch_index=1,
            state=state,
            task_manifest=self.manifest,
            run_id="test-run",
            source_version_id="model-v1",
            source_batch_id="batch-1",
        )
        self.assertTrue(audit["experience_applied"])
        self.assertEqual(audit["experience_status"], "verified")

    def test_reflection_storage_is_bounded(self) -> None:
        policy = ExperienceTrainingPolicy("reflection_only")
        state = {}
        rows = _records(["preferred", "rejected"])
        for batch_index in range(40):
            _, _, state = policy.prepare(
                rows,
                newly_released=rows,
                released_metrics=[],
                batch_index=batch_index,
                state=state,
                task_manifest=self.manifest,
                run_id="test-run",
                source_version_id="model-v1",
                source_batch_id=f"batch-{batch_index}",
            )
        self.assertEqual(len(state["reflections"]), 32)
        self.assertEqual(state["archived_reflection_count"], 8)


if __name__ == "__main__":
    unittest.main()
