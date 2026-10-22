from __future__ import annotations

import tempfile
import unittest

from core_system.mle_core.store import ArtifactStore
from technologies.technology_two.experience_lifecycle import (
    ExperienceLifecycleStore,
    ExperienceRecord,
    ExperienceStatus,
)


def experience(experience_id: str, *, conditions: dict | None = None) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        kind="skill",
        content={"steps": ["check input", "run operation"]},
        conditions=conditions or {"task": "classification"},
        source_version_id="system-v1",
        source_run_id="run-1",
        source_batch_id="batch-1",
    )


class ExperienceLifecycleTests(unittest.TestCase):
    def test_only_verified_condition_matching_experience_is_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = ExperienceLifecycleStore(ArtifactStore(directory))
            pending = experience("pending")
            verified = experience("verified")
            repository.add(pending)
            repository.add(verified)
            repository.validate("verified")

            matches = repository.retrieve({"task": "classification", "model": "m1"})
            mismatch = repository.retrieve({"task": "generation"})

        self.assertEqual([item.experience_id for item in matches], ["verified"])
        self.assertEqual(mismatch, [])

    def test_repeated_failures_require_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = ExperienceLifecycleStore(ArtifactStore(directory), failure_limit=2)
            repository.add(experience("skill-1"))
            repository.validate("skill-1")
            repository.record_outcome("skill-1", succeeded=False)
            result = repository.record_outcome("skill-1", succeeded=False)

        self.assertEqual(result.status, ExperienceStatus.NEEDS_VALIDATION)
        self.assertEqual(repository.retrieve({"task": "classification"}), [])

    def test_context_change_and_conflict_remove_experience_from_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = ExperienceLifecycleStore(ArtifactStore(directory))
            repository.add(experience("skill-a"))
            repository.add(experience("skill-b"))
            repository.validate("skill-a")
            repository.validate("skill-b")
            repository.mark_conflicted(["skill-a", "skill-b"], conflict_group_id="conflict-1")
            repository.resolve_conflict("conflict-1", preferred_id="skill-a")

            self.assertEqual(repository.retrieve({"task": "classification"}), [])
            repository.validate("skill-a")
            invalidated = repository.invalidate_for_context_change({"task"})
            status = repository.records["skill-a"].status

        self.assertEqual(invalidated, ["skill-a"])
        self.assertEqual(status, ExperienceStatus.NEEDS_VALIDATION)


if __name__ == "__main__":
    unittest.main()
