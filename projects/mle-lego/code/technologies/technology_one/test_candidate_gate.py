from __future__ import annotations

import tempfile
import unittest

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion
from core_system.mle_core.store import ArtifactStore
from technologies.technology_one.candidate_gate import (
    CandidateGateStatus,
    CandidateSafetyGate,
    SafetyChecklist,
)


def version(version_id: str, parent: str | None = None) -> DataSystemVersion:
    return DataSystemVersion(
        version_id=version_id,
        data_version=f"data-{version_id}",
        task_model_version=f"model-{version_id}",
        harness_version=f"harness-{version_id}",
        memory_version=f"memory-{version_id}",
        skill_version=f"skill-{version_id}",
        evaluator_version="evaluator-v1",
        parent_version_id=parent,
    )


class CandidateGateTests(unittest.TestCase):
    def test_safety_pass_only_qualifies_candidate_for_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            stable = version("stable-v1")
            candidate = CandidateVersion(
                "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-1"
            )
            result = CandidateSafetyGate(store).assess(
                stable_version=stable,
                candidate=candidate,
                checklist=SafetyChecklist(True, True, True, True, True, True),
            )

        self.assertEqual(result.status, CandidateGateStatus.ELIGIBLE_FOR_OBSERVATION)
        self.assertEqual(stable.version_id, "stable-v1")
        self.assertEqual(result.candidate.version.version_id, "candidate-v2")

    def test_failed_check_rejects_candidate_and_records_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            result = CandidateSafetyGate(store).assess(
                stable_version=version("stable-v1"),
                candidate=CandidateVersion(
                    "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-1"
                ),
                checklist=SafetyChecklist(True, True, False, True, True, True),
            )
            events = store.read_events()

        self.assertEqual(result.status, CandidateGateStatus.REJECTED)
        self.assertEqual(result.failed_checks, ("data_valid",))
        self.assertEqual(events[-1]["event_type"], "candidate_safety_assessed")

    def test_stale_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gate = CandidateSafetyGate(ArtifactStore(directory))
            with self.assertRaisesRegex(ValueError, "current stable version"):
                gate.assess(
                    stable_version=version("stable-v2"),
                    candidate=CandidateVersion(
                        "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-1"
                    ),
                    checklist=SafetyChecklist(True, True, True, True, True, True),
                )


if __name__ == "__main__":
    unittest.main()
