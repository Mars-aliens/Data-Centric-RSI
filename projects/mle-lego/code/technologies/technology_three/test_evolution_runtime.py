from __future__ import annotations

import tempfile
import unittest

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion, FeedbackEvent
from core_system.mle_core.store import ArtifactStore
from integration.runtime import DataSystemEvolutionRuntime
from technologies.technology_one.candidate_gate import SafetyChecklist
from technologies.technology_three.version_selection import (
    PairedMeanRetentionPolicy,
    SelectionAction,
    WindowedVersionSelector,
)
from technologies.technology_two.experience_lifecycle import (
    ExperienceRecord,
    ExperienceStatus,
)


def version(version_id: str, parent: str | None = None) -> DataSystemVersion:
    return DataSystemVersion(
        version_id,
        f"data-{version_id}",
        f"model-{version_id}",
        f"harness-{version_id}",
        f"memory-{version_id}",
        f"skill-{version_id}",
        "evaluator-v1",
        parent_version_id=parent,
    )


class EvolutionRuntimeTests(unittest.TestCase):
    def test_runtime_connects_experience_candidate_safety_and_final_retention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            runtime = DataSystemEvolutionRuntime(
                stable_version=version("stable-v1"),
                store=store,
                selector=WindowedVersionSelector(
                    store,
                    policy=PairedMeanRetentionPolicy(min_batches=3, min_mean_gain=0.01),
                ),
            )
            experience = ExperienceRecord(
                "skill-1",
                "skill",
                {"procedure": ["check provenance"]},
                {"task": "classification"},
                "stable-v1",
                "run-1",
                "batch-1",
            )
            runtime.record_experience(experience)
            runtime.experiences.validate("skill-1")
            self.assertEqual(
                runtime.retrieve_experiences({"task": "classification"})[0].status,
                ExperienceStatus.VERIFIED,
            )

            feedback = FeedbackEvent(
                "feedback-1", "stable-v1", "run-1", "batch-1", 1, 3,
                "human_preference", {"choice": "a"}, "t1", "t3",
            )
            self.assertEqual(runtime.attribute_feedback(feedback), "current_version")

            candidate = CandidateVersion(
                "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-2"
            )
            started = runtime.start_candidate(
                candidate,
                checklist=SafetyChecklist(True, True, True, True, True, True),
            )
            self.assertTrue(started.entered_observation)
            self.assertEqual(runtime.stable_version.version_id, "stable-v1")

            for index, candidate_score in enumerate((0.61, 0.62, 0.60), start=1):
                runtime.observe_candidate_batch(
                    "candidate-1",
                    batch_id=f"batch-{index + 2}",
                    stable_score=0.60,
                    candidate_score=candidate_score,
                )
            decision = runtime.decide_candidate("candidate-1")
            self.assertEqual(decision.action, SelectionAction.ACCEPT_CANDIDATE)
            self.assertEqual(runtime.stable_version.version_id, "candidate-v2")
            restored = runtime.restore_stable_version(
                "stable-v1", reason="post-promotion long-horizon regression"
            )
            self.assertEqual(restored.version_id, "stable-v1")

    def test_runtime_cannot_restore_an_unpromoted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DataSystemEvolutionRuntime(
                stable_version=version("stable-v1"),
                store=ArtifactStore(directory),
            )
            with self.assertRaisesRegex(ValueError, "previously promoted"):
                runtime.restore_stable_version("candidate-v2", reason="not promoted")

    def test_runtime_replays_candidate_window_experiences_and_stable_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_store = ArtifactStore(directory)
            first = DataSystemEvolutionRuntime(
                stable_version=version("stable-v1"),
                store=first_store,
                selector=WindowedVersionSelector(
                    first_store,
                    policy=PairedMeanRetentionPolicy(min_batches=3, min_mean_gain=0.01),
                ),
            )
            first.record_experience(
                ExperienceRecord(
                    "skill-1", "skill", {"procedure": ["check"]},
                    {"task": "classification"}, "stable-v1", "run-1", "batch-1",
                )
            )
            first.experiences.validate("skill-1")
            first.start_candidate(
                CandidateVersion(
                    "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-2"
                ),
                checklist=SafetyChecklist(True, True, True, True, True, True),
            )
            first.observe_candidate_batch(
                "candidate-1", batch_id="batch-3", stable_score=0.5, candidate_score=0.7
            )

            resumed_store = ArtifactStore(directory)
            resumed = DataSystemEvolutionRuntime(
                stable_version=version("stable-v1"),
                store=resumed_store,
                selector=WindowedVersionSelector(
                    resumed_store,
                    policy=PairedMeanRetentionPolicy(min_batches=3, min_mean_gain=0.01),
                ),
            )
            self.assertEqual(
                [item.experience_id for item in resumed.retrieve_experiences({"task": "classification"})],
                ["skill-1"],
            )
            resumed.observe_candidate_batch(
                "candidate-1", batch_id="batch-4", stable_score=0.5, candidate_score=0.6
            )
            resumed.observe_candidate_batch(
                "candidate-1", batch_id="batch-5", stable_score=0.5, candidate_score=0.6
            )
            decision = resumed.decide_candidate("candidate-1")
            self.assertEqual(decision.action, SelectionAction.ACCEPT_CANDIDATE)

            restored_runtime = DataSystemEvolutionRuntime(
                stable_version=version("stable-v1"),
                store=ArtifactStore(directory),
                selector=WindowedVersionSelector(ArtifactStore(directory)),
            )
            self.assertEqual(restored_runtime.stable_version.version_id, "candidate-v2")
            restored_runtime.restore_stable_version(
                "stable-v1", reason="resumed runtime detected a later regression"
            )
            self.assertEqual(restored_runtime.stable_version.version_id, "stable-v1")

    def test_unsafe_candidate_never_enters_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DataSystemEvolutionRuntime(
                stable_version=version("stable-v1"),
                store=ArtifactStore(directory),
            )
            candidate = CandidateVersion(
                "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-1"
            )
            result = runtime.start_candidate(
                candidate,
                checklist=SafetyChecklist(True, False, True, True, True, True),
            )

        self.assertFalse(result.entered_observation)
        self.assertEqual(runtime.stable_version.version_id, "stable-v1")


if __name__ == "__main__":
    unittest.main()
