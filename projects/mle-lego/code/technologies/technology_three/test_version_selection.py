from __future__ import annotations

import tempfile
import unittest

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion, FeedbackEvent
from core_system.mle_core.store import ArtifactStore
from technologies.technology_one.candidate_gate import (
    CandidateSafetyGate,
    SafetyChecklist,
)
from technologies.technology_three.version_selection import (
    PairedMeanRetentionPolicy,
    SelectionAction,
    WindowedVersionSelector,
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


class VersionSelectionTests(unittest.TestCase):
    def test_candidate_is_not_selected_until_minimum_paired_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            stable = version("stable-v1")
            candidate = CandidateVersion(
                "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-0"
            )
            assessment = CandidateSafetyGate(store).assess(
                stable_version=stable,
                candidate=candidate,
                checklist=SafetyChecklist(True, True, True, True, True, True),
            )
            selector = WindowedVersionSelector(
                store,
                policy=PairedMeanRetentionPolicy(min_batches=3, min_mean_gain=0.01),
            )
            selector.register_candidate(assessment)
            selector.record_batch("candidate-1", batch_id="batch-1", stable_score=0.5, candidate_score=0.7)
            selector.record_batch("candidate-1", batch_id="batch-2", stable_score=0.5, candidate_score=0.6)
            interim = selector.decide("candidate-1", current_stable_version_id="stable-v1")
            selector.record_batch("candidate-1", batch_id="batch-3", stable_score=0.5, candidate_score=0.6)
            final = selector.decide("candidate-1", current_stable_version_id="stable-v1")

        self.assertEqual(interim.action, SelectionAction.CONTINUE_OBSERVING)
        self.assertEqual(final.action, SelectionAction.ACCEPT_CANDIDATE)
        self.assertEqual(final.observed_batches, 3)

    def test_candidate_failure_rejects_without_changing_stable_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            stable = version("stable-v1")
            candidate = CandidateVersion(
                "candidate-1", version("candidate-v2", "stable-v1"), "run-1", "batch-0"
            )
            assessment = CandidateSafetyGate(store).assess(
                stable_version=stable,
                candidate=candidate,
                checklist=SafetyChecklist(True, True, True, True, True, True),
            )
            selector = WindowedVersionSelector(store)
            selector.register_candidate(assessment)
            selector.record_batch(
                "candidate-1",
                batch_id="batch-1",
                stable_score=None,
                candidate_score=None,
                candidate_failed=True,
            )
            decision = selector.decide("candidate-1", current_stable_version_id="stable-v1")

        self.assertEqual(decision.action, SelectionAction.REJECT_FAILED_CANDIDATE)
        self.assertEqual(decision.stable_version_id, "stable-v1")

    def test_late_feedback_is_attributed_to_its_historical_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selector = WindowedVersionSelector(ArtifactStore(directory))
            feedback = FeedbackEvent(
                feedback_id="feedback-1",
                target_version_id="stable-v1",
                source_run_id="run-1",
                source_batch_id="batch-1",
                source_batch_index=1,
                released_batch_index=4,
                feedback_type="human_preference",
                value={"choice": "a"},
                produced_at="t1",
                received_at="t4",
            )
            attribution = selector.attribute_feedback(
                feedback,
                known_version_ids={"stable-v1", "stable-v2"},
                current_version_id="stable-v2",
            )

        self.assertEqual(attribution, "historical_version")

    def test_feedback_for_a_candidate_under_observation_is_not_called_historical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selector = WindowedVersionSelector(ArtifactStore(directory))
            feedback = FeedbackEvent(
                "feedback-candidate", "candidate-v2", "run-1", "batch-2", 2, 3,
                "human_preference", {"choice": "candidate"}, "t2", "t3",
            )
            attribution = selector.attribute_feedback(
                feedback,
                known_version_ids={"stable-v1", "candidate-v2"},
                candidate_version_ids={"candidate-v2"},
                current_version_id="stable-v1",
            )

        self.assertEqual(attribution, "candidate_version")


if __name__ == "__main__":
    unittest.main()
