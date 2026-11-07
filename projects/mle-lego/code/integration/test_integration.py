from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion
from core_system.mle_core.store import ArtifactStore
from integration.experiment_matrix import all_feature_configs
from integration.orchestrator import BatchResult, IntegrationOrchestrator, IntegrationRunConfig
from integration.runtime import DataSystemEvolutionRuntime
from technologies.technology_one.candidate_gate import SafetyChecklist
from technologies.technology_three.version_selection import PairedMeanRetentionPolicy, WindowedVersionSelector


def make_version(version_id: str, parent: str | None = None) -> DataSystemVersion:
    return DataSystemVersion(
        version_id, f"data-{version_id}", f"model-{version_id}",
        f"harness-{version_id}", f"memory-{version_id}",
        f"skill-{version_id}", "evaluator-v1", parent_version_id=parent,
    )


def make_runtime(root: str) -> DataSystemEvolutionRuntime:
    store = ArtifactStore(root)
    return DataSystemEvolutionRuntime(
        stable_version=make_version("stable-v1"),
        store=store,
        selector=WindowedVersionSelector(
            store,
            policy=PairedMeanRetentionPolicy(min_batches=3, min_mean_gain=0.01, max_regression=0.05),
        ),
    )


class IntegrationTests(unittest.TestCase):
    def test_all_eight_feature_configs_are_distinct(self) -> None:
        configs = all_feature_configs()
        self.assertEqual(len(configs), 8)
        self.assertEqual(len({item.name for item in configs}), 8)

    def test_candidate_is_promoted_then_regression_is_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = make_runtime(directory)
            candidate = CandidateVersion("candidate-1", make_version("candidate-v2", "stable-v1"), "run", "batch-0")
            orchestrator = IntegrationOrchestrator(
                runtime=runtime,
                features=all_feature_configs()[-1],
                config=IntegrationRunConfig("integration", 4),
                candidate=candidate,
                checklist=SafetyChecklist(True, True, True, True, True, True),
            )

            def execute(index: int, active_version: str, features) -> BatchResult:
                return BatchResult(
                    index,
                    stable_score=0.60,
                    candidate_score=0.65,
                    active_score=0.40 if active_version == "candidate-v2" else 0.60,
                    feedback_version_id=active_version,
                )

            result = orchestrator.run(execute)
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.promoted_versions, ("candidate-v2",))
            self.assertEqual(result.rollback_count, 1)
            self.assertEqual(result.active_version_id, "stable-v1")

    def test_candidate_failure_is_recorded_without_stopping_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = make_runtime(directory)
            candidate = CandidateVersion("candidate-1", make_version("candidate-v2", "stable-v1"), "run", "batch-0")
            orchestrator = IntegrationOrchestrator(
                runtime=runtime,
                features=all_feature_configs()[-1],
                config=IntegrationRunConfig("failure", 4),
                candidate=candidate,
            )

            def execute(index: int, active_version: str, features) -> BatchResult:
                return BatchResult(index, 0.60, None if index == 1 else 0.65, index == 1, 0.60)

            result = orchestrator.run(execute)
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.completed_batches, 4)
            self.assertEqual(result.candidate_decision, "reject_failed_candidate")
            self.assertEqual(result.active_version_id, "stable-v1")

    def test_fixed_baseline_runs_without_candidate_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = make_runtime(directory)
            orchestrator = IntegrationOrchestrator(
                runtime=runtime,
                features=all_feature_configs()[0],
                config=IntegrationRunConfig("baseline", 4),
            )
            result = orchestrator.run(lambda i, active, features: BatchResult(i, 0.60, active_score=0.60))
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.completed_batches, 4)
            self.assertEqual(result.promoted_versions, ())


if __name__ == "__main__":
    unittest.main()
