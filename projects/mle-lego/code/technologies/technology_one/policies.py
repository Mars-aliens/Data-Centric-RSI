from __future__ import annotations

from core_system.mle_core.interfaces import PolicyAdapter
from core_system.mle_core.models import ActionProposal, Observation


class FixedHarnessReactPolicy(PolicyAdapter):
    """LEGO baseline: ReAct-style action selection over a fixed Harness."""

    def __init__(self, *, data_split: str = "seed") -> None:
        self.data_split = data_split
        self.stage = 0

    def propose(self, observation: Observation) -> ActionProposal:
        if self.stage == 0:
            self.stage += 1
            return ActionProposal(
                action_type="select_records",
                arguments={
                    "operation": "select_records",
                    "version": "1.0",
                    "split": self.data_split,
                },
                expected_versions={
                    "harness": observation.config.harness_version,
                    "data": observation.config.data_version,
                },
                rationale="Observe the fixed Harness input and select seed records.",
            )
        if self.stage == 1:
            self.stage += 1
            dataset = next(
                artifact
                for artifact in reversed(observation.artifacts)
                if artifact.kind == "dataset"
            )
            return ActionProposal(
                action_type="train",
                arguments={"dataset_artifact_id": dataset.artifact_id},
                expected_versions={"harness": observation.config.harness_version},
                rationale="Use the observed dataset to train the task model.",
            )
        if self.stage == 2:
            self.stage += 1
            return ActionProposal(
                action_type="evaluate",
                arguments={"split": "development"},
                expected_versions={"harness": observation.config.harness_version},
                rationale="Read the development result before deciding whether to finish.",
            )
        return ActionProposal(
            action_type="finish",
            rationale="Fixed Harness baseline has no persistent self-modification.",
        )
