from __future__ import annotations

import random
from typing import Any

from core_system.mle_core.interfaces import PolicyAdapter
from core_system.mle_core.models import ActionProposal, Observation


class FixedDataPipelinePolicy(PolicyAdapter):
    """The frozen no-feedback Aladdin baseline."""

    def __init__(self, *, data_split: str = "seed", report_final_test: bool = False) -> None:
        self.data_split = data_split
        self.report_final_test = report_final_test
        self.stage = 0

    def propose(self, observation: Observation) -> ActionProposal:
        if self.stage == 0:
            self.stage += 1
            return ActionProposal(
                action_type="select_records",
                arguments={"operation": "select_records", "version": "1.0", "split": self.data_split},
                expected_versions={"harness": observation.config.harness_version, "data": observation.config.data_version},
                rationale="Use the frozen seed data pipeline.",
            )
        if self.stage == 1:
            self.stage += 1
            return ActionProposal(
                action_type="train",
                arguments={"dataset_artifact_id": _latest_dataset(observation)},
                expected_versions={"harness": observation.config.harness_version},
                rationale="Train the frozen baseline model.",
            )
        if self.stage == 2:
            self.stage += 1
            return _evaluate(observation, "development")
        if self.stage == 3:
            self.stage += 1
            return _evaluate(observation, "held_out")
        if self.stage == 4 and self.report_final_test:
            self.stage += 1
            return _evaluate(observation, "final_test", allow_final_test=True)
        return ActionProposal(action_type="finish", rationale="Complete fixed baseline.")


class FeedbackPipelinePolicy(PolicyAdapter):
    """Shared local replay pipeline for random, uniform, and adaptive policies."""

    def __init__(self, *, mode: str, seed: int, candidate_order: tuple[str, ...] | None = None, report_final_test: bool = False) -> None:
        if mode not in {
            "random-feedback", "uniform-budget", "adaptive",
            "no-feedback-influence", "no-human-calibration",
            "no-training-gain-allocation",
        }:
            raise ValueError(f"Unknown Aladdin mode: {mode}")
        self.mode = mode
        self.random = random.Random(seed)
        self.report_final_test = report_final_test
        self.candidate_order = candidate_order or (
            "candidate-account",
            "candidate-billing",
            "candidate-access",
            "candidate-invoice",
        )
        if mode == "random-feedback":
            shuffled = list(self.candidate_order)
            self.random.shuffle(shuffled)
            self.candidate_order = tuple(shuffled)
        self.accepted_candidates: tuple[str, ...] = ()
        self.stage = 0

    def propose(self, observation: Observation) -> ActionProposal:
        if self.stage == 0:
            self.stage += 1
            return ActionProposal(
                action_type="select_records",
                arguments={"operation": "select_records", "version": "1.0", "split": "seed"},
                expected_versions={"harness": observation.config.harness_version, "data": observation.config.data_version},
                rationale="Establish the common seed data starting point.",
            )
        if self.stage == 1:
            self.stage += 1
            return ActionProposal(
                action_type="train",
                arguments={"dataset_artifact_id": _latest_dataset(observation)},
                expected_versions={"harness": observation.config.harness_version},
                rationale="Train the common pre-feedback model.",
            )
        if self.stage == 2:
            self.stage += 1
            return _evaluate(observation, "development")
        if self.stage == 3:
            self.stage += 1
            return ActionProposal(
                action_type="generate_records",
                arguments={
                    "operation": "generate_template_records",
                    "version": "1.0",
                    "task_id": observation.config.task_id,
                    "split": "seed",
                    "include_input_records": True,
                    "strategy": f"{self.mode}-replay",
                    "templates": _candidate_templates(self.candidate_order),
                },
                expected_versions={"harness": observation.config.harness_version, "data": observation.config.data_version},
                rationale="Create a deterministic candidate pool for feedback.",
            )
        if self.stage == 4:
            self.stage += 1
            return ActionProposal(
                action_type="request_feedback",
                arguments={"target_artifact_id": _latest_dataset(observation), "target_version": observation.config.data_version},
                expected_versions={"data": observation.config.data_version},
                rationale="Request one replayed judgment on the candidate pool.",
            )
        if self.stage == 5:
            self.stage += 1
            accepted = _accepted_candidates(observation)
            feedback = observation.feedback[-1].value if observation.feedback else {}
            if self.mode == "uniform-budget":
                accepted = _select_uniform_by_label(accepted, feedback, self.random)
            elif self.mode == "random-feedback" and accepted:
                accepted = tuple(candidate for candidate in accepted if self.random.random() >= 0.5)
            elif self.mode in {
                "adaptive", "no-feedback-influence", "no-human-calibration",
                "no-training-gain-allocation",
            }:
                accepted = _select_adaptive_candidates(accepted, feedback, self.mode)
            self.accepted_candidates = accepted
            return ActionProposal(
                action_type="transform_records",
                arguments={
                    "operation": "apply_feedback_labels",
                    "version": "1.0",
                    "input_artifact_id": _latest_dataset(observation),
                    "accepted_record_ids": list(accepted),
                    "labels": feedback.get("labels", {}) if isinstance(feedback, dict) else {},
                    "selection_diagnostics": {
                        "mode": self.mode,
                        "feedback_influence_used": self.mode != "no-feedback-influence",
                        "human_calibration_used": self.mode != "no-human-calibration",
                        "training_gain_used": self.mode != "no-training-gain-allocation",
                    },
                },
                expected_versions={"harness": observation.config.harness_version, "data": observation.config.data_version},
                rationale=("Apply feedback labels to accepted candidates." if self.mode == "adaptive" else "Apply the configured replay budget rule."),
            )
        if self.stage == 6:
            self.stage += 1
            return ActionProposal(
                action_type="select_records",
                arguments={
                    "operation": "select_records",
                    "version": "1.0",
                    "input_artifact_id": _latest_dataset(observation),
                    "record_ids": ["seed-001", "seed-002", "seed-003", "seed-004", *self.accepted_candidates],
                },
                expected_versions={"harness": observation.config.harness_version, "data": observation.config.data_version},
                rationale="Build the training set from seed and feedback-labeled candidates.",
            )
        if self.stage == 7:
            self.stage += 1
            return ActionProposal(
                action_type="train",
                arguments={"dataset_artifact_id": _latest_dataset(observation), "config": {"steps": 1}},
                expected_versions={"harness": observation.config.harness_version},
                rationale="Retrain on feedback-conditioned data.",
            )
        if self.stage == 8:
            self.stage += 1
            return _evaluate(observation, "development")
        if self.stage == 9:
            self.stage += 1
            return _evaluate(observation, "held_out")
        if self.stage == 10 and self.report_final_test:
            self.stage += 1
            return _evaluate(observation, "final_test", allow_final_test=True)
        return ActionProposal(action_type="finish", rationale="Complete replay experiment.")


class RandomFeedbackPolicy(FeedbackPipelinePolicy):
    def __init__(self, *, seed: int, report_final_test: bool = False) -> None:
        super().__init__(mode="random-feedback", seed=seed, report_final_test=report_final_test)


class UniformBudgetPolicy(FeedbackPipelinePolicy):
    def __init__(self, *, seed: int, report_final_test: bool = False) -> None:
        super().__init__(mode="uniform-budget", seed=seed, report_final_test=report_final_test)


class AdaptiveAladdinPolicy(FeedbackPipelinePolicy):
    def __init__(self, *, seed: int, report_final_test: bool = False) -> None:
        super().__init__(mode="adaptive", seed=seed, report_final_test=report_final_test)


class AblationPolicy(FeedbackPipelinePolicy):
    def __init__(self, *, mode: str, seed: int, report_final_test: bool = False) -> None:
        super().__init__(mode=mode, seed=seed, report_final_test=report_final_test)


def _latest_dataset(observation: Observation) -> str:
    for artifact in reversed(observation.artifacts):
        if artifact.kind == "dataset":
            return artifact.artifact_id
    raise RuntimeError("No dataset artifact is available")


def _evaluate(observation: Observation, split: str, *, allow_final_test: bool = False) -> ActionProposal:
    arguments: dict[str, Any] = {"split": split}
    if allow_final_test:
        arguments["allow_final_test"] = True
    return ActionProposal(
        action_type="evaluate",
        arguments=arguments,
        expected_versions={"harness": observation.config.harness_version},
        rationale=f"Evaluate the current model on {split}.",
    )


def _candidate_templates(candidate_order: tuple[str, ...]) -> list[dict[str, Any]]:
    templates = {
        "candidate-account": {"record_id": "candidate-account", "input": "I cannot access my user account", "target": None, "source_group_id": "generated-account"},
        "candidate-billing": {"record_id": "candidate-billing", "input": "please provide my order invoice", "target": None, "source_group_id": "generated-billing"},
        "candidate-access": {"record_id": "candidate-access", "input": "the customer portal refuses my credentials", "target": None, "source_group_id": "generated-account"},
        "candidate-invoice": {"record_id": "candidate-invoice", "input": "I was charged and need the transaction document", "target": None, "source_group_id": "generated-billing"},
    }
    return [templates[item] for item in candidate_order]


def _accepted_candidates(observation: Observation) -> tuple[str, ...]:
    if not observation.feedback:
        return ()
    value = observation.feedback[-1].value
    if not isinstance(value, dict) or value.get("decision") != "accept":
        return ()
    return tuple(value.get("accepted_record_ids", ()))


def _select_uniform_by_label(accepted: tuple[str, ...], feedback: Any, randomizer: random.Random) -> tuple[str, ...]:
    if not isinstance(feedback, dict):
        return accepted
    labels = feedback.get("labels", {})
    by_label: dict[str, list[str]] = {}
    for candidate in accepted:
        label = labels.get(candidate)
        if label is not None:
            by_label.setdefault(str(label), []).append(candidate)
    return tuple(randomizer.choice(by_label[label]) for label in sorted(by_label))


def _select_adaptive_candidates(accepted: tuple[str, ...], feedback: Any, mode: str) -> tuple[str, ...]:
    if not isinstance(feedback, dict):
        return accepted
    influence = feedback.get("influence_scores", {})
    calibration = feedback.get("calibration", {})
    gains = feedback.get("training_gains", {})
    selected = []
    for candidate in accepted:
        influence_score = influence.get(candidate)
        calibrated = calibration.get(candidate)
        training_gain = gains.get(candidate)
        if mode != "no-feedback-influence" and influence_score is not None and float(influence_score) < 0.5:
            continue
        if mode != "no-human-calibration" and calibrated is not None and not bool(calibrated):
            continue
        if mode != "no-training-gain-allocation" and training_gain is not None and float(training_gain) <= 0.0:
            continue
        selected.append(candidate)
    return tuple(selected)
