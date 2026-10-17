from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core_system.mle_core.budget import BudgetExceeded
from core_system.mle_core.components import (
    ExactMatchEvaluator,
    GenerateTemplateRecords,
    MajorityLabelTrainer,
    SelectRecords,
)
from core_system.mle_core.feedback import InMemoryFeedbackProvider
from core_system.mle_core.interfaces import OperationRegistry, PolicyAdapter
from core_system.mle_core.models import ActionProposal, Budget, Observation, RunConfig, TaskRecord, TaskSpec
from core_system.mle_core.registry import SplitViolation, TaskRegistry
from core_system.mle_core.runtime import FixedPipelinePolicy, RunOrchestrator
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)


class CoreSystemTests(unittest.TestCase):
    def test_policy_estimated_cost_is_charged_to_budget(self) -> None:
        class CostedPolicy(PolicyAdapter):
            def propose(self, observation):
                if observation.batch_index == 0:
                    return ActionProposal(
                        action_type="finish",
                        estimated_cost=Budget(llm_calls=1, llm_tokens=7),
                    )
                raise AssertionError("finish should stop the run")

        operations = OperationRegistry()
        operations.register(SelectRecords())
        with tempfile.TemporaryDirectory() as directory:
            run = RunOrchestrator(
                registry=self.registry,
                store_root=Path(directory),
                operations=operations,
                trainer=MajorityLabelTrainer(),
                evaluator=ExactMatchEvaluator(),
            ).create_run(
                config=RunConfig(
                    run_id="policy-cost",
                    task_id="task",
                    backbone_llm="test",
                    task_model="test",
                    evaluator="test",
                    harness_version="h0",
                    data_version="d0",
                    initial_checkpoint=None,
                    max_batches=2,
                    budget_limit=Budget(llm_calls=1, llm_tokens=7),
                    seed=1,
                ),
                policy=CostedPolicy(),
            )
            run.run()
            self.assertEqual(run.budget.used.llm_calls, 1)
            self.assertEqual(run.budget.used.llm_tokens, 7)

    def setUp(self) -> None:
        self.registry = TaskRegistry()
        self.registry.register_task(
            TaskSpec(
                task_id="task",
                name="test",
                input_schema="text",
                target_schema="label",
            )
        )
        self.registry.add_records(
            [
                TaskRecord("seed-a", "task", "seed", "a", "yes"),
                TaskRecord("seed-b", "task", "seed", "b", "no"),
                TaskRecord("dev-a", "task", "development", "a2", "yes"),
                TaskRecord("dev-b", "task", "development", "b2", "no"),
                TaskRecord("held-a", "task", "held_out", "ha", "yes"),
                TaskRecord("held-b", "task", "held_out", "hb", "no"),
            ]
        )

    def test_fixed_pipeline_runs_and_logs(self) -> None:
        operations = OperationRegistry()
        operations.register(SelectRecords())
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                run_id="test-run",
                task_id="task",
                backbone_llm="backbone",
                task_model="task-model",
                evaluator="evaluator",
                harness_version="h0",
                data_version="d0",
                initial_checkpoint=None,
                max_batches=6,
                budget_limit=Budget(training_steps=2, evaluator_calls=2),
                seed=1,
            )
            orchestrator = RunOrchestrator(
                registry=self.registry,
                store_root=directory,
                operations=operations,
                trainer=MajorityLabelTrainer(),
                evaluator=ExactMatchEvaluator(),
            )
            run = orchestrator.create_run(
                config=config,
                policy=FixedPipelinePolicy(),
            )
            run.run()
            self.assertEqual(run.status.value, "completed")
            self.assertIn("exact_match", run.metrics)
            self.assertEqual(len(run.store.read_events()) > 0, True)
            self.assertTrue(
                (Path(directory) / "test-run" / "runs" / "test-run.json").exists()
            )

    def test_final_test_is_blocked_by_default(self) -> None:
        self.registry.add_records(
            [TaskRecord("test-a", "task", "final_test", "x", "yes")]
        )
        with self.assertRaises(SplitViolation):
            self.registry.records_for("task", split="final_test")

    def test_feedback_provider_and_generation_operation(self) -> None:
        operation = GenerateTemplateRecords()
        with tempfile.TemporaryDirectory() as directory:
            from core_system.mle_core.store import ArtifactStore

            records, _ = operation.run(
                arguments={
                    "task_id": "task",
                    "split": "development",
                    "strategy": "test-generation",
                    "templates": [
                        {
                            "record_id": "generated-1",
                            "input": "new example",
                            "target": "yes",
                        }
                    ],
                },
                records=[],
                store=ArtifactStore(directory),
            )
            self.assertEqual(records[0].record_id, "generated-1")
            provider = InMemoryFeedbackProvider()
            feedback = provider.request(
                target_artifact_id="artifact-1",
                target_version="v0",
            )
            self.assertEqual(feedback.provider_type, "human_replay")

    def test_tfidf_model_runs_on_local_records(self) -> None:
        trainer = TfidfLogisticTrainer()
        evaluator = TfidfLogisticEvaluator()
        checkpoint, cost = trainer.train(
            train_records=self.registry.records_for("task", split="seed"),
            initial_checkpoint=None,
            config={"ngram_range": (1, 1)},
            seed=3,
            store=None,
        )
        self.assertEqual(cost.training_steps, 1)
        metrics, _ = evaluator.evaluate(
            checkpoint=checkpoint,
            records=self.registry.records_for("task", split="development"),
            config={},
        )
        self.assertGreaterEqual(metrics["accuracy"], 0.0)

    def test_feedback_is_visible_with_provenance_and_metrics_are_split(self) -> None:
        operations = OperationRegistry()
        operations.register(SelectRecords())
        seen: list[Observation] = []

        class FeedbackPolicy:
            def __init__(self) -> None:
                self.stage = 0

            def propose(self, observation: Observation) -> ActionProposal:
                seen.append(observation)
                if self.stage == 0:
                    self.stage += 1
                    return ActionProposal(
                        action_type="select_records",
                        arguments={"operation": "select_records", "version": "1.0", "split": "seed"},
                    )
                if self.stage == 1:
                    self.stage += 1
                    return ActionProposal(
                        action_type="request_feedback",
                        arguments={"target_artifact_id": observation.artifacts[-1].artifact_id},
                    )
                if self.stage == 2:
                    self.stage += 1
                    return ActionProposal(
                        action_type="train",
                        arguments={"dataset_artifact_id": observation.artifacts[-1].artifact_id},
                    )
                if self.stage == 3:
                    self.stage += 1
                    return ActionProposal(action_type="evaluate", arguments={"split": "development"})
                if self.stage == 4:
                    self.stage += 1
                    return ActionProposal(action_type="evaluate", arguments={"split": "held_out"})
                return ActionProposal(action_type="finish")

        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                run_id="feedback-visible",
                task_id="task",
                backbone_llm="backbone",
                task_model="task-model",
                evaluator="evaluator",
                harness_version="h0",
                data_version="d0",
                initial_checkpoint=None,
                max_batches=8,
                budget_limit=Budget(training_steps=2, evaluator_calls=2, feedback_requests=1),
                seed=1,
            )
            run = RunOrchestrator(
                registry=self.registry,
                store_root=directory,
                operations=operations,
                trainer=MajorityLabelTrainer(),
                evaluator=ExactMatchEvaluator(),
                feedback_provider=InMemoryFeedbackProvider(sequence=[{"decision": "accept"}]),
            ).create_run(config=config, policy=FeedbackPolicy())
            run.run()

        self.assertEqual(run.feedback[0].target_artifact_id, run.artifacts[0].artifact_id)
        self.assertEqual(seen[2].feedback[0].feedback_id, "feedback-0001")
        self.assertEqual(run.metrics_by_split["development"]["num_records"], 2.0)
        self.assertEqual(run.metrics_by_split["held_out"]["num_records"], 2.0)

    def test_feedback_target_and_budget_are_validated(self) -> None:
        operations = OperationRegistry()
        operations.register(SelectRecords())

        class BadFeedbackPolicy:
            def __init__(self, target: str) -> None:
                self.target = target
                self.stage = 0

            def propose(self, observation: Observation) -> ActionProposal:
                if self.stage == 0:
                    self.stage += 1
                    return ActionProposal(
                        action_type="select_records",
                        arguments={"operation": "select_records", "version": "1.0", "split": "seed"},
                    )
                return ActionProposal(
                    action_type="request_feedback",
                    arguments={"target_artifact_id": self.target, "target_version": "wrong"},
                )

        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                run_id="bad-feedback",
                task_id="task",
                backbone_llm="backbone",
                task_model="task-model",
                evaluator="evaluator",
                harness_version="h0",
                data_version="d0",
                initial_checkpoint=None,
                max_batches=3,
                budget_limit=Budget(feedback_requests=0),
                seed=1,
            )
            run = RunOrchestrator(
                registry=self.registry,
                store_root=directory,
                operations=operations,
                trainer=MajorityLabelTrainer(),
                evaluator=ExactMatchEvaluator(),
                feedback_provider=InMemoryFeedbackProvider(),
            ).create_run(config=config, policy=BadFeedbackPolicy("missing"))
            with self.assertRaises(ValueError):
                run.run()
            self.assertEqual(run.budget.used.feedback_requests, 0)

            class ValidTargetPolicy(BadFeedbackPolicy):
                def propose(self, observation: Observation) -> ActionProposal:
                    if self.stage == 0:
                        self.stage += 1
                        return ActionProposal(
                            action_type="select_records",
                            arguments={"operation": "select_records", "version": "1.0", "split": "seed"},
                        )
                    return ActionProposal(
                        action_type="request_feedback",
                        arguments={"target_artifact_id": observation.artifacts[-1].artifact_id},
                    )

            budget_run = RunOrchestrator(
                registry=self.registry,
                store_root=directory,
                operations=operations,
                trainer=MajorityLabelTrainer(),
                evaluator=ExactMatchEvaluator(),
                feedback_provider=InMemoryFeedbackProvider(),
            ).create_run(config=config, policy=ValidTargetPolicy("unused"))
            with self.assertRaises(BudgetExceeded):
                budget_run.run()
            self.assertEqual(budget_run.budget.used.feedback_requests, 0)


if __name__ == "__main__":
    unittest.main()
