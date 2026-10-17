from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import BudgetExceeded, BudgetLedger
from .interfaces import (
    Evaluator,
    FeedbackProvider,
    OperationRegistry,
    PolicyAdapter,
    TaskModelTrainer,
)
from .models import (
    ActionProposal,
    ArtifactRef,
    Budget,
    Observation,
    RunConfig,
    RunStatus,
    FeedbackRecord,
    TaskRecord,
    TaskSpec,
    to_jsonable,
)
from .registry import TaskRegistry
from .store import ArtifactStore


@dataclass
class CoreRun:
    config: RunConfig
    task: TaskSpec
    registry: TaskRegistry
    store: ArtifactStore
    operations: OperationRegistry
    trainer: TaskModelTrainer
    evaluator: Evaluator
    policy: PolicyAdapter
    feedback_provider: FeedbackProvider | None = None
    status: RunStatus = RunStatus.CREATED
    batch_index: int = 0
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_by_split: dict[str, dict[str, float]] = field(default_factory=dict)
    checkpoint: dict[str, Any] | None = None
    feedback: list[FeedbackRecord] = field(default_factory=list)
    budget: BudgetLedger | None = None

    def __post_init__(self) -> None:
        self.budget = BudgetLedger(self.config.budget_limit)

    @property
    def batch_id(self) -> str:
        return f"{self.config.run_id}-batch-{self.batch_index:04d}"

    def _event(self, event_type: str, **payload: Any) -> None:
        self.store.append_event(
            event_type,
            {
                "run_id": self.config.run_id,
                "batch_id": self.batch_id,
                **payload,
            },
        )

    def _snapshot(self) -> None:
        self.store.write_run_snapshot(
            self.config.run_id,
            {
                "config": to_jsonable(self.config),
                "status": self.status.value,
                "batch_index": self.batch_index,
                "artifacts": to_jsonable(self.artifacts),
                "metrics": self.metrics,
                "metrics_by_split": self.metrics_by_split,
                "checkpoint": self.checkpoint,
                "feedback": self.feedback,
                "used_budget": to_jsonable(self.budget.used if self.budget else Budget()),
            },
        )

    def start(self) -> None:
        if self.status != RunStatus.CREATED:
            raise RuntimeError(f"Cannot start run in status {self.status}")
        self.status = RunStatus.READY
        self._event("run_created", config=self.config)
        self._snapshot()

    def _observation(self) -> Observation:
        assert self.budget is not None
        return Observation(
            run_id=self.config.run_id,
            batch_id=self.batch_id,
            batch_index=self.batch_index,
            status=self.status,
            task=self.task,
            config=self.config,
            artifacts=tuple(self.artifacts),
            metrics=dict(self.metrics),
            pending_feedback=(),
            remaining_budget=self.budget.remaining(),
            feedback=tuple(self.feedback),
        )

    def _validate_proposal(self, proposal: ActionProposal) -> None:
        allowed = {
            "select_records",
            "transform_records",
            "filter_records",
            "generate_records",
            "train",
            "evaluate",
            "request_feedback",
            "finish",
        }
        if proposal.action_type not in allowed:
            raise ValueError(f"Unsupported core action: {proposal.action_type}")
        for key, expected in proposal.expected_versions.items():
            current = {
                "harness": self.config.harness_version,
                "data": self.config.data_version,
                "task": self.config.task_id,
            }.get(key)
            if current is None or current != expected:
                raise ValueError(
                    f"Version precondition failed for {key}: "
                    f"expected={expected}, current={current}"
                )
        assert self.budget is not None
        if self.budget.used.add(proposal.estimated_cost).exceeds(
            self.config.budget_limit
        ):
            raise BudgetExceeded("Proposal exceeds the declared run budget")

    def _records_from_argument(self, arguments: dict[str, Any]) -> list[TaskRecord]:
        input_artifact_id = arguments.get("input_artifact_id")
        if input_artifact_id:
            value = self.store.get_json(input_artifact_id)
            return [
                TaskRecord(
                    record_id=item["record_id"],
                    task_id=item["task_id"],
                    split=item["split"],
                    input=item["input"],
                    target=item.get("target"),
                    source_group_id=item.get("source_group_id"),
                    metadata=item.get("metadata", {}),
                    parent_record_ids=tuple(item.get("parent_record_ids", [])),
                )
                for item in value
            ]
        split = arguments.get("split", "seed")
        include_final_test = split == "final_test" and arguments.get(
            "allow_final_test", False
        )
        return self.registry.records_for(
            self.config.task_id,
            split=split,
            include_final_test=include_final_test,
        )

    def _run_data_operation(self, proposal: ActionProposal) -> None:
        operation_name = proposal.arguments["operation"]
        operation_version = proposal.arguments.get("version", "1.0")
        operation = self.operations.get(operation_name, operation_version)
        records = self._records_from_argument(proposal.arguments)
        output, cost = operation.run(
            arguments=proposal.arguments,
            records=records,
            store=self.store,
        )
        for record in output:
            self.registry.validate_provenance(record)
        self.budget.charge(cost)
        artifact = self.store.put_json(
            kind="dataset",
            value=[to_jsonable(record) for record in output],
            version=self.config.data_version,
            parents=[record.record_id for record in output],
        )
        self.artifacts.append(artifact)
        self._event(
            "operation_completed",
            operation=f"{operation_name}@{operation_version}",
            output_artifact=artifact,
            count=len(output),
            cost=cost,
        )

    def _request_feedback(self, proposal: ActionProposal) -> None:
        if self.feedback_provider is None:
            raise RuntimeError("This run has no feedback provider")
        target_artifact_id = proposal.arguments["target_artifact_id"]
        target = next(
            (item for item in self.artifacts if item.artifact_id == target_artifact_id),
            None,
        )
        if target is None:
            raise ValueError(f"Feedback target is not a run artifact: {target_artifact_id}")
        target_version = proposal.arguments.get("target_version", target.version)
        if target.version != target_version:
            raise ValueError(
                f"Feedback target version mismatch: artifact={target.version}, "
                f"requested={target_version}"
            )
        feedback = self.feedback_provider.request(
            target_artifact_id=target_artifact_id,
            target_version=target_version,
        )
        if (
            feedback.target_artifact_id != target_artifact_id
            or feedback.target_version != target_version
        ):
            raise ValueError("Feedback provider returned mismatched target provenance")
        assert self.budget is not None
        cost = Budget(feedback_requests=1)
        self.budget.charge(cost)
        self.feedback.append(feedback)
        self._event("feedback_received", feedback=feedback, cost=cost)

    def _train(self, proposal: ActionProposal) -> None:
        dataset_artifact_id = proposal.arguments["dataset_artifact_id"]
        value = self.store.get_json(dataset_artifact_id)
        records = [
            TaskRecord(
                record_id=item["record_id"],
                task_id=item["task_id"],
                split=item["split"],
                input=item["input"],
                target=item.get("target"),
                source_group_id=item.get("source_group_id"),
                metadata=item.get("metadata", {}),
                parent_record_ids=tuple(item.get("parent_record_ids", [])),
            )
            for item in value
        ]
        self._event("training_started", dataset_artifact_id=dataset_artifact_id)
        checkpoint, cost = self.trainer.train(
            train_records=records,
            initial_checkpoint=self.checkpoint,
            config=proposal.arguments.get("config", {}),
            seed=self.config.seed + self.batch_index,
            store=self.store,
        )
        self.budget.charge(cost)
        checkpoint_artifact = self.store.put_json(
            kind="checkpoint",
            value=checkpoint,
            version=f"{self.trainer.name}-{self.trainer.version}",
            parents=[dataset_artifact_id],
        )
        self.artifacts.append(checkpoint_artifact)
        self.checkpoint = checkpoint
        self._event(
            "training_completed",
            checkpoint_artifact=checkpoint_artifact,
            cost=cost,
        )

    def _evaluate(self, proposal: ActionProposal) -> None:
        if self.checkpoint is None:
            raise RuntimeError("Cannot evaluate before training")
        split = proposal.arguments.get("split", "development")
        include_final_test = split == "final_test" and bool(
            proposal.arguments.get("allow_final_test", False)
            and self.config.metadata.get("final_test_unlocked", False)
        )
        records = self.registry.records_for(
            self.config.task_id,
            split=split,
            include_final_test=include_final_test,
        )
        metrics, cost = self.evaluator.evaluate(
            checkpoint=self.checkpoint,
            records=records,
            config=proposal.arguments.get("config", {}),
        )
        self.budget.charge(cost)
        self.metrics.update(metrics)
        self.metrics_by_split[split] = dict(metrics)
        self._event(
            "evaluation_completed",
            split=split,
            metrics=metrics,
            cost=cost,
        )

    def _execute(self, proposal: ActionProposal) -> bool:
        self._validate_proposal(proposal)
        assert self.budget is not None
        # The policy estimate accounts for decision/evolution calls. Operation,
        # training, evaluation, and feedback implementations charge their own
        # measured costs after execution.
        if proposal.estimated_cost != Budget():
            self.budget.charge(proposal.estimated_cost)
            self._event("policy_cost_charged", cost=proposal.estimated_cost)
        self._event("action_proposed", proposal=proposal)
        if proposal.action_type in {
            "select_records",
            "transform_records",
            "filter_records",
            "generate_records",
        }:
            self._run_data_operation(proposal)
        elif proposal.action_type == "train":
            self._train(proposal)
        elif proposal.action_type == "evaluate":
            self._evaluate(proposal)
        elif proposal.action_type == "request_feedback":
            self._request_feedback(proposal)
        elif proposal.action_type == "finish":
            return False
        return True

    def run(self) -> None:
        if self.status == RunStatus.CREATED:
            self.start()
        if self.status not in {RunStatus.READY, RunStatus.PAUSED}:
            raise RuntimeError(f"Cannot run in status {self.status}")
        self.status = RunStatus.RUNNING
        self._event("run_started")
        try:
            while self.batch_index < self.config.max_batches:
                proposal = self.policy.propose(self._observation())
                should_continue = self._execute(proposal)
                self._snapshot()
                if not should_continue:
                    break
                self.batch_index += 1
            self.status = RunStatus.COMPLETED
            self._event("run_completed", metrics=self.metrics)
        except Exception as exc:
            self.status = RunStatus.FAILED
            self._event("run_failed", error=repr(exc))
            raise
        finally:
            self._snapshot()


class RunOrchestrator:
    def __init__(
        self,
        *,
        registry: TaskRegistry,
        store_root: str | Path,
        operations: OperationRegistry,
        trainer: TaskModelTrainer,
        evaluator: Evaluator,
        feedback_provider: FeedbackProvider | None = None,
    ):
        self.registry = registry
        self.store_root = Path(store_root)
        self.operations = operations
        self.trainer = trainer
        self.evaluator = evaluator
        self.feedback_provider = feedback_provider

    def create_run(
        self,
        *,
        config: RunConfig,
        policy: PolicyAdapter,
    ) -> CoreRun:
        store = ArtifactStore(self.store_root / config.run_id)
        task = self.registry.task(config.task_id)
        return CoreRun(
            config=config,
            task=task,
            registry=self.registry,
            store=store,
            operations=self.operations,
            trainer=self.trainer,
            evaluator=self.evaluator,
            policy=policy,
            feedback_provider=self.feedback_provider,
        )


class FixedPipelinePolicy(PolicyAdapter):
    """A transparent baseline used to verify the shared Action System loop."""

    def __init__(self, *, data_split: str = "seed"):
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
            )
        if self.stage == 2:
            self.stage += 1
            return ActionProposal(
                action_type="evaluate",
                arguments={"split": "development"},
                expected_versions={"harness": observation.config.harness_version},
            )
        return ActionProposal(action_type="finish")
