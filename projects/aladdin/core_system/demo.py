from __future__ import annotations

from pathlib import Path

from core_system.mle_core.components import (
    ExactMatchEvaluator,
    MajorityLabelTrainer,
    PassThroughTransform,
    SelectRecords,
)
from core_system.mle_core.interfaces import OperationRegistry
from core_system.mle_core.models import Budget, RunConfig, TaskRecord, TaskSpec
from core_system.mle_core.registry import TaskRegistry
from core_system.mle_core.runtime import FixedPipelinePolicy, RunOrchestrator


def build_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register_task(
        TaskSpec(
            task_id="toy-intent",
            name="Toy intent classification",
            input_schema="text",
            target_schema="label",
        )
    )
    registry.add_records(
        [
            TaskRecord(
                record_id="seed-001",
                task_id="toy-intent",
                split="seed",
                input="reset my password",
                target="account",
                source_group_id="demo",
            ),
            TaskRecord(
                record_id="seed-002",
                task_id="toy-intent",
                split="seed",
                input="where is my invoice",
                target="billing",
                source_group_id="demo",
            ),
            TaskRecord(
                record_id="dev-001",
                task_id="toy-intent",
                split="development",
                input="my account is locked",
                target="account",
                source_group_id="demo-dev",
            ),
            TaskRecord(
                record_id="dev-002",
                task_id="toy-intent",
                split="development",
                input="please send the receipt",
                target="billing",
                source_group_id="demo-dev",
            ),
        ]
    )
    return registry


def main() -> None:
    registry = build_registry()
    operations = OperationRegistry()
    operations.register(SelectRecords())
    operations.register(PassThroughTransform())

    config = RunConfig(
        run_id="core-demo",
        task_id="toy-intent",
        backbone_llm="placeholder-backbone-v0",
        task_model="majority-label-toy-v0",
        evaluator="exact-match-v0",
        harness_version="core-fixed-harness-v0",
        data_version="demo-data-v0",
        initial_checkpoint=None,
        max_batches=6,
        budget_limit=Budget(
            training_steps=5,
            evaluator_calls=5,
            wall_clock_seconds=60,
        ),
        seed=7,
    )

    orchestrator = RunOrchestrator(
        registry=registry,
        store_root=Path(__file__).parent / "demo_runs",
        operations=operations,
        trainer=MajorityLabelTrainer(),
        evaluator=ExactMatchEvaluator(),
    )
    run = orchestrator.create_run(config=config, policy=FixedPipelinePolicy())
    run.run()
    print(f"run={config.run_id}")
    print(f"status={run.status.value}")
    print(f"metrics={run.metrics}")
    print(f"artifacts={[artifact.artifact_id for artifact in run.artifacts]}")


if __name__ == "__main__":
    main()
