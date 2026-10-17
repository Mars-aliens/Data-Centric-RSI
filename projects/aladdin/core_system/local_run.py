from __future__ import annotations

import argparse
from pathlib import Path

from .mle_core.components import PassThroughTransform, SelectRecords
from .mle_core.interfaces import OperationRegistry
from .mle_core.local_data import build_local_text_registry
from .mle_core.models import Budget, RunConfig
from .mle_core.runtime import FixedPipelinePolicy, RunOrchestrator
from .mle_core.text_components import TextExactMatchEvaluator, TokenPrototypeTrainer


def build_components() -> OperationRegistry:
    operations = OperationRegistry()
    operations.register(SelectRecords())
    operations.register(PassThroughTransform())
    return operations


def run_fixed_baseline(output_dir: Path, run_id: str) -> dict[str, object]:
    registry = build_local_text_registry()
    config = RunConfig(
        run_id=run_id,
        task_id="local-intent",
        backbone_llm="placeholder-backbone-v0",
        task_model="token-prototype-v0",
        evaluator="text-exact-match-v0",
        harness_version="fixed-harness-v0",
        data_version="local-seed-v0",
        initial_checkpoint=None,
        max_batches=6,
        budget_limit=Budget(
            training_steps=3,
            evaluator_calls=3,
            wall_clock_seconds=120,
        ),
        seed=17,
        metadata={
            "experiment_family": "shared-core-baseline",
            "policy": "fixed-pipeline",
        },
    )
    orchestrator = RunOrchestrator(
        registry=registry,
        store_root=output_dir,
        operations=build_components(),
        trainer=TokenPrototypeTrainer(),
        evaluator=TextExactMatchEvaluator(),
    )
    run = orchestrator.create_run(config=config, policy=FixedPipelinePolicy())
    run.run()
    return {
        "run_id": config.run_id,
        "status": run.status.value,
        "metrics": run.metrics,
        "artifact_ids": [artifact.artifact_id for artifact in run.artifacts],
        "output_dir": str(output_dir / run_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local shared-core baseline.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "local_runs"),
    )
    parser.add_argument("--run-id", default="local-core-baseline")
    args = parser.parse_args()
    result = run_fixed_baseline(Path(args.output_dir), args.run_id)
    print(result)


if __name__ == "__main__":
    main()
