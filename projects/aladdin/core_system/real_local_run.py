from __future__ import annotations

import argparse
from pathlib import Path

from .mle_core.components import PassThroughTransform, SelectRecords
from .mle_core.interfaces import OperationRegistry
from .mle_core.local_data import build_local_text_registry
from .mle_core.models import Budget, RunConfig
from .mle_core.runtime import FixedPipelinePolicy, RunOrchestrator
from .mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local scikit-learn Task Model baseline."
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "real_local_runs"),
    )
    parser.add_argument("--run-id", default="local-tfidf-baseline")
    args = parser.parse_args()

    operations = OperationRegistry()
    operations.register(SelectRecords())
    operations.register(PassThroughTransform())
    config = RunConfig(
        run_id=args.run_id,
        task_id="local-intent",
        backbone_llm="placeholder-backbone-v0",
        task_model="tfidf-logistic-v0",
        evaluator="tfidf-logistic-exact-match-v0",
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
        metadata={"experiment_family": "shared-core-real-local-baseline"},
    )
    orchestrator = RunOrchestrator(
        registry=build_local_text_registry(),
        store_root=Path(args.output_dir),
        operations=operations,
        trainer=TfidfLogisticTrainer(),
        evaluator=TfidfLogisticEvaluator(),
    )
    run = orchestrator.create_run(config=config, policy=FixedPipelinePolicy())
    run.run()
    print(
        {
            "run_id": config.run_id,
            "status": run.status.value,
            "metrics": run.metrics,
            "output_dir": str(Path(args.output_dir) / config.run_id),
        }
    )


if __name__ == "__main__":
    main()
