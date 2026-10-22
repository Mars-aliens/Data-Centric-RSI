from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_system.mle_core.components import PassThroughTransform, SelectRecords
from core_system.mle_core.interfaces import OperationRegistry
from core_system.mle_core.local_data import build_local_text_registry
from core_system.mle_core.models import Budget, RunConfig
from core_system.mle_core.runtime import RunOrchestrator
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)

from .policies import FixedHarnessReactPolicy


def load_config() -> dict[str, object]:
    return json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def build_operations() -> OperationRegistry:
    operations = OperationRegistry()
    operations.register(SelectRecords())
    operations.register(PassThroughTransform())
    return operations


def run_fixed_baseline(output_dir: Path, run_id: str) -> dict[str, object]:
    settings = load_config()
    budget = settings["budget"]
    assert isinstance(budget, dict)
    config = RunConfig(
        run_id=run_id,
        task_id=str(settings["task_id"]),
        backbone_llm=str(settings["backbone_llm"]),
        task_model=str(settings["task_model"]),
        evaluator=str(settings["evaluator"]),
        harness_version=str(settings["harness_version"]),
        data_version=str(settings["data_version"]),
        initial_checkpoint=None,
        max_batches=int(settings["max_batches"]),
        budget_limit=Budget(
            training_steps=int(budget["training_steps"]),
            evaluator_calls=int(budget["evaluator_calls"]),
            wall_clock_seconds=float(budget["wall_clock_seconds"]),
        ),
        seed=int(settings["seed"]),
        metadata={
            "experiment_family": str(settings["experiment"]),
            "baseline": str(settings["baseline"]),
            "implementation_status": str(settings["implementation_status"]),
            "harness_mutation": False,
            "memory_update": False,
            "rollback_enabled": False,
        },
    )
    orchestrator = RunOrchestrator(
        registry=build_local_text_registry(),
        store_root=output_dir,
        operations=build_operations(),
        trainer=TfidfLogisticTrainer(),
        evaluator=TfidfLogisticEvaluator(),
    )
    run = orchestrator.create_run(
        config=config,
        policy=FixedHarnessReactPolicy(),
    )
    run.run()
    return {
        "experiment": "lego",
        "baseline": str(settings["baseline"]),
        "run_id": config.run_id,
        "status": run.status.value,
        "metrics": run.metrics,
        "output_dir": str(output_dir / run_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent LEGO baseline.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "runs"),
    )
    parser.add_argument("--run-id", default="lego-fixed-baseline")
    args = parser.parse_args()
    print(run_fixed_baseline(Path(args.output_dir), args.run_id))


if __name__ == "__main__":
    main()
