from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_system.mle_core.components import SelectRecords
from core_system.mle_core.interfaces import OperationRegistry, PolicyAdapter
from core_system.mle_core.models import ActionProposal, Budget, RunConfig, TaskSpec
from core_system.mle_core.registry import TaskRegistry
from core_system.mle_core.runtime import RunOrchestrator
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)

from .data_validation import validate_records
from .helpsteer2 import DATASET_REVISION
from .protocol import JsonlTaskAdapter, load_protocol


class FixedRealDataPolicy(PolicyAdapter):
    def __init__(self, *, report_final_test: bool = False):
        self.stage = 0
        self.report_final_test = report_final_test

    def propose(self, observation) -> ActionProposal:
        if self.stage == 0:
            self.stage += 1
            return ActionProposal(
                "select_records",
                {"operation": "select_records", "version": "1.0", "split": "seed"},
                {"harness": observation.config.harness_version, "data": observation.config.data_version},
                rationale="Train the fixed baseline on the declared seed split.",
            )
        if self.stage == 1:
            self.stage += 1
            dataset = next(item for item in reversed(observation.artifacts) if item.kind == "dataset")
            return ActionProposal(
                "train",
                {"dataset_artifact_id": dataset.artifact_id},
                {"harness": observation.config.harness_version},
            )
        splits = ["held_in", "held_out"]
        if self.report_final_test:
            splits.append("final_test")
        eval_index = self.stage - 2
        if eval_index < len(splits):
            split = splits[eval_index]
            self.stage += 1
            arguments = {"split": split}
            if split == "final_test":
                arguments["allow_final_test"] = True
            return ActionProposal(
                "evaluate",
                arguments,
                {"harness": observation.config.harness_version},
                rationale=f"Report fixed baseline performance on {split}.",
            )
        return ActionProposal("finish")


def run_fixed_real_baseline(
    dataset_path: Path,
    output_dir: Path,
    run_id: str,
    *,
    final_test_path: Path | None = None,
    report_final_test: bool = False,
) -> dict[str, object]:
    if report_final_test and final_test_path is None:
        raise ValueError("--report-final-test requires an explicit --final-test dataset")
    protocol = load_protocol(Path(__file__).with_name("data_protocol.json"))
    records = JsonlTaskAdapter(protocol).load(dataset_path)
    summary = validate_records(records, protocol)
    if final_test_path is not None:
        final_records = JsonlTaskAdapter(protocol).load(final_test_path)
        final_summary = validate_records(final_records, protocol)
        validate_records(records + final_records, protocol)
        records.extend(final_records)
    else:
        final_summary = None
    registry = TaskRegistry()
    registry.register_task(
        TaskSpec(
            task_id=protocol.task_id,
            name="HelpSteer2 pairwise preference classification",
            input_schema="prompt and response text",
            target_schema="preferred/rejected/tie from prompt-group mean human rating",
            allowed_splits=protocol.allowed_splits,
        )
    )
    registry.add_records(records)
    operations = OperationRegistry()
    operations.register(SelectRecords())
    config = RunConfig(
        run_id=run_id,
        task_id=protocol.task_id,
        backbone_llm="not-used-fixed-policy-v0",
        task_model="tfidf-logistic-v0",
        evaluator="tfidf-logistic-exact-match-v0",
        harness_version="fixed-baseline-pipeline",
        data_version="helpsteer2-normalized-v0",
        initial_checkpoint=None,
        max_batches=6,
        budget_limit=Budget(training_steps=2, evaluator_calls=3, wall_clock_seconds=600),
        seed=17,
        metadata={
            "experiment_family": "lego-real-data-baseline",
            "dataset": "nvidia/HelpSteer2",
            "dataset_revision": DATASET_REVISION,
            "dataset_license": "CC-BY-4.0",
            "dataset_summary": summary.to_dict(),
            "final_test_summary": final_summary.to_dict() if final_summary else None,
            "label_policy": "highest/lowest/middle mean of five rating dimensions within prompt group",
            "final_test_unlocked": report_final_test,
        },
    )
    run = RunOrchestrator(
        registry=registry,
        store_root=output_dir,
        operations=operations,
        trainer=TfidfLogisticTrainer(),
        evaluator=TfidfLogisticEvaluator(),
    ).create_run(
        config=config,
        policy=FixedRealDataPolicy(report_final_test=report_final_test),
    )
    run.run()
    return {
        "status": run.status.value,
        "run_id": run_id,
        "dataset_summary": summary.to_dict(),
        "metrics_by_split": run.metrics_by_split,
        "output_dir": str(output_dir / run_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HelpSteer2 fixed LEGO baseline.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--final-test", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs" / "real-data")
    parser.add_argument("--run-id", default="helpsteer2-fixed-baseline")
    parser.add_argument("--report-final-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_fixed_real_baseline(
                args.dataset,
                args.output_dir,
                args.run_id,
                final_test_path=args.final_test,
                report_final_test=args.report_final_test,
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

