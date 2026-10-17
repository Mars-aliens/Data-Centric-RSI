from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_system.mle_core.components import PassThroughTransform, SelectRecords
from core_system.mle_core.components import ApplyFeedbackLabels, GenerateTemplateRecords
from core_system.mle_core.feedback import InMemoryFeedbackProvider
from core_system.mle_core.interfaces import OperationRegistry
from core_system.mle_core.local_data import build_local_text_registry
from core_system.mle_core.models import Budget, RunConfig, to_jsonable
from core_system.mle_core.runtime import RunOrchestrator
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)

from .policies import (
    AblationPolicy,
    AdaptiveAladdinPolicy,
    FixedDataPipelinePolicy,
    RandomFeedbackPolicy,
    UniformBudgetPolicy,
)


def load_config() -> dict[str, object]:
    return json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def build_operations() -> OperationRegistry:
    operations = OperationRegistry()
    operations.register(SelectRecords())
    operations.register(PassThroughTransform())
    operations.register(GenerateTemplateRecords())
    operations.register(ApplyFeedbackLabels())
    return operations


def run_strategy(
    output_dir: Path,
    run_id: str,
    *,
    strategy: str = "fixed-data-pipeline",
    seed: int | None = None,
    feedback_sequence: list[object] | None = None,
    report_final_test: bool = False,
) -> dict[str, object]:
    run_root = output_dir / run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_root}. "
            "Use a new run_id/output directory to preserve the append-only audit log."
        )
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
            feedback_requests=int(budget.get("feedback_requests", 0)),
            wall_clock_seconds=float(budget["wall_clock_seconds"]),
        ),
        seed=int(settings["seed"] if seed is None else seed),
        metadata={
            "experiment_family": str(settings["experiment"]),
            "baseline": strategy,
            "adaptive_feedback": strategy == "adaptive-aladdin",
            "adaptive_synthesis_budget": strategy in {
                "uniform-budget", "adaptive-aladdin",
                "ablation-no-feedback-influence",
                "ablation-no-human-calibration",
                "ablation-no-training-gain-allocation",
            },
            "ablation": {
                "ablation-no-feedback-influence": "feedback influence estimation",
                "ablation-no-human-calibration": "human calibration",
                "ablation-no-training-gain-allocation": "training-gain budget allocation",
            }.get(strategy),
            "final_test_unlocked": report_final_test,
        },
    )
    policies = {
        "seed-only": lambda: FixedDataPipelinePolicy(report_final_test=report_final_test),
        "seed-sft": lambda: FixedDataPipelinePolicy(report_final_test=report_final_test),
        "fixed-data-pipeline": lambda: FixedDataPipelinePolicy(report_final_test=report_final_test),
        "random-feedback": lambda: RandomFeedbackPolicy(seed=config.seed, report_final_test=report_final_test),
        "uniform-budget": lambda: UniformBudgetPolicy(seed=config.seed, report_final_test=report_final_test),
        "adaptive-aladdin": lambda: AdaptiveAladdinPolicy(seed=config.seed, report_final_test=report_final_test),
        "ablation-no-feedback-influence": lambda: AblationPolicy(mode="no-feedback-influence", seed=config.seed, report_final_test=report_final_test),
        "ablation-no-human-calibration": lambda: AblationPolicy(mode="no-human-calibration", seed=config.seed, report_final_test=report_final_test),
        "ablation-no-training-gain-allocation": lambda: AblationPolicy(mode="no-training-gain-allocation", seed=config.seed, report_final_test=report_final_test),
    }
    try:
        policy = policies[strategy]()
    except KeyError as exc:
        raise ValueError(f"Unknown Aladdin strategy: {strategy}") from exc
    orchestrator = RunOrchestrator(
        registry=build_local_text_registry(),
        store_root=output_dir,
        operations=build_operations(),
        trainer=TfidfLogisticTrainer(),
        evaluator=TfidfLogisticEvaluator(),
        feedback_provider=(
            InMemoryFeedbackProvider(sequence=feedback_sequence or [{
                "decision": "accept",
                "accepted_record_ids": [
                    "candidate-account",
                    "candidate-billing",
                    "candidate-access",
                    "candidate-invoice",
                ],
                "labels": {
                    "candidate-account": "account",
                    "candidate-billing": "billing",
                    "candidate-access": "account",
                    "candidate-invoice": "billing",
                },
                "influence_scores": {
                    "candidate-account": 0.9,
                    "candidate-billing": 0.8,
                    "candidate-access": 0.4,
                    "candidate-invoice": 0.7,
                },
                "calibration": {
                    "candidate-account": True,
                    "candidate-billing": True,
                    "candidate-access": True,
                    "candidate-invoice": False,
                },
                "training_gains": {
                    "candidate-account": 0.4,
                    "candidate-billing": 0.0,
                    "candidate-access": 0.2,
                    "candidate-invoice": 0.2,
                },
            }])
            if strategy not in {"seed-only", "seed-sft", "fixed-data-pipeline"}
            else None
        ),
    )
    run = orchestrator.create_run(
        config=config,
        policy=policy,
    )
    run.run()
    return {
        "experiment": "aladdin",
        "baseline": strategy,
        "run_id": config.run_id,
        "status": run.status.value,
        "metrics": run.metrics,
        "metrics_by_split": run.metrics_by_split,
        "used_budget": run.budget.used if run.budget else Budget(),
        "output_dir": str(output_dir / run_id),
    }


def run_fixed_baseline(output_dir: Path, run_id: str) -> dict[str, object]:
    return run_strategy(output_dir, run_id, strategy="fixed-data-pipeline")


def run_experiment_matrix(output_dir: Path, *, seeds: tuple[int, ...] = (17, 18)) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for seed in seeds:
        for strategy in (
            "seed-only",
            "seed-sft",
            "fixed-data-pipeline",
            "random-feedback",
            "uniform-budget",
            "adaptive-aladdin",
            "ablation-no-feedback-influence",
            "ablation-no-human-calibration",
            "ablation-no-training-gain-allocation",
        ):
            results.append(
                run_strategy(
                    output_dir,
                    f"{strategy}-{seed}",
                    strategy=strategy,
                    seed=seed,
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix-summary.json").write_text(
        json.dumps(to_jsonable(results), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return results


def run_final_test_matrix(output_dir: Path, *, seeds: tuple[int, ...] = (17, 18)) -> list[dict[str, object]]:
    """Read final_test only after the development matrix and settings are frozen."""
    strategies = (
        "seed-only",
        "seed-sft",
        "fixed-data-pipeline",
        "random-feedback",
        "uniform-budget",
        "adaptive-aladdin",
        "ablation-no-feedback-influence",
        "ablation-no-human-calibration",
        "ablation-no-training-gain-allocation",
    )
    results = [
        run_strategy(
            output_dir,
            f"final-{strategy}-{seed}",
            strategy=strategy,
            seed=seed,
            report_final_test=True,
        )
        for seed in seeds
        for strategy in strategies
    ]
    (output_dir / "final-test-summary.json").write_text(
        json.dumps(to_jsonable(results), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return results


def build_local_freeze_report(development_dir: Path, final_dir: Path) -> dict[str, object]:
    development = json.loads((development_dir / "matrix-summary.json").read_text(encoding="utf-8"))
    final = json.loads((final_dir / "final-test-summary.json").read_text(encoding="utf-8"))
    audit = json.loads((development_dir / "audit-report.json").read_text(encoding="utf-8"))
    leakage = []
    for result in development:
        run_dir = Path(result["output_dir"])
        snapshot = json.loads((run_dir / "runs" / f"{result['run_id']}.json").read_text(encoding="utf-8"))
        if "final_test" in snapshot.get("metrics_by_split", {}):
            leakage.append(result["run_id"])
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event_type") == "evaluation_completed" and event.get("payload", {}).get("split") == "final_test":
                leakage.append(result["run_id"])
    final_results = []
    for result in final:
        run_dir = Path(result["output_dir"])
        snapshot = json.loads((run_dir / "runs" / f"{result['run_id']}.json").read_text(encoding="utf-8"))
        final_results.append({
            "run_id": result["run_id"],
            "strategy": result["baseline"],
            "seed": snapshot["config"]["seed"],
            "development_accuracy": snapshot["metrics_by_split"].get("development", {}).get("accuracy"),
            "held_out_accuracy": snapshot["metrics_by_split"].get("held_out", {}).get("accuracy"),
            "final_test_accuracy": snapshot["metrics_by_split"].get("final_test", {}).get("accuracy"),
            "final_test_records": snapshot["metrics_by_split"].get("final_test", {}).get("num_records"),
            "final_test_unlocked": snapshot["config"]["metadata"].get("final_test_unlocked", False),
            "used_budget": snapshot["used_budget"],
        })
    grouped_final: dict[str, list[float]] = {}
    for item in final_results:
        score = item["final_test_accuracy"]
        if score is not None:
            grouped_final.setdefault(item["strategy"], []).append(float(score))
    final_test_summary = {
        strategy: {
            "mean_accuracy": sum(scores) / len(scores),
            "seed_scores": scores,
        }
        for strategy, scores in sorted(grouped_final.items())
    }
    final_is_valid = bool(final_results) and all(
        item["final_test_unlocked"]
        and item["final_test_accuracy"] is not None
        and item["final_test_records"] == 10.0
        for item in final_results
    )
    report = {
        "status": "complete" if not leakage and final_is_valid else "error",
        "experiment": "aladdin-local-toy-freeze",
        "dataset_version": "aladdin-seed-v1",
        "seeds": sorted({item["seed"] for item in final_results}),
        "strategy_count": len({item["strategy"] for item in final_results}),
        "development_final_test_leakage": sorted(set(leakage)),
        "development_runs": len(development),
        "final_test_runs": final_results,
        "final_test_summary": final_test_summary,
        "development_audit_findings": audit["findings"],
        "limitations": [
            "The task and candidate feedback are deterministic toy data, not evidence of real-task effectiveness.",
            "seed-only and seed-sft are operationally identical TF-IDF seed-training proxies; they are not LLM SFT or DPO.",
            "The three ablations exercise synthetic replay diagnostics and do not yet implement statistically estimated influence, calibrated LLM evaluation, or measured multi-arm training gains.",
            "Only two seeds are available; all reported metrics are descriptive and must not be treated as paper conclusions.",
            "Feedback, data-generation, and training costs are local proxies; external paper baselines remain for server reproduction.",
        ],
        "server_stage_required": [
            "Reproduce DITTO and DataEnvGym with their task-appropriate LLMs and data.",
            "Reproduce DataFlow/DataEvolver under matched preparation and training budgets.",
            "Implement real feedback influence estimation, human-evaluator calibration, and training-gain budget allocation.",
            "Increase seeds and use a real task dataset before drawing effectiveness conclusions.",
        ],
    }
    return report


def audit_matrix(output_dir: Path) -> dict[str, object]:
    """Audit completed runs for trace integrity and toy-task limitations."""
    runs: list[dict[str, object]] = []
    findings: list[dict[str, str]] = []
    for run_dir in sorted(item for item in output_dir.iterdir() if item.is_dir()):
        snapshot_path = run_dir / "runs" / f"{run_dir.name}.json"
        events_path = run_dir / "events.jsonl"
        if not snapshot_path.exists() or not events_path.exists():
            findings.append({"severity": "error", "run": run_dir.name, "message": "Missing snapshot or event log."})
            continue
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        created = sum(event["event_type"] == "run_created" for event in events)
        completed = sum(event["event_type"] == "run_completed" for event in events)
        if created != 1 or completed != 1:
            findings.append({"severity": "error", "run": run_dir.name, "message": f"Expected one run_created/run_completed, found {created}/{completed}."})
        runs.append({
            "run_id": run_dir.name,
            "baseline": snapshot["config"]["metadata"]["baseline"],
            "status": snapshot["status"],
            "events": len(events),
            "feedback_requests": snapshot["used_budget"]["feedback_requests"],
            "training_steps": snapshot["used_budget"]["training_steps"],
            "metrics_by_split": snapshot.get("metrics_by_split", {}),
        })
    accuracies = {
        run["metrics_by_split"].get("held_out", {}).get("accuracy")
        for run in runs
        if run["metrics_by_split"].get("held_out")
    }
    if len(accuracies) <= 1 and runs:
        findings.append({"severity": "warning", "run": "matrix", "message": "Held-out accuracy has no variation; the toy task is not discriminative enough for method conclusions."})
    record_counts = {
        metrics.get("num_records")
        for run in runs
        for metrics in run["metrics_by_split"].values()
        if metrics.get("num_records") is not None
    }
    if record_counts and max(record_counts) <= 2:
        findings.append({"severity": "warning", "run": "matrix", "message": "Each evaluation split has at most two records; treat accuracy as a pipeline check, not a statistical result."})
    fixed = {
        run["run_id"][-2:]: run["metrics_by_split"].get("held_out", {}).get("accuracy")
        for run in runs
        if run["baseline"] == "fixed-data-pipeline"
    }
    adaptive = {
        run["run_id"][-2:]: run["metrics_by_split"].get("held_out", {}).get("accuracy")
        for run in runs
        if run["baseline"] == "adaptive-aladdin"
    }
    if fixed and adaptive and fixed == adaptive:
        findings.append({"severity": "warning", "run": "adaptive-aladdin", "message": "Adaptive Aladdin matches fixed baseline on held-out accuracy for every seed; no local improvement is demonstrated."})
    uniform = {
        run["run_id"][-2:]: run["metrics_by_split"].get("held_out", {}).get("accuracy")
        for run in runs
        if run["baseline"] == "uniform-budget"
    }
    if fixed and uniform and any(
        uniform.get(seed) is not None
        and fixed.get(seed) is not None
        and uniform[seed] < fixed[seed]
        for seed in uniform
    ):
        findings.append({"severity": "warning", "run": "uniform-budget", "message": "Uniform budget is below fixed baseline on held-out accuracy; inspect candidate-class balance because this policy selected one feedback-labeled candidate."})
    by_baseline: dict[str, list[float]] = {}
    for run in runs:
        held_out = run["metrics_by_split"].get("held_out", {}).get("accuracy")
        if held_out is not None:
            by_baseline.setdefault(run["baseline"], []).append(float(held_out))
        development = run["metrics_by_split"].get("development", {}).get("accuracy")
        if held_out is not None and development is not None and abs(float(development) - float(held_out)) > 0.2:
            findings.append({"severity": "warning", "run": run["run_id"], "message": f"Development/held-out accuracy gap is {abs(float(development) - float(held_out)):.2f}; inspect distribution shift or overfitting."})
    for baseline, values in by_baseline.items():
        if len(values) >= 2 and max(values) - min(values) > 0.1:
            findings.append({"severity": "warning", "run": baseline, "message": f"Held-out accuracy varies by {max(values) - min(values):.2f} across seeds; increase seeds before drawing conclusions."})
    return {"runs": runs, "findings": findings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent Aladdin baseline.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "runs"),
    )
    parser.add_argument("--run-id", default="aladdin-fixed-baseline")
    parser.add_argument("--final-dir", default=None)
    parser.add_argument(
        "--strategy",
        choices=(
            "seed-only", "seed-sft", "fixed-data-pipeline", "random-feedback",
            "uniform-budget", "adaptive-aladdin", "ablation-no-feedback-influence",
            "ablation-no-human-calibration", "ablation-no-training-gain-allocation",
            "matrix", "final-test", "audit", "freeze-report",
        ),
        default="fixed-data-pipeline",
    )
    args = parser.parse_args()
    if args.strategy == "matrix":
        print(run_experiment_matrix(Path(args.output_dir)))
    elif args.strategy == "final-test":
        print(run_final_test_matrix(Path(args.output_dir)))
    elif args.strategy == "audit":
        report = audit_matrix(Path(args.output_dir))
        (Path(args.output_dir) / "audit-report.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))
    elif args.strategy == "freeze-report":
        if args.final_dir is None:
            parser.error("--final-dir is required for freeze-report")
        report = build_local_freeze_report(Path(args.output_dir), Path(args.final_dir))
        report_path = Path(args.final_dir) / "local-freeze-report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(run_strategy(Path(args.output_dir), args.run_id, strategy=args.strategy))


if __name__ == "__main__":
    main()
