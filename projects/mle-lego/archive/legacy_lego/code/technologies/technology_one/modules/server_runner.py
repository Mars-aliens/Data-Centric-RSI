from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core_system.mle_core.components import (
    FilterByMetadata,
    GenerateTemplateRecords,
    PassThroughTransform,
    SelectRecords,
)
from core_system.mle_core.interfaces import OperationRegistry
from core_system.mle_core.models import Budget, RunConfig, TaskSpec
from core_system.mle_core.registry import TaskRegistry
from core_system.mle_core.runtime import RunOrchestrator

from .data_validation import validate_records
from .protocol import JsonlTaskAdapter, load_protocol
from .server_adapters import CommandEvaluator, CommandPolicyAdapter, CommandTaskModelTrainer, JsonCommand
from .server_preflight import run_preflight


def _required_string(profile: dict[str, Any], path: str) -> str:
    value: Any = profile
    for key in path.split("."):
        value = value[key]
    if not isinstance(value, str) or value.startswith("SET_"):
        raise ValueError(f"Profile field {path} must be set to a real server value")
    return value


def _budget(profile: dict[str, Any]) -> Budget:
    raw = profile["budget"]
    fields = (
        "llm_calls",
        "llm_tokens",
        "training_steps",
        "training_seconds",
        "evaluator_calls",
        "feedback_requests",
        "wall_clock_seconds",
    )
    values: dict[str, int | float] = {}
    for field in fields:
        value = raw[field]
        if isinstance(value, bool) or isinstance(value, str):
            raise ValueError(f"Profile budget field {field} must be numeric")
        values[field] = value
    return Budget(**values)


def _command(profile: dict[str, Any], key: str) -> JsonCommand:
    commands = profile.get("commands", {})
    value = commands.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Profile commands.{key} must be an argv array")
    action_system = profile.get("action_system", profile.get("execution", {}))
    return JsonCommand.from_value(value, cwd=action_system.get("command_cwd"))


def run_server_method(
    profile_path: Path,
    *,
    root: Path,
    method: str,
    run_id: str,
    output_dir: Path,
    seed: int | None = None,
    report_final_test: bool = False,
) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preflight = run_preflight(profile_path, root)
    if preflight["errors"]:
        raise ValueError("Server preflight failed: " + "; ".join(preflight["errors"]))
    if method not in profile["methods"]:
        raise ValueError(f"Method {method!r} is not registered in the server profile")

    dataset = profile["dataset"]
    protocol = load_protocol(root / dataset["protocol"])
    records = JsonlTaskAdapter(protocol).load(root / dataset["train_jsonl"])
    final_records = JsonlTaskAdapter(protocol).load(root / dataset["final_test_jsonl"])
    if report_final_test:
        validate_records(records + final_records, protocol)
        records = records + final_records
    else:
        final_records = []

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
    for operation in (SelectRecords(), PassThroughTransform(), FilterByMetadata(), GenerateTemplateRecords()):
        operations.register(operation)

    decision_model = _required_string(profile, "decision_model.model_id")
    decision_revision = _required_string(profile, "decision_model.revision")
    decision_model_settings = profile.get("decision_model", {})
    task_model = _required_string(profile, "task_model.model_id")
    evaluator_name = _required_string(profile, "evaluator.name")
    run_seed = int(profile["seeds"][0] if seed is None else seed)
    config = RunConfig(
        run_id=run_id,
        task_id=protocol.task_id,
        backbone_llm=decision_model,
        task_model=task_model,
        evaluator=evaluator_name,
        harness_version=f"lego-server-{method}-v1",
        data_version="helpsteer2-normalized-v0",
        initial_checkpoint=_required_string(profile, "task_model.initial_checkpoint"),
        # This runner advances once per policy action. A complete baseline
        # cycle needs select, train, held-in, held-out, and finish actions.
        max_batches=max(5, int(profile["stream"].get("max_batches", 5))),
        budget_limit=_budget(profile),
        seed=run_seed,
        metadata={
            "experiment_family": "lego-server-method",
            "method": method,
            "profile_version": profile["profile_version"],
            "dataset_source": dataset["source"],
            "dataset_revision": dataset["source_revision"],
            "dataset_summary": preflight["dataset_summary"],
            "final_test_summary": preflight["final_test_summary"] if report_final_test else None,
            "final_test_unlocked": report_final_test,
            "server_profile": str(profile_path),
        },
    )
    run = RunOrchestrator(
        registry=registry,
        store_root=output_dir,
        operations=operations,
        trainer=CommandTaskModelTrainer(
            _command(profile, "trainer"),
            name=task_model,
            version=_required_string(profile, "task_model.revision"),
            initial_checkpoint_path=_required_string(profile, "task_model.initial_checkpoint"),
            default_config=profile.get("task_model", {}).get("training_config", {}),
        ),
        evaluator=CommandEvaluator(_command(profile, "evaluator"), name=evaluator_name, version=_required_string(profile, "evaluator.revision")),
    ).create_run(
        config=config,
        policy=CommandPolicyAdapter(
            _command(profile, "policy"),
            method=method,
            model_id=decision_model,
            model_revision=decision_revision,
            model_settings=decision_model_settings,
            reflection_path=(
                output_dir / run_id / "reflections.jsonl"
                if method == "react-reflexion"
                else None
            ),
        ),
    )
    run.run()
    result = {
        "status": run.status.value,
        "run_id": run_id,
        "seed": run_seed,
        "method": method,
        "metrics_by_split": run.metrics_by_split,
        "final_test_used": report_final_test,
        "decision_llm_used": True,
        "held_out_used_for_acceptance": False,
        "output_dir": str(output_dir / run_id),
    }
    run_root = output_dir / run_id
    (run_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-model LEGO method through the server JSON contract.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--report-final-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_server_method(args.profile, root=args.root, method=args.method, run_id=args.run_id, output_dir=args.output_dir, seed=args.seed, report_final_test=args.report_final_test), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
