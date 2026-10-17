from __future__ import annotations

"""Run the technology-one candidate matrix with the server trainer/evaluator.

Candidate construction is deterministic and auditable in this first server
stage. The configured task-model trainer and evaluator are real server
commands, while the decision LLM is reserved for the later policy-driven
Harness evolution runs.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from core_system.mle_core.models import to_jsonable
from core_system.mle_core.store import ArtifactStore

from .data_validation import validate_records
from .harness import (
    HarnessVersionChain,
    apply_edits,
    dependency_closure,
    random_joint_edit,
)
from .protocol import JsonlTaskAdapter, load_protocol
from .progress import ProgressReporter
from .server_adapters import CommandEvaluator, CommandTaskModelTrainer
from .server_runner import _command, _required_string
from .server_preflight import run_preflight
from .technical_one_local import (
    METHODS,
    SPLITS,
    _eligible_groups,
    _groups,
    _limit_groups,
    _make_candidates,
    _select_groups,
    _train_records,
    initial_manifest,
)


def _group_hash(group_ids: list[str]) -> str:
    payload = "\n".join(sorted(group_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_candidate(
    *,
    root: Path,
    label: str,
    manifest: Any,
    train_records: list[Any],
    train_group_ids: list[str],
    held_in_records: list[Any],
    held_out_records: list[Any],
    profile: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    candidate_root = root / label
    store = ArtifactStore(candidate_root)
    task_model = _required_string(profile, "task_model.model_id")
    task_revision = _required_string(profile, "task_model.revision")
    evaluator_name = _required_string(profile, "evaluator.name")
    evaluator_revision = _required_string(profile, "evaluator.revision")
    trainer = CommandTaskModelTrainer(
        _command(profile, "trainer"),
        name=task_model,
        version=task_revision,
        initial_checkpoint_path=_required_string(
            profile, "task_model.initial_checkpoint"
        ),
        default_config=profile.get("task_model", {}).get("training_config", {}),
    )
    evaluator = CommandEvaluator(
        _command(profile, "evaluator"),
        name=evaluator_name,
        version=evaluator_revision,
    )

    harness_artifact = store.put_json(
        kind="harness_manifest",
        value=manifest.to_dict(),
        version=manifest.version,
    )
    dataset_artifact = store.put_json(
        kind="dataset",
        value=[
            {
                "record_id": record.record_id,
                "task_id": record.task_id,
                "split": record.split,
                "input": record.input,
                "target": record.target,
                "source_group_id": record.source_group_id,
                "metadata": record.metadata,
                "parent_record_ids": list(record.parent_record_ids),
            }
            for record in train_records
        ],
        version="helpsteer2-normalized-v0",
        parents=[harness_artifact.artifact_id],
    )
    store.append_event(
        "candidate_started",
        {
            "method": label,
            "harness": manifest,
            "dataset_artifact": dataset_artifact,
            "train_group_count": len(train_group_ids),
            "train_group_hash": _group_hash(train_group_ids),
        },
        event_id=f"{candidate_root.resolve()}:candidate-started",
    )
    checkpoint, train_cost = trainer.train(
        train_records=train_records,
        initial_checkpoint=None,
        config={"technology_one": True},
        seed=seed,
        store=store,
    )
    checkpoint_artifact = store.put_json(
        kind="checkpoint",
        value=checkpoint,
        version=f"{task_model}-{task_revision}",
        parents=[dataset_artifact.artifact_id],
    )
    held_in, held_in_cost = evaluator.evaluate(
        checkpoint=checkpoint,
        records=held_in_records,
        config={"split": "held_in", "metric": "macro_f1"},
    )
    held_out, held_out_cost = evaluator.evaluate(
        checkpoint=checkpoint,
        records=held_out_records,
        config={"split": "held_out", "metric": "macro_f1"},
    )
    result = {
        "label": label,
        "harness": manifest.to_dict(),
        "dataset_artifact": to_jsonable(dataset_artifact),
        "checkpoint_artifact": to_jsonable(checkpoint_artifact),
        "train_groups": len(train_group_ids),
        "train_group_hash": _group_hash(train_group_ids),
        "held_in": held_in,
        "held_out": held_out,
        "budget": {
            "llm_calls": 0,
            "training_steps": train_cost.training_steps,
            "training_seconds": train_cost.training_seconds,
            "evaluator_calls": held_in_cost.evaluator_calls + held_out_cost.evaluator_calls,
        },
    }
    store.append_event(
        "candidate_completed",
        result,
        event_id=f"{candidate_root.resolve()}:candidate-completed",
    )
    (candidate_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return result


def run_server_technology_one(
    profile_path: Path,
    *,
    root: Path,
    output_dir: Path,
    run_id: str,
    seed: int | None = None,
    max_train_groups: int = 1000,
    max_eval_groups: int = 1000,
    train_group_budget: int | None = None,
) -> dict[str, Any]:
    experiment_started = time.perf_counter()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    run_seed = int(profile["seeds"][0] if seed is None else seed)
    preflight = run_preflight(profile_path, root)
    if preflight["errors"]:
        raise ValueError("Server preflight failed: " + "; ".join(preflight["errors"]))
    protocol = load_protocol(root / profile["dataset"]["protocol"])
    records = JsonlTaskAdapter(protocol).load(root / profile["dataset"]["train_jsonl"])
    validate_records(records, protocol)
    by_split = {
        split: _groups(record for record in records if record.split == split)
        for split in SPLITS
    }
    train_groups = _limit_groups(by_split["seed"], max_train_groups)
    held_in_groups = _limit_groups(by_split["held_in"], max_eval_groups)
    held_out_groups = _limit_groups(by_split["held_out"], max_eval_groups)
    current = initial_manifest(
        train_group_budget=train_group_budget
        or int(profile["stream"].get("train_group_budget", 1000))
    )
    held_in_records = [record for group in held_in_groups.values() for record in group]
    held_out_records = [record for group in held_out_groups.values() for record in group]
    edits, suspected, diagnosis = _make_candidates(
        current, train_groups, held_in_groups
    )
    manifests = {
        "fixed": current,
        "independent-edit": apply_edits(current, (edits["generator"],)),
        "selector-only-edit": apply_edits(current, (edits["selector"],)),
        "random-joint-edit": apply_edits(
            current, random_joint_edit(edits, seed=run_seed, size=2)
        ),
        "dependency-aware-joint-edit": apply_edits(
            current,
            tuple(
                edits[name]
                for name in dependency_closure(current, suspected)
                if name in edits
            ),
        ),
    }
    requested_budget = min(
        max_train_groups,
        int(current.components["selector"].config["train_group_budget"]),
    )
    eligible_counts = {
        name: len(_eligible_groups(manifest, train_groups))
        for name, manifest in manifests.items()
    }
    insufficient = {
        name: count
        for name, count in eligible_counts.items()
        if count < requested_budget
    }
    if insufficient:
        raise ValueError(
            "Candidate matrix cannot satisfy the declared shared training "
            f"budget={requested_budget}; eligible groups={insufficient}. "
            "Increase --max-train-groups or reduce --train-group-budget."
        )
    shared_budget = requested_budget

    run_root = output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    estimated_candidate_seconds = float(
        os.environ.get(
            "LEGO_ESTIMATED_CANDIDATE_SECONDS",
            profile.get("progress", {}).get("estimated_candidate_seconds", 180.0),
        )
    )
    setup_seconds = time.perf_counter() - experiment_started
    estimated_total_seconds = setup_seconds + estimated_candidate_seconds * len(METHODS)
    print(
        "[LEGO] Technology 1 estimated total: "
        f"~{estimated_total_seconds / 60.0:.1f} minutes "
        f"({estimated_candidate_seconds:.0f}s/candidate; setup {setup_seconds:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    progress = ProgressReporter(
        "Technology 1 candidate matrix",
        len(METHODS),
        unit="candidates",
        expected_item_seconds=estimated_candidate_seconds,
        initial_elapsed_seconds=setup_seconds,
    )
    for method in METHODS:
        selected, group_ids = _select_groups(
            manifests[method], train_groups, shared_budget, run_seed
        )
        results[method] = _run_candidate(
            root=run_root,
            label=method,
            manifest=manifests[method],
            train_records=_train_records(manifests[method], selected),
            train_group_ids=group_ids,
            held_in_records=held_in_records,
            held_out_records=held_out_records,
            profile=profile,
            seed=run_seed,
        )
        progress.report(len(results))
    progress.finish()
    wall_clock_seconds = time.perf_counter() - experiment_started

    baseline = results["fixed"]
    baseline_score = float(baseline["held_in"].get("macro_f1", 0.0))
    selected_method = max(
        METHODS,
        key=lambda method: (
            float(results[method]["held_in"].get("macro_f1", 0.0)),
            method == "fixed",
        ),
    )
    candidates: dict[str, Any] = {}
    for method in METHODS[1:]:
        candidate = results[method]
        score = float(candidate["held_in"].get("macro_f1", 0.0))
        candidates[method] = {
            "held_in_macro_f1": score,
            "accepted_on_held_in": score > baseline_score and method == selected_method,
            "held_out_report_only": True,
            "held_out_macro_f1": float(candidate["held_out"].get("macro_f1", 0.0)),
            "rollback_required": not (score > baseline_score and method == selected_method),
        }
    chain = HarnessVersionChain(current)
    transitions: list[dict[str, Any]] = []
    for method in METHODS[1:]:
        candidate = results[method]
        score = float(candidate["held_in"].get("macro_f1", 0.0))
        accepted = score > baseline_score and method == selected_method
        if accepted:
            chain.accept(manifests[method])
            transitions.append(
                {
                    "method": method,
                    "candidate_version": manifests[method].version,
                    "action": "accepted",
                    "active_version": chain.current.version,
                }
            )
        else:
            candidate_chain = HarnessVersionChain(current)
            candidate_chain.accept(manifests[method])
            restored = candidate_chain.rollback()
            transitions.append(
                {
                    "method": method,
                    "candidate_version": manifests[method].version,
                    "action": "rejected_and_rolled_back",
                    "active_version": restored.version,
                }
            )
    active_method = selected_method
    summary = {
        "experiment": "technology-one-server-candidate-matrix",
        "run_id": run_id,
        "dataset": profile["dataset"],
        "decision_policy": "deterministic-diagnostic-candidates",
        "task_model": profile["task_model"],
        "evaluator": profile["evaluator"],
        "seed": run_seed,
        "diagnosis": diagnosis,
        "suspected_components": list(suspected),
        "candidate_edits": {
            name: {"component": edit.component, "patch": edit.patch}
            for name, edit in edits.items()
        },
        "requested_train_group_budget": requested_budget,
        "shared_train_group_budget": shared_budget,
        "eligible_train_groups_by_method": eligible_counts,
        "selection": {
            "metric": "held_in_macro_f1",
            "baseline_held_in_macro_f1": baseline_score,
            "active_method": active_method,
            "active_harness_version": manifests[active_method].version,
            "candidates": candidates,
            "version_transitions": transitions,
        },
        "timing": {
            "estimated_candidate_seconds_at_start": estimated_candidate_seconds,
            "estimated_setup_seconds_at_start": setup_seconds,
            "estimated_total_seconds_at_start": estimated_total_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            "wall_clock_minutes": wall_clock_seconds / 60.0,
            "actual_average_candidate_seconds": (
                (wall_clock_seconds - setup_seconds) / len(METHODS)
                if METHODS
                else 0.0
            ),
        },
        "results": results,
        "final_test_used": False,
        "interpretation": (
            "Server trainer/evaluator candidate matrix with deterministic candidate "
            "construction; not yet a decision-LLM-generated Harness evolution run."
        ),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LEGO technology one with server trainer/evaluator commands."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-groups", type=int, default=1000)
    parser.add_argument("--max-eval-groups", type=int, default=1000)
    parser.add_argument("--train-group-budget", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run_server_technology_one(
                args.profile,
                root=args.root,
                output_dir=args.output_dir,
                run_id=args.run_id,
                seed=args.seed,
                max_train_groups=args.max_train_groups,
                max_eval_groups=args.max_eval_groups,
                train_group_budget=args.train_group_budget,
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
