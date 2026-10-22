from __future__ import annotations

"""Small real-data smoke test for LEGO technology one.

This intentionally uses the existing lightweight TF-IDF task model so the
comparison is runnable on a CPU. It tests the Harness edit protocol on real
HelpSteer2 records; it is not a final LLM result.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from sklearn.metrics import f1_score

from core_system.mle_core.models import TaskRecord
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
    _checkpoint_scores,
)
from core_system.mle_core.store import ArtifactStore

from .harness import (
    HarnessComponent,
    HarnessEdit,
    HarnessManifest,
    HarnessVersionChain,
    apply_edits,
    dependency_closure,
    random_joint_edit,
)
from .protocol import DataProtocol, JsonlTaskAdapter, load_protocol


METHODS = (
    "fixed",
    "independent-edit",
    "selector-only-edit",
    "random-joint-edit",
    "dependency-aware-joint-edit",
)
SPLITS = ("seed", "held_in", "held_out")


def _bucket(group_id: str) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 10


def _groups(records: Iterable[TaskRecord]) -> dict[str, list[TaskRecord]]:
    grouped: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in records:
        if record.source_group_id is None:
            raise ValueError(f"HelpSteer2 record has no source group: {record.record_id}")
        grouped[record.source_group_id].append(record)
    return dict(sorted(grouped.items()))


def _response_words(record: TaskRecord) -> int:
    text = str(record.input)
    return len(text.split("Response:", 1)[-1].split())


def _group_margin(records: list[TaskRecord]) -> float:
    scores = [float(record.metadata.get("score_mean", 0.0)) for record in records]
    return max(scores) - min(scores) if scores else 0.0


def initial_manifest(train_group_budget: int = 240) -> HarnessManifest:
    if train_group_budget < 1:
        raise ValueError("train_group_budget must be positive")
    # The components are deliberately named by their real responsibilities:
    # group coverage feeds response selection, which feeds quality filtering.
    return HarnessManifest(
        components={
            "generator": HarnessComponent(
                "generator",
                "group_coverage",
                {"allowed_buckets": list(range(9))},
            ),
            "selector": HarnessComponent(
                "selector",
                "response_selection",
                {"max_response_words": 80, "train_group_budget": train_group_budget},
                ("generator",),
            ),
            "filter": HarnessComponent(
                "filter",
                "quality_filter",
                {"min_group_margin": 0.0},
                ("selector",),
            ),
            "memory": HarnessComponent("memory", "experience_memory", {"enabled": False}),
            "skill": HarnessComponent("skill", "executable_skill", {"enabled": False}, ("memory",)),
        }
    )


def _eligible_groups(
    manifest: HarnessManifest, groups: dict[str, list[TaskRecord]]
) -> dict[str, list[TaskRecord]]:
    generator = manifest.components["generator"].config
    quality_filter = manifest.components["filter"].config
    allowed = set(int(item) for item in generator["allowed_buckets"])
    min_margin = float(quality_filter["min_group_margin"])
    eligible: dict[str, list[TaskRecord]] = {}
    for group_id, records in groups.items():
        # HelpSteer2 labels are relative within a prompt group; keep complete pairs.
        if len(records) != 2:
            continue
        if _bucket(group_id) not in allowed:
            continue
        if _group_margin(records) < min_margin:
            continue
        eligible[group_id] = records
    return eligible


def _select_groups(
    manifest: HarnessManifest,
    groups: dict[str, list[TaskRecord]],
    group_budget: int,
    seed: int,
) -> tuple[list[TaskRecord], list[str]]:
    eligible = _eligible_groups(manifest, groups)
    if len(eligible) < group_budget:
        raise ValueError(
            f"Harness {manifest.version} has {len(eligible)} eligible groups, "
            f"below the shared budget of {group_budget}"
        )
    rng = random.Random(seed)
    group_ids = rng.sample(sorted(eligible), group_budget)
    selected_records: list[TaskRecord] = []
    # Keep every response in a prompt group together because labels are relative
    # to that group. Sampling is without replacement for fair training exposure.
    for group_id in group_ids:
        selected_records.extend(eligible[group_id])
    return selected_records, group_ids


def _format_record(record: TaskRecord, max_words: int) -> TaskRecord:
    text = str(record.input)
    prompt, response = text.split("Response:", 1)
    response_words = response.split()
    formatted = f"{prompt}Response: {' '.join(response_words[:max_words])}"
    return TaskRecord(
        record_id=record.record_id,
        task_id=record.task_id,
        split=record.split,
        input=formatted,
        target=record.target,
        source_group_id=record.source_group_id,
        metadata={**record.metadata, "harness_formatted": True},
        parent_record_ids=record.parent_record_ids,
    )


def _train_records(manifest: HarnessManifest, records: list[TaskRecord]) -> list[TaskRecord]:
    max_words = int(manifest.components["selector"].config["max_response_words"])
    return [_format_record(record, max_words) for record in records]


def _evaluate(checkpoint: dict[str, Any], records: list[TaskRecord]) -> dict[str, float]:
    labels: list[str] = []
    predictions: list[str] = []
    for record in records:
        scores = _checkpoint_scores(checkpoint, str(record.input))
        labels.append(str(record.target))
        predictions.append(max(scores, key=lambda label: (scores[label], label)))
    return {
        "accuracy": sum(left == right for left, right in zip(labels, predictions, strict=True)) / len(labels),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "num_records": float(len(labels)),
    }


def _make_candidates(
    current: HarnessManifest,
    train_groups: dict[str, list[TaskRecord]],
    held_in_groups: dict[str, list[TaskRecord]],
) -> tuple[dict[str, HarnessEdit], tuple[str, ...], dict[str, Any]]:
    configured = set(current.components["generator"].config["allowed_buckets"])
    missing_counts: dict[int, int] = defaultdict(int)
    for group_id in held_in_groups:
        bucket = _bucket(group_id)
        if bucket not in configured:
            missing_counts[bucket] += 1
    if not missing_counts:
        raise RuntimeError("The smoke-test baseline did not expose a held-in coverage failure")
    missing_bucket = max(missing_counts, key=lambda bucket: (missing_counts[bucket], -bucket))
    current_max_words = int(current.components["selector"].config["max_response_words"])
    edits = {
        "generator": HarnessEdit(
            "generator",
            {"allowed_buckets": sorted(configured | {missing_bucket})},
            "Restore the most common held-in prompt-group coverage gap.",
        ),
        "selector": HarnessEdit(
            "selector",
            {"max_response_words": current_max_words * 2},
            "Increase the complete-group response limit after observing truncation pressure.",
        ),
        "filter": HarnessEdit(
            "filter",
            {"min_group_margin": 0.5},
            "Test a quality-margin filter as a same-budget distractor edit.",
        ),
    }
    return edits, ("generator", "selector"), {
        "missing_bucket": missing_bucket,
        "missing_group_counts": dict(sorted(missing_counts.items())),
        "initial_train_group_count": len(train_groups),
        "held_in_group_count": len(held_in_groups),
    }


def _run_model(
    root: Path,
    label: str,
    manifest: HarnessManifest,
    train_records: list[TaskRecord],
    train_group_ids: list[str],
    held_in_records: list[TaskRecord],
    held_out_records: list[TaskRecord],
    seed: int,
) -> dict[str, Any]:
    store = ArtifactStore(root / label)
    trainer = TfidfLogisticTrainer()
    checkpoint, train_cost = trainer.train(
        train_records=train_records,
        initial_checkpoint=None,
        config={"max_iter": 80, "max_features": 12000, "min_df": 2, "steps": 1},
        seed=seed,
        store=store,
    )
    evaluator = TfidfLogisticEvaluator()
    held_in, held_in_cost = evaluator.evaluate(
        checkpoint=checkpoint, records=held_in_records, config={}
    )
    held_out, held_out_cost = evaluator.evaluate(
        checkpoint=checkpoint, records=held_out_records, config={}
    )
    held_in.update(_evaluate(checkpoint, held_in_records))
    held_out.update(_evaluate(checkpoint, held_out_records))
    result = {
        "label": label,
        "harness": manifest.to_dict(),
        "train_records": len(train_records),
        "train_groups": len(train_group_ids),
        "train_group_ids": train_group_ids,
        "held_in": held_in,
        "held_out": held_out,
        "budget": {
            "training_steps": train_cost.training_steps,
            "evaluator_calls": held_in_cost.evaluator_calls + held_out_cost.evaluator_calls,
        },
    }
    (root / label / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _limit_groups(groups: dict[str, list[TaskRecord]], limit: int) -> dict[str, list[TaskRecord]]:
    return dict(sorted(groups.items(), key=lambda item: hashlib.sha256(item[0].encode()).hexdigest())[:limit])


def run_experiment(
    dataset: Path,
    output_dir: Path,
    *,
    max_train_groups: int = 300,
    max_eval_groups: int = 300,
    seed: int = 17,
    train_group_budget: int = 240,
) -> dict[str, Any]:
    protocol: DataProtocol = load_protocol(Path(__file__).with_name("data_protocol.json"))
    records = JsonlTaskAdapter(protocol).load(dataset)
    by_split = {split: _groups(record for record in records if record.split == split) for split in SPLITS}
    train_groups = _limit_groups(by_split["seed"], max_train_groups)
    held_in_groups = _limit_groups(by_split["held_in"], max_eval_groups)
    held_out_groups = _limit_groups(by_split["held_out"], max_eval_groups)
    current = initial_manifest(train_group_budget)
    held_in_records = [record for group in held_in_groups.values() for record in group]
    held_out_records = [record for group in held_out_groups.values() for record in group]
    edits, suspected, diagnosis = _make_candidates(current, train_groups, held_in_groups)
    independent = apply_edits(current, (edits["generator"],))
    selector_only = apply_edits(current, (edits["selector"],))
    random_edits = random_joint_edit(edits, seed=seed, size=2)
    random_manifest = apply_edits(current, random_edits)
    dependency_names = dependency_closure(current, suspected)
    dependency_manifest = apply_edits(
        current, tuple(edits[name] for name in dependency_names if name in edits)
    )
    manifests = {
        "fixed": current,
        "independent-edit": independent,
        "selector-only-edit": selector_only,
        "random-joint-edit": random_manifest,
        "dependency-aware-joint-edit": dependency_manifest,
    }
    requested_group_budget = min(
        max_train_groups,
        int(current.components["selector"].config["train_group_budget"]),
    )
    eligible_counts = {
        name: len(_eligible_groups(manifest, train_groups))
        for name, manifest in manifests.items()
    }
    shared_group_budget = min(requested_group_budget, *eligible_counts.values())
    if shared_group_budget < 1:
        raise ValueError(f"No common training groups across methods: {eligible_counts}")
    root = output_dir
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for method, manifest in manifests.items():
        selected_records, selected_group_ids = _select_groups(
            manifest, train_groups, shared_group_budget, seed
        )
        results[method] = _run_model(
            root,
            method,
            manifest,
            _train_records(manifest, selected_records),
            selected_group_ids,
            held_in_records,
            held_out_records,
            seed,
        )
    baseline = results["fixed"]
    baseline_key = (baseline["held_in"]["macro_f1"], baseline["held_in"]["accuracy"])
    ranked_methods = ["fixed", *METHODS[1:]]
    selected_method = max(
        ranked_methods,
        key=lambda method: (
            results[method]["held_in"]["macro_f1"],
            results[method]["held_in"]["accuracy"],
            method == "fixed",
        ),
    )
    candidate_selection: dict[str, Any] = {}
    for method in METHODS[1:]:
        candidate = results[method]
        candidate_key = (candidate["held_in"]["macro_f1"], candidate["held_in"]["accuracy"])
        accepted = method == selected_method and candidate_key > baseline_key
        candidate_selection[method] = {
            "accepted_on_held_in": accepted,
            "selected_as_active_harness": method == selected_method,
            "held_out_used_for_selection": False,
            "held_out_regression_report": candidate["held_out"]["macro_f1"] < baseline["held_out"]["macro_f1"],
            "rollback_required": not accepted,
        }
    version_chain = HarnessVersionChain(current)
    version_transitions: list[dict[str, Any]] = []
    for method in METHODS[1:]:
        candidate_manifest = manifests[method]
        version_chain.accept(candidate_manifest)
        if method == selected_method:
            version_transitions.append(
                {
                    "method": method,
                    "candidate_version": candidate_manifest.version,
                    "action": "accepted",
                    "active_version": version_chain.current.version,
                }
            )
            break
        restored = version_chain.rollback()
        version_transitions.append(
            {
                "method": method,
                "candidate_version": candidate_manifest.version,
                "action": "rejected_and_rolled_back",
                "active_version": restored.version,
            }
        )
    if selected_method == "fixed":
        active_version = current.version
    else:
        active_version = version_chain.current.version
    summary = {
        "experiment": "technology-one-local-smoke-test",
        "dataset": str(dataset),
        "task_model": "tfidf-logistic-scaffold",
        "max_train_groups": max_train_groups,
        "requested_train_group_budget": requested_group_budget,
        "shared_train_group_budget": shared_group_budget,
        "eligible_train_groups_by_method": eligible_counts,
        "max_eval_groups_per_split": max_eval_groups,
        "seed": seed,
        "diagnosis": diagnosis,
        "suspected_components": list(suspected),
        "candidate_edits": {name: {"component": edit.component, "patch": edit.patch} for name, edit in edits.items()},
        "selection": {
            "baseline_held_in": baseline_key,
            "candidates": candidate_selection,
            "active_harness_after_selection": selected_method,
            "active_harness_version": active_version,
            "version_transitions": version_transitions,
        },
        "results": results,
        "interpretation": "This is a real-data Harness comparison with a lightweight task model, not a final LLM result.",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local HelpSteer2 technology-one smoke test.")
    parser.add_argument("dataset", type=Path, default=Path("data/helpsteer2-train.jsonl"), nargs="?")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/technology_one/runs/technology-one-local-fair"))
    parser.add_argument("--max-train-groups", type=int, default=300)
    parser.add_argument("--max-eval-groups", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-group-budget", type=int, default=240)
    args = parser.parse_args()
    summary = run_experiment(
        args.dataset,
        args.output_dir,
        max_train_groups=args.max_train_groups,
        max_eval_groups=args.max_eval_groups,
        seed=args.seed,
        train_group_budget=args.train_group_budget,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
