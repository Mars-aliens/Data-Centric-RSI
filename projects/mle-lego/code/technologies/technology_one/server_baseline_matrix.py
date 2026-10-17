from __future__ import annotations

"""Run the deployable LEGO baseline/control/adaptation matrix.

This runner intentionally labels external methods as task adaptations.  It
does not claim to reproduce the original papers' environments or code.  The
goal is to make every required comparison arm executable under the same
HelpSteer2/Qwen Task Model protocol before final-test unlock.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .data_validation import validate_records
from .harness import HarnessManifest, apply_edits, dependency_closure, random_joint_edit
from .protocol import JsonlTaskAdapter, load_protocol
from .progress import ProgressReporter
from .server_preflight import REQUIRED_METHODS, run_preflight
from .server_technical_one import _run_candidate
from .server_technical_one_baseline import _audit_training_records
from .technical_one_local import (
    SPLITS,
    _eligible_groups,
    _groups,
    _limit_groups,
    _make_candidates,
    _select_groups,
    _train_records,
    initial_manifest,
)


BASELINE_METHOD_ORDER = (
    "fixed-harness-react",
    "react-reflexion",
    "general-self-harness",
    "curateevo-adaptation",
    "independent-edit",
    "random-joint-edit",
    "dependency-aware-joint-edit",
    "one-batch-selection",
    "three-batch-selection",
    "same-memory-no-process-update",
)


METHOD_SEMANTICS: dict[str, dict[str, Any]] = {
    "fixed-harness-react": {
        "role": "primary_control",
        "implementation_kind": "project_control",
        "persistent_harness_update": False,
        "external_reproduction": False,
        "adaptation_note": "Fixed data Harness control; no persistent edit.",
    },
    "react-reflexion": {
        "role": "memory_control",
        "implementation_kind": "published_method_adaptation",
        "persistent_harness_update": False,
        "external_reproduction": False,
        "adaptation_note": "Text-only Reflexion memory; the executable Harness remains fixed.",
    },
    "general-self-harness": {
        "role": "method_baseline",
        "implementation_kind": "published_method_adaptation",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "adaptation_note": "Local edit baseline without LEGO dependency-guided grouping.",
    },
    "curateevo-adaptation": {
        "role": "data_pipeline_baseline",
        "implementation_kind": "published_method_adaptation",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "adaptation_note": "Curation/filter edit adaptation; not the original CurateEvo benchmark.",
    },
    "independent-edit": {
        "role": "required_ablation",
        "implementation_kind": "project_ablation",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "adaptation_note": "Single component edit only.",
    },
    "random-joint-edit": {
        "role": "search_control",
        "implementation_kind": "project_control",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "adaptation_note": "Same-size joint edit sampled without dependency guidance.",
    },
    "dependency-aware-joint-edit": {
        "role": "proposed_method",
        "implementation_kind": "project_method",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "adaptation_note": "Dependency closure chooses the component group.",
    },
    "one-batch-selection": {
        "role": "horizon_ablation",
        "implementation_kind": "project_ablation",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "selection_horizon_batches": 1,
        "adaptation_note": "Uses the same candidate as the dependency-aware arm but records a one-batch acceptance horizon.",
    },
    "three-batch-selection": {
        "role": "horizon_ablation",
        "implementation_kind": "project_ablation",
        "persistent_harness_update": True,
        "external_reproduction": False,
        "selection_horizon_batches": 3,
        "adaptation_note": "Uses the same candidate as the dependency-aware arm and records the proposed three-batch acceptance horizon.",
    },
    "same-memory-no-process-update": {
        "role": "memory_vs_process_ablation",
        "implementation_kind": "project_ablation",
        "persistent_harness_update": False,
        "external_reproduction": False,
        "adaptation_note": "Receives the same initial memory metadata but keeps the executable process fixed.",
    },
}


def _manifest_hash(manifest: HarnessManifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_baseline_manifests(
    current: HarnessManifest,
    *,
    edits: dict[str, Any],
    suspected_components: tuple[str, ...],
    seed: int,
) -> dict[str, HarnessManifest]:
    dependency_edits = tuple(
        edits[name]
        for name in dependency_closure(current, suspected_components)
        if name in edits
    )
    dependency_manifest = apply_edits(current, dependency_edits)
    return {
        "fixed-harness-react": current,
        "react-reflexion": current,
        "general-self-harness": apply_edits(current, (edits["selector"],)),
        "curateevo-adaptation": apply_edits(current, (edits["filter"],)),
        "independent-edit": apply_edits(current, (edits["generator"],)),
        "random-joint-edit": apply_edits(
            current, random_joint_edit(edits, seed=seed, size=2)
        ),
        "dependency-aware-joint-edit": dependency_manifest,
        "one-batch-selection": dependency_manifest,
        "three-batch-selection": dependency_manifest,
        "same-memory-no-process-update": current,
    }


def run_server_baseline_matrix(
    profile_path: Path,
    *,
    root: Path,
    output_dir: Path,
    run_id: str,
    seed: int | None = None,
    max_train_groups: int = 3000,
    max_eval_groups: int = 1000,
    train_group_budget: int | None = None,
    methods: tuple[str, ...] = BASELINE_METHOD_ORDER,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preflight = run_preflight(profile_path, root)
    if preflight["errors"]:
        raise ValueError("Server preflight failed: " + "; ".join(preflight["errors"]))

    unknown = set(methods) - set(BASELINE_METHOD_ORDER)
    if unknown:
        raise ValueError(f"Unknown deployable baseline methods: {sorted(unknown)}")
    missing_required = REQUIRED_METHODS - set(profile.get("methods", []))
    if missing_required:
        raise ValueError(f"Profile is missing required baseline methods: {sorted(missing_required)}")

    protocol = load_protocol(root / profile["dataset"]["protocol"])
    records = JsonlTaskAdapter(protocol).load(root / profile["dataset"]["train_jsonl"])
    validate_records(records, protocol)
    run_seed = int(profile["seeds"][0] if seed is None else seed)
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
    edits, suspected, diagnosis = _make_candidates(current, train_groups, held_in_groups)
    manifests = build_baseline_manifests(
        current,
        edits=edits,
        suspected_components=tuple(suspected),
        seed=run_seed,
    )
    requested_budget = min(
        max_train_groups,
        int(current.components["selector"].config["train_group_budget"]),
    )
    eligible_counts = {
        method: len(_eligible_groups(manifests[method], train_groups))
        for method in methods
    }
    insufficient = {
        method: count
        for method, count in eligible_counts.items()
        if count < requested_budget
    }
    if insufficient:
        raise ValueError(
            "Baseline matrix cannot satisfy the declared natural-support "
            f"training budget={requested_budget}; eligible groups={insufficient}."
        )

    run_root = output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    estimated_candidate_seconds = float(
        os.environ.get(
            "LEGO_ESTIMATED_CANDIDATE_SECONDS",
            profile.get("progress", {}).get("estimated_candidate_seconds", 300.0),
        )
    )
    print(
        "[LEGO] Baseline matrix estimated total: "
        f"~{estimated_candidate_seconds * len(methods) / 60.0:.1f} minutes "
        f"({estimated_candidate_seconds:.0f}s/method)",
        file=sys.stderr,
        flush=True,
    )
    progress = ProgressReporter(
        "Technology 1 baseline matrix",
        len(methods),
        unit="methods",
        expected_item_seconds=estimated_candidate_seconds,
        initial_elapsed_seconds=time.perf_counter() - started,
    )
    results: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for method in methods:
        manifest = manifests[method]
        selected, group_ids = _select_groups(
            manifest, train_groups, requested_budget, run_seed
        )
        formatted = _train_records(manifest, selected)
        result_label = method.replace("/", "_")
        results[method] = _run_candidate(
            root=run_root,
            label=result_label,
            manifest=manifest,
            train_records=formatted,
            train_group_ids=group_ids,
            held_in_records=held_in_records,
            held_out_records=held_out_records,
            profile=profile,
            seed=run_seed,
        )
        audits[method] = _audit_training_records(
            selected,
            formatted,
            group_ids,
            manifest=manifest,
        )
        progress.report(len(results))
    progress.finish()

    baseline = results["fixed-harness-react"]
    baseline_held_in = float(baseline["held_in"].get("macro_f1", 0.0))
    method_reports: dict[str, Any] = {}
    for method in methods:
        score = float(results[method]["held_in"].get("macro_f1", 0.0))
        method_reports[method] = {
            **METHOD_SEMANTICS[method],
            "harness_version": manifests[method].version,
            "harness_hash": _manifest_hash(manifests[method]),
            "held_in_macro_f1": score,
            "held_out_macro_f1_report_only": float(
                results[method]["held_out"].get("macro_f1", 0.0)
            ),
            "delta_vs_fixed_held_in_macro_f1": score - baseline_held_in,
            "held_out_used_for_selection": False,
            "train_group_hash": audits[method]["train_group_hash"],
        }
    selected_method = max(
        methods,
        key=lambda method: (
            float(results[method]["held_in"].get("macro_f1", 0.0)),
            method == "fixed-harness-react",
        ),
    )
    summary = {
        "experiment": "technology-one-server-baseline-matrix",
        "run_id": run_id,
        "seed": run_seed,
        "dataset": profile["dataset"],
        "task_model": profile["task_model"],
        "evaluator": profile["evaluator"],
        "decision_policy": "deterministic-baseline-adaptation-matrix",
        "decision_llm_used_for_harness_edits": False,
        "common_support": False,
        "final_test_used": False,
        "max_train_groups": max_train_groups,
        "max_eval_groups": max_eval_groups,
        "requested_train_group_budget": requested_budget,
        "eligible_train_groups_by_method": eligible_counts,
        "diagnosis": diagnosis,
        "suspected_components": list(suspected),
        "candidate_edits": {
            name: {"component": edit.component, "patch": edit.patch, "rationale": edit.rationale}
            for name, edit in edits.items()
        },
        "selection": {
            "metric": "held_in_macro_f1",
            "selected_method": selected_method,
            "baseline_method": "fixed-harness-react",
            "held_out_report_only": True,
            "final_test_locked": True,
        },
        "method_reports": method_reports,
        "train_group_audits": audits,
        "results": results,
        "timing": {
            "wall_clock_seconds": time.perf_counter() - started,
            "estimated_candidate_seconds": estimated_candidate_seconds,
        },
        "interpretation": (
            "Deployable baseline/control/adaptation matrix for the current "
            "HelpSteer2 technology-one protocol. External-paper rows are "
            "task adaptations, not official reproductions."
        ),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all deployable LEGO baseline/control/adaptation arms."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-groups", type=int, default=3000)
    parser.add_argument("--max-eval-groups", type=int, default=1000)
    parser.add_argument("--train-group-budget", type=int)
    parser.add_argument("--methods", nargs="*", default=list(BASELINE_METHOD_ORDER))
    args = parser.parse_args()
    print(
        json.dumps(
            run_server_baseline_matrix(
                args.profile,
                root=args.root,
                output_dir=args.output_dir,
                run_id=args.run_id,
                seed=args.seed,
                max_train_groups=args.max_train_groups,
                max_eval_groups=args.max_eval_groups,
                train_group_budget=args.train_group_budget,
                methods=tuple(args.methods),
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
