from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ARMS = (
    "no_experience",
    "reflection_only",
    "executable_unmanaged",
    "lifecycle_governed",
)
COMPARISONS = (
    ("lifecycle_governed", "executable_unmanaged"),
    ("executable_unmanaged", "reflection_only"),
    ("reflection_only", "no_experience"),
)


def _load(path: Path, *, expected_batches: int | None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing summary: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"run is not complete: {path}")
    if expected_batches is not None and value.get("completed_batches") != expected_batches:
        raise ValueError(f"unexpected completed batch count: {path}")
    if value.get("held_out_used") is not False or value.get("final_test_used") is not False:
        raise ValueError(f"held-out/final-test boundary violated: {path}")
    if value.get("training_data_policy", {}).get("arm") not in ARMS:
        raise ValueError(f"missing or invalid T2 arm in summary: {path}")
    return value


def summarize(root: Path, seeds: list[int], *, expected_batches: int | None) -> dict[str, Any]:
    runs: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            path = root / arm / f"seed-{seed}" / "summary.json"
            summary = _load(path, expected_batches=expected_batches)
            if summary.get("seed") != seed:
                raise ValueError(f"seed mismatch: {path}")
            if summary["training_data_policy"].get("arm") != arm:
                raise ValueError(f"arm mismatch: {path}")
            runs[arm][seed] = summary

    reference = runs[ARMS[0]][seeds[0]]
    reference_model = reference.get("model_config", {})
    for arm in ARMS:
        for seed, summary in runs[arm].items():
            model = summary.get("model_config", {})
            for key in (
                "base_model_id", "base_model_revision", "max_steps", "learning_rate",
                "per_device_batch_size", "gradient_accumulation_steps", "max_length",
                "lora_r", "lora_alpha", "lora_dropout", "target_modules",
            ):
                if model.get(key) != reference_model.get(key):
                    raise ValueError(f"model config mismatch at {key}: arm={arm}, seed={seed}")
    for seed in seeds:
        expected_manifest = runs[ARMS[0]][seed].get("task_manifest_sha256")
        for arm in ARMS[1:]:
            if runs[arm][seed].get("task_manifest_sha256") != expected_manifest:
                raise ValueError(f"task manifest mismatch within paired seed {seed}: arm={arm}")

    arm_results: dict[str, Any] = {}
    for arm, by_seed in runs.items():
        values = {
            seed: summary.get("prequential_metrics", {}).get("macro_f1")
            for seed, summary in by_seed.items()
        }
        if any(not isinstance(value, (int, float)) for value in values.values()):
            raise ValueError(f"missing prequential macro-F1 for arm {arm}")
        arm_results[arm] = {
            "runs": len(by_seed),
            "macro_f1_by_seed": values,
            "macro_f1_mean": statistics.mean(values.values()),
            "macro_f1_stdev": statistics.stdev(values.values()) if len(values) > 1 else None,
            "completed_batches": {
                seed: summary["completed_batches"] for seed, summary in by_seed.items()
            },
            "training_records_selected": {
                seed: sum(
                    batch.get("training_data_policy", {}).get("records_selected", 0)
                    for batch in summary.get("batch_history", [])
                )
                for seed, summary in by_seed.items()
            },
            "training_records_dropped": {
                seed: sum(
                    batch.get("training_data_policy", {}).get("records_dropped", 0)
                    for batch in summary.get("batch_history", [])
                )
                for seed, summary in by_seed.items()
            },
        }

    paired: dict[str, Any] = {}
    for left, right in COMPARISONS:
        differences = {
            seed: (
                runs[left][seed]["prequential_metrics"]["macro_f1"]
                - runs[right][seed]["prequential_metrics"]["macro_f1"]
            )
            for seed in seeds
        }
        paired[f"{left}_minus_{right}"] = {
            "difference_by_seed": differences,
            "mean_difference": statistics.mean(differences.values()),
        }

    return {
        "schema": "mle-lego-t2-summary-v1",
        "root": str(root.resolve()),
        "seeds": seeds,
        "expected_batches": expected_batches,
        "task_manifest_sha256_by_seed": {
            seed: runs[ARMS[0]][seed].get("task_manifest_sha256") for seed in seeds
        },
        "held_out_used": False,
        "final_test_used": False,
        "arms": arm_results,
        "paired_macro_f1_differences": paired,
        "interpretation": "Descriptive paired pilot results; not a formal significance test.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize a paired T2 run matrix.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-batches", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.root, args.seeds, expected_batches=args.expected_batches)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
