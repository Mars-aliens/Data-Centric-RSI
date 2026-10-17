"""Audit and summarize the frozen multi-split HH-RLHF Qwen experiment.

This command only reads completed JSON reports. It does not load models, call
vLLM, or recompute metrics. A non-zero exit status means at least one report
is not eligible for final reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from local_experiments.aladdin.analyze_hh_rlhf_qwen_lora_label_diagnostic import (
    _validate_qwen_diagnostic_report,
)


SEEDS = tuple(range(17, 27))
EXPECTED_SPLITS = (17, 23, 29, 37)
REQUIRED_STRATEGIES = {"seed-only", "random", "aladdin-qwen-reliable"}


def _read_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _validate_qwen_diagnostic_report(data, path)
    if data.get("status") != "completed":
        raise ValueError(f"{path}: status must be 'completed'")
    strategies = set(data["configuration"].get("strategies", ()))
    if not REQUIRED_STRATEGIES.issubset(strategies):
        raise ValueError(f"{path}: required main strategies are missing")
    return data


def _split_seed(data: dict[str, Any], path: Path) -> int:
    configuration = data["configuration"]
    audits = configuration.get("split_audit_by_seed", {})
    split_seeds = {int(audit["split_seed"]) for audit in audits.values()}
    if len(split_seeds) != 1:
        raise ValueError(f"{path}: expected one fixed split_seed, got {split_seeds}")
    split_seed = split_seeds.pop()
    if any(audit.get("prompts_with_multiple_splits") != 0 for audit in audits.values()):
        raise ValueError(f"{path}: prompt overlap is non-zero")
    return split_seed


def _paired_means(data: dict[str, Any]) -> dict[str, float]:
    rows = {(row["strategy"], int(row["seed"])): row for row in data["results"]}
    result: dict[str, float] = {}
    for channel in ("llm_feedback", "oracle_feedback"):
        for baseline in ("random", "seed-only"):
            values = []
            for seed in SEEDS:
                left = rows[("aladdin-qwen-reliable", seed)]["metrics"][channel][
                    "pairwise_accuracy"
                ]
                right = rows[(baseline, seed)]["metrics"][channel]["pairwise_accuracy"]
                values.append(float(left) - float(right))
            result[f"reliable_minus_{baseline}_{channel}"] = mean(values) * 100
    return result


def audit(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_SPLITS):
        raise ValueError(f"expected exactly {len(EXPECTED_SPLITS)} reports")

    reports = []
    seen_splits = set()
    for path in paths:
        data = _read_report(path)
        split_seed = _split_seed(data, path)
        if split_seed in seen_splits:
            raise ValueError(f"duplicate split_seed={split_seed}")
        seen_splits.add(split_seed)
        reports.append((split_seed, path, data))

    if seen_splits != set(EXPECTED_SPLITS):
        raise ValueError(f"expected split seeds {EXPECTED_SPLITS}, got {sorted(seen_splits)}")

    reports.sort(key=lambda item: item[0])
    split_rows = []
    for split_seed, path, data in reports:
        configuration = data["configuration"]
        summary = _paired_means(data)
        split_rows.append(
            {
                "split_seed": split_seed,
                "path": str(path),
                "status": data.get("status"),
                "results": len(data.get("results", [])),
                "strategies": configuration.get("strategies", []),
                "eval_batch_size": configuration.get("eval_batch_size"),
                "base_model_revision": data.get("base_model_revision"),
                "llm_model_revision": configuration.get("llm_model_revision"),
                "paired_mean_pp": summary,
            }
        )

    averages = {}
    for key in split_rows[0]["paired_mean_pp"]:
        averages[key] = mean(row["paired_mean_pp"][key] for row in split_rows)

    return {
        "status": "audited",
        "split_count": len(split_rows),
        "seeds": list(SEEDS),
        "split_seeds": list(EXPECTED_SPLITS),
        "split_rows": split_rows,
        "descriptive_split_level_average_pp": averages,
        "batch_size_note": (
            "split 17/23/29 use eval_batch_size=8; split 37 uses 16. "
            "This audit verifies metadata only; batch 8/16 numerical equivalence "
            "requires rerunning evaluation from the same checkpoints."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=4, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.reports)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
