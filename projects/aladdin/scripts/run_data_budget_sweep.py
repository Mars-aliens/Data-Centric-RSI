"""Run fixed-vs-recursive Aladdin comparisons over nested seed-data budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from experiments.rsi.runner import ROOT, run_feedback_rsi


def run_data_budget_sweep(
    raw_config: dict[str, Any], *, run_callable: Callable[..., dict[str, Any]] = run_feedback_rsi
) -> dict[str, Any]:
    limits = raw_config.get("seed_group_limits")
    if (not isinstance(limits, list) or len(limits) < 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in limits)
            or limits != sorted(set(limits))):
        raise ValueError("seed_group_limits must contain at least two increasing unique positive integers")
    output_root = Path(str(raw_config["output"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    runs = []
    for limit in limits:
        config = {key: value for key, value in raw_config.items() if key != "seed_group_limits"}
        config["seed_group_limit"] = limit
        config["output"] = str(output_root / f"seed-groups-{limit}")
        result = run_callable(config)
        if result.get("status") != "completed":
            raise RuntimeError(f"data-budget run did not complete for seed_group_limit={limit}")
        runs.append({
            "seed_group_limit": limit,
            "output": config["output"],
            "data": result.get("data", {}),
            "summary": result.get("summary", {}),
            "arms": result.get("arms", {}),
        })
    paired_deltas = []
    for run in runs:
        arm_scores = {}
        for arm_name in ("fixed_policy", "recursive_policy"):
            by_seed: dict[int, list[float]] = {}
            for row in run["arms"].get(arm_name, {}).get("rounds", []):
                score = (row.get("training") or {}).get("pairwise_accuracy")
                if score is not None:
                    by_seed.setdefault(int(row["seed"]), []).append(float(score))
            arm_scores[arm_name] = {
                seed: sum(values) / len(values) for seed, values in by_seed.items()
            }
        common = sorted(set(arm_scores["fixed_policy"]) & set(arm_scores["recursive_policy"]))
        paired_deltas.append({
            "seed_group_limit": run["seed_group_limit"],
            "per_seed_recursive_minus_fixed": {
                str(seed): arm_scores["recursive_policy"][seed] - arm_scores["fixed_policy"][seed]
                for seed in common
            },
        })
    report = {
        "status": "completed",
        "comparison": "data_budget_sweep",
        "seed_group_limits": limits,
        "nested_budgets": True,
        "paired_development_deltas": paired_deltas,
        "runs": runs,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep-summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = run_data_budget_sweep(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
