from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(root: Path, seeds: list[int]) -> dict[str, Any]:
    runs: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = root / f"seed-{seed}" / "summary.json"
        if not path.is_file():
            raise ValueError(f"missing T3 summary: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "mle-lego-t3-evaluation-v1":
            raise ValueError(f"invalid T3 summary: {path}")
        if value.get("held_out_used") is not False or value.get("final_test_used") is not False:
            raise ValueError(f"held-out/final-test boundary violated: {path}")
        runs[seed] = value

    policies = ("single_source_batch", "arrival_as_current", "source_attributed_window")
    policy_results: dict[str, Any] = {}
    for policy in policies:
        actions = {seed: value["policies"][policy]["action"] for seed, value in runs.items()}
        scores = {
            seed: value["policies"][policy].get("mean_gain", value["policies"][policy].get("gain"))
            for seed, value in runs.items()
        }
        scores = {seed: score for seed, score in scores.items() if isinstance(score, (int, float))}
        policy_results[policy] = {
            "actions_by_seed": actions,
            "action_counts": {
                action: sum(value == action for value in actions.values())
                for action in sorted(set(actions.values()))
            },
            "gain_by_seed": scores,
            "mean_gain": statistics.mean(scores.values()) if scores else None,
            "misattributed_feedback_by_seed": {
                seed: value["policies"][policy].get("misattributed_feedback_count", 0)
                for seed, value in runs.items()
            },
        }
    future_results = {
        seed: value.get("future_window_evaluation") for seed, value in runs.items()
    }
    future_policy_summary: dict[str, Any] = {}
    for policy in policies:
        scored = {
            seed: item["recommendations"][policy]
            for seed, item in future_results.items()
            if isinstance(item, dict)
            and isinstance(item.get("recommendations"), dict)
            and item["recommendations"].get(policy, {}).get("scorable") is True
        }
        accepted = {
            seed: item for seed, item in scored.items()
            if item["action"] == "accept_candidate"
        }
        rejected = {
            seed: item for seed, item in scored.items()
            if item["action"] == "keep_stable"
        }
        false_accept = sum(item["false_accept"] is True for item in accepted.values())
        false_reject = sum(item["false_reject"] is True for item in rejected.values())
        future_policy_summary[policy] = {
            "scorable_runs": len(scored),
            "accept_recommendations": len(accepted),
            "false_accept_count": false_accept,
            "false_accept_rate": false_accept / len(accepted) if accepted else None,
            "reject_recommendations": len(rejected),
            "false_reject_count": false_reject,
            "false_reject_rate": false_reject / len(rejected) if rejected else None,
            "by_seed": scored,
        }
    return {
        "schema": "mle-lego-t3-matrix-summary-v1",
        "seeds": seeds,
        "runs": len(runs),
        "policies": policy_results,
        "future_window_validation": future_policy_summary,
        "future_window_by_seed": future_results,
        "held_out_used": False,
        "final_test_used": False,
        "interpretation": "Descriptive offline replay; does not establish online promotion or rollback effectiveness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired T3 feedback-replay runs.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.root, args.seeds)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
