from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from local_experiments.lego.evolution_reporting import (  # noqa: E402
    EXPECTED_SEEDS,
    audit_evolution,
    summarize_evolution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real Technology One evolution runs.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    audit = audit_evolution(args.root, tuple(args.seeds), args.batches)
    report = {"audit": audit, "summary": summarize_evolution(args.root, tuple(args.seeds))}
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"audit_status: {audit['status']}")
    print(f"completed: {report['summary']['completed_rows']}/{len(args.seeds)}")
    print("seed  initial_out  final_out  delta_out  accepted  rejected  rollback  llm_calls")
    for row in report["summary"]["rows"]:
        print(
            f"{row['seed']:4d}  {row['initial_held_out']:.4f}       "
            f"{row['final_active_held_out']:.4f}    {row['delta_held_out']:+.4f}    "
            f"{row['accepted']:8d}  {row['rejected']:8d}  {row['rollbacks']:8d}  "
            f"{row['llm_calls']:9d}"
        )
    aggregate = report["summary"]["aggregate"]
    for field in ("initial_held_out", "final_active_held_out", "delta_held_out"):
        value = aggregate[field]
        print(f"{field}: mean={value['mean']:.4f} sd={value['sd']:.4f} n={value['n']}")

    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
