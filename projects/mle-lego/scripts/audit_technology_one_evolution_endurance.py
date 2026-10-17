from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from local_experiments.lego.evolution_reporting import audit_evolution  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a long-horizon Technology One evolution run."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit_evolution(args.root, (args.seed,), args.batches)
    progress_path = args.root / "progress.json"
    if not progress_path.exists():
        direct_summary = args.root / "summary.json"
        if direct_summary.exists():
            progress_path = args.root / "progress.json"
    progress = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    report["progress"] = progress
    if not progress:
        report["errors"].append("missing progress.json")
        report["status"] = "FAIL"
    elif progress.get("status") != "complete":
        report["errors"].append("progress.json does not report complete")
        report["status"] = "FAIL"

    summaries = list(args.root.glob("summary.json"))
    if summaries:
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        if summary.get("run_status") != "complete":
            report["errors"].append("summary run_status is not complete")
            report["status"] = "FAIL"
        if summary.get("resume_supported") is not True:
            report["errors"].append("summary does not declare resume support")
            report["status"] = "FAIL"
        if not isinstance(summary.get("reliability"), dict):
            report["errors"].append("summary lacks reliability accounting")
            report["status"] = "FAIL"

    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(f"status: {report['status']}")
        print(f"run_count: {report['run_count']}")
        print(f"batches: {args.batches}")
        print(f"missing: {report['missing_seeds']}")
        if progress:
            print(
                "progress: "
                f"{progress.get('completed_batches')}/{progress.get('target_batches')} "
                f"phase={progress.get('phase')}"
            )
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
