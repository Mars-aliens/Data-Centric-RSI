from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from local_experiments.lego.evolution_reporting import (  # noqa: E402
    EXPECTED_SEEDS,
    audit_evolution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real Technology One evolution summaries.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_evolution(args.root, tuple(args.seeds), args.batches)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(f"status: {report['status']}")
        print(f"run_count: {report['run_count']}")
        print(f"seeds: {report['seeds']}")
        print(f"missing: {report['missing_seeds']}")
        print(f"unexpected: {report['unexpected_seeds']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
