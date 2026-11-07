"""Compare model-simulated feedback with human feedback on the same queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.feedback.agreement import compare_feedback
from experiments.feedback.annotation import read_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_feedback", help="model-simulated feedback JSONL")
    parser.add_argument("human_feedback", help="independent human feedback JSONL")
    parser.add_argument("--output", required=True, help="output JSON report")
    args = parser.parse_args()

    model_rows = read_feedback(args.model_feedback, expected_source="model_simulated")
    human_rows = read_feedback(args.human_feedback, expected_source="live_human")
    report = compare_feedback(model_rows, human_rows)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
