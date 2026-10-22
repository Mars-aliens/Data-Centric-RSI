from __future__ import annotations

import argparse
from pathlib import Path

from .helpsteer2 import load_helpsteer2_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized HelpSteer2 JSONL for LEGO.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--task-id", default="helpsteer2-preference")
    parser.add_argument("--feedback-delay", type=int, default=2, choices=(0, 2, 5))
    args = parser.parse_args()
    records = load_helpsteer2_records(
        split=args.split,
        task_id=args.task_id,
        feedback_delay=args.feedback_delay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print({"output": str(args.output), "records": write_jsonl(records, str(args.output))})


if __name__ == "__main__":
    main()

