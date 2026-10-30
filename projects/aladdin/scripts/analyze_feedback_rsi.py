"""Analyze a completed Aladdin feedback-driven RSI run."""

from __future__ import annotations

import argparse
import json

from experiments.rsi.analysis import analyze_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="path to the completed result.json")
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    report = analyze_file(args.result, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
