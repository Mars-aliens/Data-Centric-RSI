from __future__ import annotations

import argparse
import json
from pathlib import Path

from technologies.technology_three.trace_evaluation import (
    evaluate_trace,
    evaluate_future_window,
    load_t1_summary,
    trace_from_t1_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay delayed, version-attributed feedback for T3 from a completed T1 run."
    )
    parser.add_argument("--t1-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-batches", type=int, default=3)
    parser.add_argument("--outcome-window-batches", type=int, default=3)
    parser.add_argument("--min-mean-gain", type=float, default=0.0)
    parser.add_argument("--max-regression", type=float, default=0.05)
    args = parser.parse_args()
    try:
        trace = trace_from_t1_summary(load_t1_summary(args.t1_summary))
        result = evaluate_trace(
            trace,
            window_batches=args.window_batches,
            min_mean_gain=args.min_mean_gain,
            max_regression=args.max_regression,
        )
        result["future_window_evaluation"] = evaluate_future_window(
            trace,
            decision_window_batches=args.window_batches,
            outcome_window_batches=args.outcome_window_batches,
            min_mean_gain=args.min_mean_gain,
            max_regression=args.max_regression,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feedback-trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
