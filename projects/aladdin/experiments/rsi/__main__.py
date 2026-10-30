"""Command-line entry point for the feedback-driven RSI experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import COMPARISONS, run_feedback_rsi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--comparison", choices=sorted(COMPARISONS))
    parser.add_argument("--annotator-id", help="pseudonymous reviewer ID; may use ALADDIN_ANNOTATOR_ID")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted run after validating its config and inputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    if args.comparison:
        config["comparison"] = args.comparison
    result = run_feedback_rsi(config, annotator_id=args.annotator_id, resume=args.resume)
    print(json.dumps({"status": result["status"], "result": str(Path(config["output"]) / "result.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
