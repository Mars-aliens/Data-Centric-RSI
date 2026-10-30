"""Prepare, collect, and analyze the final blind human comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.evaluation.blind_pairwise import (
    _read_jsonl,
    analyze_blind_pairwise_feedback,
    collect_blind_pairwise_feedback,
    prepare_blind_pairwise_queue,
)
from experiments.rsi.runner import MODEL_ID, MODEL_REVISION
from experiments.evaluation.model_pairwise import collect_model_pairwise
from experiments.feedback.model_feedback import MODEL_ID as JUDGE_MODEL
from experiments.synthesis.generator import OpenAICompatibleGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="generate blinded outputs on final-test inputs")
    prepare.add_argument("--final-test", required=True)
    prepare.add_argument("--adapter-a", required=True)
    prepare.add_argument("--adapter-b", required=True)
    prepare.add_argument("--system-a", default="fixed_policy")
    prepare.add_argument("--system-b", default="recursive_policy")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--max-items", required=True, type=int)
    prepare.add_argument("--seed", type=int, default=1701)
    prepare.add_argument("--base-model", default=MODEL_ID)
    prepare.add_argument("--model-revision", default=MODEL_REVISION)
    prepare.add_argument("--max-new-tokens", type=int, default=256)
    prepare.add_argument("--allow-cpu", action="store_true")

    collect = subparsers.add_parser("collect", help="collect blinded human preferences")
    collect.add_argument("--queue", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--annotator-id", default=os.environ.get("ALADDIN_ANNOTATOR_ID", ""))

    model = subparsers.add_parser("collect-model", help="automatic Qwen3-32B judging in both answer orders")
    model.add_argument("--queue", required=True)
    model.add_argument("--output", required=True)
    model.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/chat/completions")
    model.add_argument("--api-key-env", default="OPENAI_API_KEY")
    model.add_argument("--max-tokens", type=int, default=512)
    model.add_argument("--timeout-seconds", type=float, default=300)

    analyze = subparsers.add_parser("analyze", help="analyze collected final-test ratings")
    analyze.add_argument("--feedback", required=True, nargs="+", help="one or more per-annotator JSONL files")
    analyze.add_argument("--key", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--seed", type=int, default=17)
    analyze.add_argument("--evaluation-mode", choices=("human", "model"), default="human")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_blind_pairwise_queue(
            final_test_path=args.final_test,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
            system_a=args.system_a,
            system_b=args.system_b,
            output_dir=args.output_dir,
            max_items=args.max_items,
            seed=args.seed,
            base_model=args.base_model,
            model_revision=args.model_revision,
            max_new_tokens=args.max_new_tokens,
            require_cuda=not args.allow_cpu,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "collect":
        events = collect_blind_pairwise_feedback(
            args.queue,
            output_path=args.output,
            annotator_id=args.annotator_id,
        )
        print(json.dumps({"status": "completed", "ratings": len(events)}, ensure_ascii=False))
    elif args.command == "collect-model":
        client = OpenAICompatibleGenerator(args.endpoint, JUDGE_MODEL,
                                          api_key=os.environ.get(args.api_key_env),
                                          timeout_seconds=args.timeout_seconds)
        events = collect_model_pairwise(args.queue, output_path=args.output,
                                        generator=client, max_tokens=args.max_tokens)
        print(json.dumps({"status": "completed", "ratings": len(events)}))
    else:
        report = analyze_blind_pairwise_feedback(
            [row for path in args.feedback for row in _read_jsonl(Path(path))],
            _read_jsonl(Path(args.key)),
            seed=args.seed,
            expected_source="final_test_blind_model" if args.evaluation_mode == "model" else "final_test_blind_human",
        )
        if args.evaluation_mode == "model":
            report["interpretation"] = "Qwen3-32B automatic evaluation; same model family used for feedback. Not human preference evidence."
            report["order_disagreements"] = sum(not row.get("order_consistent", False) for path in args.feedback for row in _read_jsonl(Path(path)))
            report.pop("inter_annotator_agreement", None)
            report["model_evaluation_seconds"] = report.pop("feedback_seconds")
            attempts = [row for path in args.feedback for row in _read_jsonl(Path(path).with_suffix(".attempts.jsonl"))]
            report["judge_cost"] = {"attempts": len(attempts), "failed_attempts": sum(row["status"] == "error" for row in attempts),
                                    "prompt_tokens": sum(row["prompt_tokens"] for row in attempts),
                                    "completion_tokens": sum(row["completion_tokens"] for row in attempts)}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
