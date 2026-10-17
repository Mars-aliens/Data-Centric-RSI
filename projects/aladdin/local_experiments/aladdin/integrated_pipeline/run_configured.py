from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import (
    ConsoleFeedbackReviewer,
    LLMCandidateGenerator,
    LLMQualityScorer,
    LLMRequirementScorer,
    MappingFeedbackReviewer,
    MappingQualityAuditor,
    OpenAICompatibleJsonClient,
    OfflineReplayCandidateGenerator,
    OfflineReplayFeedbackReviewer,
    api_key_from_env,
    load_calibrator,
    load_feedback,
    load_quality_audits,
    load_requirement_state,
    load_task_records,
)
from .calibration import ReliabilityCalibrator
from .evaluation import EnsembleQualityScorer, QualityEvaluatorState
from .recursive import load_recursive_state, run_recursive, write_recursive_manifest
from .system import SynthesisBudgetAllocator, run_round


def _client(args: argparse.Namespace) -> OpenAICompatibleJsonClient:
    return OpenAICompatibleJsonClient(
        endpoint=args.endpoint,
        model=args.model,
        api_key=api_key_from_env(args.api_key_env),
        timeout_seconds=args.timeout_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one configured integrated MLE-Aladdin development round."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--feedback-file", type=Path)
    parser.add_argument("--quality-audit-file", type=Path)
    parser.add_argument(
        "--feedback-mode",
        choices=("interactive", "file", "offline-replay"),
        default="interactive",
    )
    parser.add_argument("--interactive-feedback", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--strategies", nargs="+", default=["coverage", "revision", "contrastive"])
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--feedback-budget", type=int, default=4)
    parser.add_argument("--feedback-seconds", type=float, default=120.0)
    parser.add_argument("--generation-tokens", type=int, default=1536)
    parser.add_argument("--seed-example-limit", type=int, default=16)
    parser.add_argument("--seed-example-chars", type=int, default=700)
    parser.add_argument("--selected-data-limit", type=int, default=8)
    parser.add_argument("--random-feedback-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--quality-ensemble-size", type=int, default=1)
    parser.add_argument("--resume-state", type=Path)
    args = parser.parse_args()

    if args.feedback_mode == "offline-replay":
        if args.feedback_file or args.interactive_feedback:
            parser.error("offline-replay cannot be combined with a feedback source")
    elif args.feedback_mode == "file":
        if not args.feedback_file or args.interactive_feedback:
            parser.error("file feedback mode requires --feedback-file only")
    else:
        if args.feedback_file and args.interactive_feedback:
            parser.error("choose either --feedback-file or --interactive-feedback")
        if not args.feedback_file and not args.interactive_feedback:
            parser.error("an explicit human feedback source is required")
        if args.rounds > 1 and not args.interactive_feedback:
            parser.error("multi-round configured runs require --interactive-feedback")
    if args.quality_ensemble_size <= 0:
        parser.error("--quality-ensemble-size must be positive")
    resumable_existing = (
        args.resume_state is not None
        and args.output_dir.resolve() == args.resume_state.resolve().parent
        and args.resume_state.is_file()
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not resumable_existing:
        parser.error("--output-dir must be absent or empty; artifacts are append-only")

    records = load_task_records(args.data)
    seed_records = [item for item in records if item.split == "seed"]
    development_records = [item for item in records if item.split == "development"]
    candidate_pool = [item for item in records if item.split == "held_in"]
    allowed_splits = {"seed", "development"}
    if args.feedback_mode == "offline-replay":
        allowed_splits.add("held_in")
    extra_splits = sorted({item.split for item in records} - allowed_splits)
    if extra_splits:
        parser.error(
            f"configured rounds do not accept these splits; found {extra_splits}"
        )
    if args.feedback_mode == "offline-replay" and not candidate_pool:
        parser.error("offline-replay requires a non-empty held_in candidate pool")
    state = load_requirement_state(args.requirements)
    calibrator = load_calibrator(args.calibration) if args.calibration else ReliabilityCalibrator()
    allocator = SynthesisBudgetAllocator(args.strategies)
    evaluator_state = QualityEvaluatorState()
    start_round = 0
    if args.resume_state:
        restored = load_recursive_state(args.resume_state)
        state = restored["requirement_state"]
        if restored["current_seed_records"]:
            seed_records = restored["current_seed_records"]
        calibrator = restored["quality_calibrator"] or ReliabilityCalibrator()
        allocator = restored["allocator"]
        evaluator_state = restored["evaluator_state"] or QualityEvaluatorState()
        start_round = restored["completed_rounds"]
    if args.feedback_mode == "offline-replay":
        reviewer = OfflineReplayFeedbackReviewer(candidate_pool)
        generator = OfflineReplayCandidateGenerator(candidate_pool)
    else:
        reviewer = (
            ConsoleFeedbackReviewer()
            if args.interactive_feedback
            else MappingFeedbackReviewer(load_feedback(args.feedback_file))
        )
        generator = LLMCandidateGenerator(
            _client(args),
            seed_example_limit=args.seed_example_limit,
            seed_example_chars=args.seed_example_chars,
        )
    quality_auditor = (
        MappingQualityAuditor(load_quality_audits(args.quality_audit_file))
        if args.quality_audit_file
        else None
    )
    requirement_scorer = LLMRequirementScorer(_client(args))
    quality_scorers = [
        LLMQualityScorer(_client(args), rubric_variant=f"prompt-{index + 1}")
        for index in range(args.quality_ensemble_size)
    ]
    quality_scorer = (
        quality_scorers[0]
        if len(quality_scorers) == 1
        else EnsembleQualityScorer(quality_scorers)
    )
    quality_scorer.set_prompt_context(evaluator_state.prompt_context())
    results = run_recursive(
        run_id=args.run_id,
        rounds=args.rounds,
        seed_records=seed_records,
        development_records=development_records,
        requirement_state=state,
        requirement_scorer=requirement_scorer,
        quality_scorer=quality_scorer,
        quality_calibrator=calibrator,
        quality_auditor=quality_auditor,
        generator=generator,
        reviewer=reviewer,
        allocator=allocator,
        output_dir=args.output_dir,
        candidate_count=args.candidate_count,
        feedback_budget=args.feedback_budget,
        feedback_cost_budget=args.feedback_seconds,
        generation_cost_budget=float(args.generation_tokens),
        selected_data_limit=args.selected_data_limit,
        random_feedback_fraction=args.random_feedback_fraction,
        model_seed=args.seed,
        evaluator_state=evaluator_state,
        start_round=start_round,
    )
    write_recursive_manifest(args.output_dir, results)
    print(
        json.dumps(
            {
                "status": "completed",
                "feedback_mode": args.feedback_mode,
                "rounds": [result.__dict__ for result in results],
            },
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
