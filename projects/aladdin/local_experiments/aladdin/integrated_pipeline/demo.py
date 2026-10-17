from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from core_system.mle_core.local_data import build_local_text_registry
from core_system.mle_core.models import Budget, TaskRecord

from .calibration import ReliabilityCalibrator
from .system import (
    FeedbackQuery,
    HumanFeedback,
    QualityAssessment,
    RequirementHypothesis,
    RequirementState,
    SynthesisBudgetAllocator,
)
from .recursive import run_recursive
from .strategies import strategy_for


class FixtureQualityAuditor:
    """Separate fixture channel; never reuses training feedback."""

    def audit(self, record: TaskRecord, proposed_decision: str) -> bool | None:
        del record
        return proposed_decision == "accept"


class FixtureRequirementScorer:
    """Explicit vote fixture. It does not interpret natural-language text."""

    def score(self, hypothesis: RequirementHypothesis, record: TaskRecord) -> float:
        votes = record.metadata.get("fixture_profile_votes", {})
        return float(votes.get(hypothesis.hypothesis_id, 0.5))


class FixtureQualityScorer:
    def assess(self, record: TaskRecord) -> QualityAssessment:
        return QualityAssessment(
            score=float(record.metadata.get("fixture_quality", 0.8)),
            confidence=float(record.metadata.get("fixture_confidence", 0.9)),
            reason="fixture only",
        )


class FixtureGenerator:
    """Deterministic templates for control-flow smoke tests, not synthesis."""

    def __init__(self, candidates: Sequence[TaskRecord]):
        self.candidates = tuple(candidates)

    def generate(
        self,
        *,
        seed_records: Sequence[TaskRecord],
        requirements: Sequence[RequirementHypothesis],
        confirmed_edits: Sequence[str],
        strategy: str,
        count: int,
        round_index: int,
        cost_budget: float | None,
    ) -> tuple[list[TaskRecord], Budget]:
        del requirements, confirmed_edits, cost_budget
        output = []
        for record in self.candidates[:count]:
            parent_id = record.parent_record_ids[0]
            if parent_id not in {item.record_id for item in seed_records}:
                raise ValueError("fixture candidate refers to a missing seed parent")
            output.append(
                TaskRecord(
                    record_id=f"r{round_index}-{record.record_id}",
                    task_id=record.task_id,
                    split="seed",
                    input=(
                        record.input
                        if strategy == "coverage"
                        else (
                            f"{record.input} with a clearer, requirement-focused response"
                            if strategy == "revision"
                            else f"{record.input} with a close contrastive alternative"
                        )
                    ),
                    target=record.target,
                    source_group_id=record.source_group_id,
                    metadata={
                        **record.metadata,
                        "generator_strategy": strategy,
                        "strategy_objective": strategy_for(strategy).objective,
                    },
                    parent_record_ids=record.parent_record_ids,
                )
            )
        return output, Budget()


class FixtureReviewer:
    """Pre-scripted reviewer used only to exercise the synchronous demo."""

    def __init__(self, responses: dict[str, HumanFeedback]):
        self.responses = responses

    def review(self, record: TaskRecord, query: FeedbackQuery) -> HumanFeedback | None:
        del query
        return self.responses.get(record.record_id.split("-", 1)[-1])


def _candidate(
    record_id: str,
    text: str,
    parent: str,
    votes: dict[str, float],
    *,
    quality: float = 0.85,
    confidence: float = 0.9,
    review_cost: float = 1.0,
) -> TaskRecord:
    return TaskRecord(
        record_id=record_id,
        task_id="local-intent",
        split="seed",
        input=text,
        target=None,
        source_group_id=f"generated-{record_id}",
        metadata={
            "fixture_profile_votes": votes,
            "fixture_quality": quality,
            "fixture_confidence": confidence,
            "review_cost": review_cost,
        },
        parent_record_ids=(parent,),
    )


def run_demo(output_dir: Path, rounds: int = 1) -> dict[str, object]:
    registry = build_local_text_registry()
    seed = registry.records_for("local-intent", split="seed")
    development = registry.records_for("local-intent", split="development")
    requirements = [
        RequirementHypothesis("account", "Prioritize account access and sign-in cases."),
        RequirementHypothesis("billing", "Prioritize invoices, receipts, and payment cases."),
    ]
    state = RequirementState.uniform(requirements)
    candidates = [
        _candidate(
            "candidate-portal",
            "I cannot enter the customer account portal",
            "seed-001",
            {"account": 0.95, "billing": 0.05},
            review_cost=8.0,
        ),
        _candidate(
            "candidate-login",
            "my sign-in stopped working and the account is locked",
            "seed-002",
            {"account": 0.95, "billing": 0.05},
            quality=0.45,
            confidence=0.45,
            review_cost=10.0,
        ),
        _candidate(
            "candidate-invoice",
            "I need the invoice for a recent payment",
            "seed-003",
            {"account": 0.05, "billing": 0.95},
            review_cost=7.0,
        ),
        _candidate(
            "candidate-receipt",
            "please send a receipt for my order payment",
            "seed-004",
            {"account": 0.05, "billing": 0.95},
            review_cost=7.0,
        ),
        _candidate(
            "candidate-noise",
            "unreadable unrelated placeholder content",
            "seed-001",
            {"account": 0.5, "billing": 0.5},
            quality=0.1,
            confidence=0.98,
        ),
    ]
    reviewer = FixtureReviewer(
        {
            "candidate-portal": HumanFeedback(
                True,
                "account",
                "Include customer-portal lockouts in account-access coverage.",
                8.0,
            ),
            "candidate-login": HumanFeedback(True, "account", review_seconds=10.0),
            "candidate-invoice": HumanFeedback(True, "billing", review_seconds=7.0),
            "candidate-receipt": HumanFeedback(True, "billing", review_seconds=7.0),
        }
    )
    allocator = SynthesisBudgetAllocator(("coverage", "revision", "contrastive"))
    calibrator = ReliabilityCalibrator()
    calibrator.add_audits("local-intent", "accept", [True] * 25)
    calibrator.add_audits("local-intent", "reject", [False] * 25)
    results = run_recursive(
        run_id="integrated-pipeline-smoke",
        rounds=rounds,
        seed_records=seed,
        development_records=development,
        requirement_state=state,
        requirement_scorer=FixtureRequirementScorer(),
        quality_scorer=FixtureQualityScorer(),
        quality_calibrator=calibrator,
        quality_auditor=FixtureQualityAuditor(),
        generator=FixtureGenerator(candidates),
        reviewer=reviewer,
        allocator=allocator,
        output_dir=output_dir,
        candidate_count=5,
        feedback_budget=4,
        feedback_cost_budget=32.0,
        generation_cost_budget=5.0,
        selected_data_limit=3,
        random_feedback_fraction=0.34,
        round_run_id_suffix=False,
    )
    result = results[-1]
    return {
        **asdict(result),
        "recursive_rounds": [asdict(item) for item in results],
        "requirement_state_after_round": {
            "weights": state.weights,
            "confirmed_edits": state.confirmed_edits,
        },
        "allocator_after_round": {
            "counts": allocator.counts,
            "reward_per_cost": allocator.reward_per_cost,
        },
        "fixture_warning": (
            "Scores, generated samples, and feedback are deterministic fixtures. "
            "This run checks plumbing only and is not model or method evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the integrated Aladdin local smoke test.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    report = run_demo(args.output_dir, rounds=args.rounds)
    report_path = args.output_dir / "smoke-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
