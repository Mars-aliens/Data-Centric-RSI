from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from core_system.mle_core.models import TaskRecord

from .system import QualityAssessment, QualityScorer


@dataclass
class QualityEvaluatorState:
    """Persisted notes used to revise future evaluator prompts."""

    prompt_version: int = 1
    confirmed_notes: list[str] = field(default_factory=list)
    audit_count: int = 0
    disagreement_count: int = 0

    def record_audit(
        self,
        record: TaskRecord,
        proposed_decision: str,
        human_accept: bool,
    ) -> None:
        human_decision = "accept" if human_accept else "reject"
        self.audit_count += 1
        if proposed_decision != human_decision:
            self.disagreement_count += 1
            self.confirmed_notes.append(
                f"Audit disagreement on {record.record_id}: "
                f"automatic={proposed_decision}, human={human_decision}."
            )
            self.prompt_version += 1

    def prompt_context(self) -> str:
        if not self.confirmed_notes:
            return "No confirmed evaluator corrections are available."
        return "Confirmed evaluator corrections:\n" + "\n".join(
            f"- {note}" for note in self.confirmed_notes[-8:]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_version": self.prompt_version,
            "confirmed_notes": list(self.confirmed_notes),
            "audit_count": self.audit_count,
            "disagreement_count": self.disagreement_count,
        }


class EnsembleQualityScorer:
    """Conservative ensemble: disagreement lowers confidence and never auto-accepts."""

    def __init__(self, scorers: Sequence[QualityScorer]):
        self.scorers = tuple(scorers)
        if not self.scorers:
            raise ValueError("at least one quality scorer is required")

    def assess(self, record: TaskRecord) -> QualityAssessment:
        assessments = [scorer.assess(record) for scorer in self.scorers]
        scores = [assessment.score for assessment in assessments]
        confidences = [assessment.confidence for assessment in assessments]
        score = sum(scores) / len(scores)
        agreement = max(0.0, 1.0 - (max(scores) - min(scores)))
        confidence = min(confidences) * agreement
        reasons = "; ".join(
            assessment.reason for assessment in assessments if assessment.reason
        )
        return QualityAssessment(
            score=score,
            confidence=confidence,
            reason=f"ensemble agreement={agreement:.3f}; {reasons}".strip("; "),
        )

    @property
    def budget(self):
        from core_system.mle_core.models import Budget

        total = Budget()
        for scorer in self.scorers:
            total = total.add(getattr(scorer, "budget", Budget()))
        return total

    def set_prompt_context(self, value: str) -> None:
        for scorer in self.scorers:
            if hasattr(scorer, "set_prompt_context"):
                scorer.set_prompt_context(value)
