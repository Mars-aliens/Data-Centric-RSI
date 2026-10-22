from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Protocol

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion, FeedbackEvent
from core_system.mle_core.store import ArtifactStore
from technologies.technology_one.candidate_gate import CandidateAssessment


class SelectionAction(str, Enum):
    CONTINUE_OBSERVING = "continue_observing"
    ACCEPT_CANDIDATE = "accept_candidate"
    KEEP_STABLE = "keep_stable"
    REJECT_FAILED_CANDIDATE = "reject_failed_candidate"


@dataclass(frozen=True)
class BatchComparison:
    batch_id: str
    stable_score: float | None
    candidate_score: float | None
    candidate_failed: bool = False


@dataclass(frozen=True)
class SelectionDecision:
    candidate_id: str
    action: SelectionAction
    stable_version_id: str
    candidate_version_id: str
    observed_batches: int
    mean_gain: float | None
    reason: str


@dataclass
class _CandidateWindow:
    candidate: CandidateVersion
    base_version_id: str
    comparisons: list[BatchComparison] = field(default_factory=list)


class RetentionPolicy(Protocol):
    def decide(self, comparisons: tuple[BatchComparison, ...]) -> tuple[SelectionAction, str, float | None]: ...


@dataclass(frozen=True)
class PairedMeanRetentionPolicy:
    """Replaceable baseline policy; the research selection rule remains configurable."""

    min_batches: int = 3
    min_mean_gain: float = 0.0
    max_regression: float = 0.0
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.min_batches < 1:
            raise ValueError("min_batches must be positive")
        if self.min_mean_gain < 0 or self.max_regression < 0:
            raise ValueError("gain and regression thresholds cannot be negative")

    def decide(
        self, comparisons: tuple[BatchComparison, ...]
    ) -> tuple[SelectionAction, str, float | None]:
        if any(item.candidate_failed for item in comparisons):
            return (
                SelectionAction.REJECT_FAILED_CANDIDATE,
                "candidate_failed_during_observation",
                None,
            )
        complete = tuple(
            item
            for item in comparisons
            if item.stable_score is not None and item.candidate_score is not None
        )
        if len(complete) < self.min_batches:
            return (
                SelectionAction.CONTINUE_OBSERVING,
                "insufficient_paired_batches",
                None,
            )
        direction = 1.0 if self.higher_is_better else -1.0
        gains = tuple(
            direction * (item.candidate_score - item.stable_score)
            for item in complete
        )
        mean_gain = fmean(gains)
        if mean_gain > self.min_mean_gain and min(gains) >= -self.max_regression:
            return (
                SelectionAction.ACCEPT_CANDIDATE,
                "paired_window_passed_provisional_policy",
                mean_gain,
            )
        return (
            SelectionAction.KEEP_STABLE,
            "paired_window_did_not_pass_provisional_policy",
            mean_gain,
        )


class WindowedVersionSelector:
    """T3 keeps the stable version active until a multi-batch decision is made."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        policy: RetentionPolicy | None = None,
    ):
        self.store = store
        self.policy = policy or PairedMeanRetentionPolicy()
        self._windows: dict[str, _CandidateWindow] = {}
        self._restore_from_events()

    def register_candidate(self, assessment: CandidateAssessment) -> None:
        if not assessment.eligible:
            raise ValueError("only safety-approved candidates can enter observation")
        candidate = assessment.candidate
        if candidate.candidate_id in self._windows:
            raise ValueError(f"candidate already registered: {candidate.candidate_id}")
        base_version = candidate.version.parent_version_id
        if base_version is None:
            raise ValueError("candidate must identify its parent stable version")
        self._windows[candidate.candidate_id] = _CandidateWindow(candidate, base_version)
        self.store.append_event(
            "candidate_observation_started",
            {
                "candidate_id": candidate.candidate_id,
                "candidate_version_id": candidate.version.version_id,
                "candidate_version": candidate.version.to_dict(),
                "base_version_id": base_version,
                "created_run_id": candidate.created_run_id,
                "created_batch_id": candidate.created_batch_id,
            },
        )

    def record_batch(
        self,
        candidate_id: str,
        *,
        batch_id: str,
        stable_score: float | None,
        candidate_score: float | None,
        candidate_failed: bool = False,
    ) -> None:
        window = self._get_window(candidate_id)
        if any(item.batch_id == batch_id for item in window.comparisons):
            raise ValueError(f"duplicate observation batch: {batch_id}")
        if candidate_failed:
            stable_score = candidate_score = None
        elif stable_score is None or candidate_score is None:
            raise ValueError("paired scores are required unless the candidate failed")
        elif not all(math.isfinite(value) for value in (stable_score, candidate_score)):
            raise ValueError("candidate comparison scores must be finite")
        comparison = BatchComparison(
            batch_id, stable_score, candidate_score, candidate_failed
        )
        window.comparisons.append(comparison)
        self.store.append_event(
            "candidate_batch_observed",
            {
                "candidate_id": candidate_id,
                "candidate_version_id": window.candidate.version.version_id,
                "base_version_id": window.base_version_id,
                "comparison": comparison,
            },
        )

    def decide(self, candidate_id: str, *, current_stable_version_id: str) -> SelectionDecision:
        window = self._get_window(candidate_id)
        if current_stable_version_id != window.base_version_id:
            action = SelectionAction.KEEP_STABLE
            reason = "candidate_base_is_no_longer_current"
            mean_gain = None
        else:
            action, reason, mean_gain = self.policy.decide(tuple(window.comparisons))
        decision = SelectionDecision(
            candidate_id=candidate_id,
            action=action,
            stable_version_id=current_stable_version_id,
            candidate_version_id=window.candidate.version.version_id,
            observed_batches=len(window.comparisons),
            mean_gain=mean_gain,
            reason=reason,
        )
        self.store.append_event("candidate_version_decided", {"decision": decision})
        return decision

    def attribute_feedback(
        self,
        feedback: FeedbackEvent,
        *,
        known_version_ids: set[str],
        candidate_version_ids: set[str] | None = None,
        current_version_id: str,
    ) -> str:
        if feedback.target_version_id == current_version_id:
            attribution = "current_version"
        elif feedback.target_version_id in (candidate_version_ids or set()):
            attribution = "candidate_version"
        elif feedback.target_version_id in known_version_ids:
            attribution = "historical_version"
        else:
            attribution = "unknown_version"
        self.store.append_event(
            "feedback_attributed",
            {
                "feedback": feedback,
                "attribution": attribution,
                "current_version_id": current_version_id,
                "used_for_current_decision": False,
            },
            event_id=f"feedback:{feedback.feedback_id}:attributed",
        )
        return attribution

    def candidate(self, candidate_id: str) -> CandidateVersion:
        return self._get_window(candidate_id).candidate

    def _get_window(self, candidate_id: str) -> _CandidateWindow:
        try:
            return self._windows[candidate_id]
        except KeyError as exc:
            raise KeyError(f"unknown candidate: {candidate_id}") from exc

    def _restore_from_events(self) -> None:
        terminal_candidates: set[str] = set()
        for event in self.store.read_events():
            payload = event.get("payload", {})
            event_type = event.get("event_type")
            if event_type == "candidate_observation_started":
                candidate_id = payload.get("candidate_id")
                manifest = payload.get("candidate_version")
                if candidate_id and isinstance(manifest, dict):
                    candidate = CandidateVersion(
                        candidate_id=candidate_id,
                        version=DataSystemVersion.from_dict(manifest),
                        created_run_id=payload["created_run_id"],
                        created_batch_id=payload["created_batch_id"],
                    )
                    self._windows[candidate_id] = _CandidateWindow(
                        candidate, payload["base_version_id"]
                    )
            elif event_type == "candidate_batch_observed":
                candidate_id = payload.get("candidate_id")
                if candidate_id in self._windows:
                    raw = payload["comparison"]
                    self._windows[candidate_id].comparisons.append(
                        BatchComparison(
                            batch_id=raw["batch_id"],
                            stable_score=raw.get("stable_score"),
                            candidate_score=raw.get("candidate_score"),
                            candidate_failed=raw.get("candidate_failed", False),
                        )
                    )
            elif event_type == "candidate_version_decided":
                decision = payload.get("decision", {})
                candidate_id = decision.get("candidate_id")
                action = decision.get("action")
                if action in {
                    SelectionAction.ACCEPT_CANDIDATE.value,
                    SelectionAction.KEEP_STABLE.value,
                    SelectionAction.REJECT_FAILED_CANDIDATE.value,
                }:
                    terminal_candidates.add(candidate_id)
        for candidate_id in terminal_candidates:
            self._windows.pop(candidate_id, None)
