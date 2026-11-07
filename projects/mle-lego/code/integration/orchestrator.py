from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion
from core_system.mle_core.store import ArtifactStore
from technologies.technology_one.candidate_gate import SafetyChecklist
from technologies.technology_three.version_selection import (
    PairedMeanRetentionPolicy,
    SelectionAction,
    WindowedVersionSelector,
)
from technologies.technology_two.experience_lifecycle import ExperienceRecord

from .experiment_matrix import IntegrationFeatures, validate_feature_config
from .runtime import DataSystemEvolutionRuntime


@dataclass(frozen=True)
class BatchResult:
    batch_index: int
    stable_score: float
    candidate_score: float | None = None
    candidate_failed: bool = False
    active_score: float | None = None
    feedback_version_id: str | None = None
    experience: ExperienceRecord | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationRunConfig:
    run_id: str
    max_batches: int
    seed: int = 17
    observation_batches: int = 3
    min_mean_gain: float = 0.0
    max_regression: float = 0.05

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.max_batches < 1 or self.observation_batches < 1:
            raise ValueError("batch counts must be positive")


@dataclass(frozen=True)
class IntegrationRunResult:
    run_id: str
    status: str
    features: IntegrationFeatures
    completed_batches: int
    active_version_id: str
    promoted_versions: tuple[str, ...]
    rollback_count: int
    candidate_decision: str | None
    batch_history: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "features": self.features.to_dict(),
            "completed_batches": self.completed_batches,
            "active_version_id": self.active_version_id,
            "promoted_versions": list(self.promoted_versions),
            "rollback_count": self.rollback_count,
            "candidate_decision": self.candidate_decision,
            "batch_history": list(self.batch_history),
        }


BatchExecutor = Callable[[int, str, IntegrationFeatures], BatchResult]


class IntegrationOrchestrator:
    """Executable cross-technology loop for smoke and controlled integration runs.

    The executor supplies task-specific stable/candidate scores and data events.
    The orchestrator owns the shared version ledger, experience lifecycle,
    candidate observation window, promotion, and rollback events.
    """

    def __init__(
        self,
        *,
        runtime: DataSystemEvolutionRuntime,
        features: IntegrationFeatures,
        config: IntegrationRunConfig,
        candidate: CandidateVersion | None = None,
        checklist: SafetyChecklist | None = None,
    ) -> None:
        validate_feature_config(features)
        self.runtime = runtime
        self.features = features
        self.config = config
        self.candidate = candidate
        self.checklist = checklist or SafetyChecklist(True, True, True, True, True, True)
        self.history: list[dict[str, object]] = []
        self.rollback_count = 0
        self.candidate_decision: str | None = None
        self._candidate_started = False
        self._promoted_versions: list[str] = []

    def _start_candidate(self) -> None:
        if not self.features.candidate_isolation or self.candidate is None:
            return
        result = self.runtime.start_candidate(self.candidate, checklist=self.checklist)
        self._candidate_started = result.entered_observation
        self.history.append(
            {
                "event": "candidate_started",
                "candidate_id": self.candidate.candidate_id,
                "status": result.assessment.status.value,
                "failed_checks": list(result.assessment.failed_checks),
            }
        )

    def _record_experience(self, result: BatchResult) -> None:
        if result.experience is None:
            return
        if not self.features.experience_governance:
            return
        if result.experience.experience_id not in self.runtime.experiences.records:
            self.runtime.record_experience(result.experience)
            self.runtime.experiences.validate(result.experience.experience_id)

    def _maybe_decide_candidate(self) -> None:
        if not self._candidate_started or self.candidate_decision is not None:
            return
        if self.runtime.selector.observed_batch_count(self.candidate.candidate_id) < self.config.observation_batches:
            return
        decision = self.runtime.decide_candidate(self.candidate.candidate_id)
        self.candidate_decision = decision.action.value
        if decision.action == SelectionAction.ACCEPT_CANDIDATE:
            self._promoted_versions.append(decision.candidate_version_id)
        self.history.append(
            {
                "event": "candidate_decided",
                "action": decision.action.value,
                "reason": decision.reason,
                "observed_batches": decision.observed_batches,
                "mean_gain": decision.mean_gain,
            }
        )

    def run(self, executor: BatchExecutor) -> IntegrationRunResult:
        self._start_candidate()
        try:
            for batch_index in range(self.config.max_batches):
                active_version = self.runtime.stable_version.version_id
                result = executor(batch_index, active_version, self.features)
                if result.batch_index != batch_index:
                    raise ValueError("executor returned a non-sequential batch index")
                if result.stable_score < 0 or result.stable_score > 1:
                    raise ValueError("stable_score must be between 0 and 1")

                self._record_experience(result)
                if self._candidate_started and self.candidate_decision is None:
                    self.runtime.observe_candidate_batch(
                        self.candidate.candidate_id,
                        batch_id=f"{self.config.run_id}-batch-{batch_index:04d}",
                        stable_score=result.stable_score,
                        candidate_score=result.candidate_score,
                        candidate_failed=result.candidate_failed,
                    )
                    self._maybe_decide_candidate()

                if (
                    self.features.delayed_feedback_selection
                    and result.feedback_version_id is not None
                ):
                    self.history.append(
                        {
                            "event": "feedback_attributed",
                            "batch_index": batch_index,
                            "target_version_id": result.feedback_version_id,
                            "current_version_id": self.runtime.stable_version.version_id,
                        }
                    )

                active_score = result.active_score
                if active_score is not None and active_score < 0:
                    raise ValueError("active_score cannot be negative")
                if (
                    active_score is not None
                    and self.runtime.stable_version.version_id != self.runtime.stable_version_history[0]
                    and active_score < result.stable_score - self.config.max_regression
                ):
                    previous = self.runtime.stable_version_history[-2]
                    self.runtime.restore_stable_version(
                        previous,
                        reason=f"batch {batch_index} exceeded regression threshold",
                    )
                    self.rollback_count += 1
                    self.history.append(
                        {
                            "event": "rollback",
                            "batch_index": batch_index,
                            "restored_version_id": previous,
                        }
                    )

                self.history.append(
                    {
                        "event": "batch_completed",
                        "batch_index": batch_index,
                        "stable_version_id": self.runtime.stable_version.version_id,
                        "stable_score": result.stable_score,
                        "candidate_score": result.candidate_score,
                        "candidate_failed": result.candidate_failed,
                        "active_score": active_score,
                        **result.metadata,
                    }
                )
        except Exception as exc:
            self.history.append({"event": "run_failed", "error": repr(exc)})
            status = "failed"
        else:
            status = "complete"
        return IntegrationRunResult(
            run_id=self.config.run_id,
            status=status,
            features=self.features,
            completed_batches=sum(item.get("event") == "batch_completed" for item in self.history),
            active_version_id=self.runtime.stable_version.version_id,
            promoted_versions=tuple(self._promoted_versions),
            rollback_count=self.rollback_count,
            candidate_decision=self.candidate_decision,
            batch_history=tuple(self.history),
        )


def write_result(result: IntegrationRunResult, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
