from __future__ import annotations

from dataclasses import dataclass

from core_system.mle_core.evolution import (
    CandidateVersion,
    DataSystemVersion,
    FeedbackEvent,
)
from core_system.mle_core.store import ArtifactStore
from technologies.technology_one.candidate_gate import (
    CandidateAssessment,
    CandidateSafetyGate,
    SafetyChecklist,
)
from technologies.technology_three.version_selection import (
    SelectionAction,
    SelectionDecision,
    WindowedVersionSelector,
)
from technologies.technology_two.experience_lifecycle import (
    ExperienceLifecycleStore,
    ExperienceRecord,
)


@dataclass(frozen=True)
class CandidateStart:
    assessment: CandidateAssessment
    entered_observation: bool


class DataSystemEvolutionRuntime:
    """Small integration boundary for T1 safety, T2 experience, and T3 retention."""

    def __init__(
        self,
        *,
        stable_version: DataSystemVersion,
        store: ArtifactStore,
        experience_failure_limit: int = 2,
        selector: WindowedVersionSelector | None = None,
    ):
        self.stable_version = stable_version
        self.store = store
        self.candidate_gate = CandidateSafetyGate(store)
        self.experiences = ExperienceLifecycleStore(
            store, failure_limit=experience_failure_limit
        )
        self.selector = selector or WindowedVersionSelector(store)
        events = store.read_events()
        if not any(event.get("event_type") == "data_system_runtime_initialized" for event in events):
            store.append_event(
                "data_system_runtime_initialized",
                {"stable_version": stable_version.to_dict()},
                event_id=f"runtime-initialized:{stable_version.version_id}",
            )
        self._restore_runtime_state(stable_version)

    def _restore_runtime_state(self, fallback: DataSystemVersion) -> None:
        self.version_catalog: dict[str, DataSystemVersion] = {}
        self.stable_version_history: list[str] = []
        self.known_version_ids: set[str] = set()
        self.candidate_version_ids: set[str] = set()
        active_version = fallback

        for event in self.store.read_events():
            payload = event.get("payload", {})
            event_type = event.get("event_type")
            if event_type == "data_system_runtime_initialized":
                version = DataSystemVersion.from_dict(payload["stable_version"])
                self.version_catalog[version.version_id] = version
                self.known_version_ids.add(version.version_id)
                if not self.stable_version_history:
                    active_version = version
                    self.stable_version_history.append(version.version_id)
            elif event_type == "candidate_safety_assessed":
                manifest = payload.get("candidate_version_manifest")
                if isinstance(manifest, dict):
                    version = DataSystemVersion.from_dict(manifest)
                    self.version_catalog[version.version_id] = version
                    if payload.get("status") == "eligible_for_observation":
                        self.known_version_ids.add(version.version_id)
                        self.candidate_version_ids.add(version.version_id)
            elif event_type == "data_system_version_promoted":
                manifest = payload.get("new_stable_version")
                if isinstance(manifest, dict):
                    version = DataSystemVersion.from_dict(manifest)
                    self.version_catalog[version.version_id] = version
                    self.known_version_ids.add(version.version_id)
                    self.candidate_version_ids.discard(version.version_id)
                    self.stable_version_history.append(version.version_id)
                    active_version = version
            elif event_type == "data_system_stable_version_retained":
                action = payload.get("decision")
                if action in {"keep_stable", "reject_failed_candidate"}:
                    candidate = payload.get("candidate_version_id")
                    if candidate:
                        self.candidate_version_ids.discard(candidate)
            elif event_type == "data_system_version_restored":
                restored_id = payload.get("restored_version_id")
                if restored_id in self.version_catalog:
                    active_version = self.version_catalog[restored_id]

        if not self.stable_version_history:
            self.version_catalog[fallback.version_id] = fallback
            self.known_version_ids.add(fallback.version_id)
            self.stable_version_history.append(fallback.version_id)
        self.stable_version = active_version

    def retrieve_experiences(self, conditions: dict[str, object]) -> list[ExperienceRecord]:
        return self.experiences.retrieve(conditions)

    def record_experience(self, experience: ExperienceRecord) -> None:
        self.experiences.add(experience)

    def attribute_feedback(self, feedback: FeedbackEvent) -> str:
        attribution = self.selector.attribute_feedback(
            feedback,
            known_version_ids=self.known_version_ids,
            candidate_version_ids=self.candidate_version_ids,
            current_version_id=self.stable_version.version_id,
        )
        return attribution

    def start_candidate(
        self,
        candidate: CandidateVersion,
        *,
        checklist: SafetyChecklist,
    ) -> CandidateStart:
        assessment = self.candidate_gate.assess(
            stable_version=self.stable_version,
            candidate=candidate,
            checklist=checklist,
        )
        self.version_catalog[candidate.version.version_id] = candidate.version
        if assessment.eligible:
            self.known_version_ids.add(candidate.version.version_id)
            self.candidate_version_ids.add(candidate.version.version_id)
            self.selector.register_candidate(assessment)
        return CandidateStart(assessment, assessment.eligible)

    def observe_candidate_batch(
        self,
        candidate_id: str,
        *,
        batch_id: str,
        stable_score: float | None,
        candidate_score: float | None,
        candidate_failed: bool = False,
    ) -> None:
        self.selector.record_batch(
            candidate_id,
            batch_id=batch_id,
            stable_score=stable_score,
            candidate_score=candidate_score,
            candidate_failed=candidate_failed,
        )

    def decide_candidate(self, candidate_id: str) -> SelectionDecision:
        decision = self.selector.decide(
            candidate_id,
            current_stable_version_id=self.stable_version.version_id,
        )
        if decision.action == SelectionAction.ACCEPT_CANDIDATE:
            previous = self.stable_version
            self.stable_version = self.selector.candidate(candidate_id).version
            self.stable_version_history.append(self.stable_version.version_id)
            self.store.append_event(
                "data_system_version_promoted",
                {
                    "previous_stable_version_id": previous.version_id,
                    "new_stable_version_id": self.stable_version.version_id,
                    "new_stable_version": self.stable_version.to_dict(),
                    "candidate_id": candidate_id,
                },
            )
        else:
            if decision.action in {
                SelectionAction.KEEP_STABLE,
                SelectionAction.REJECT_FAILED_CANDIDATE,
            }:
                self.candidate_version_ids.discard(decision.candidate_version_id)
            self.store.append_event(
                "data_system_stable_version_retained",
                {
                    "stable_version_id": self.stable_version.version_id,
                    "candidate_id": candidate_id,
                    "candidate_version_id": decision.candidate_version_id,
                    "decision": decision.action.value,
                },
            )
        return decision

    def restore_stable_version(self, version_id: str, *, reason: str) -> DataSystemVersion:
        """Restore a previously promoted stable version after a later regression."""
        if version_id not in self.stable_version_history:
            raise ValueError("only a previously promoted stable version can be restored")
        if not reason.strip():
            raise ValueError("a restoration reason is required")
        if version_id == self.stable_version.version_id:
            return self.stable_version
        previous = self.stable_version
        self.stable_version = self.version_catalog[version_id]
        self.store.append_event(
            "data_system_version_restored",
            {
                "previous_version_id": previous.version_id,
                "restored_version_id": version_id,
                "restored_version": self.stable_version.to_dict(),
                "reason": reason,
            },
        )
        return self.stable_version
