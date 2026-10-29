from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion
from core_system.mle_core.store import ArtifactStore


class CandidateGateStatus(str, Enum):
    ELIGIBLE_FOR_OBSERVATION = "eligible_for_observation"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SafetyChecklist:
    candidate_branch_isolated: bool
    executable: bool
    data_valid: bool
    interfaces_valid: bool
    within_budget: bool
    state_isolated: bool

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "candidate_branch_isolated",
                "executable",
                "data_valid",
                "interfaces_valid",
                "within_budget",
                "state_isolated",
            )
            if not getattr(self, name)
        )

    @property
    def passed(self) -> bool:
        return not self.failed_checks


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: CandidateVersion
    status: CandidateGateStatus
    failed_checks: tuple[str, ...]
    evidence: dict[str, Any]

    @property
    def eligible(self) -> bool:
        return self.status == CandidateGateStatus.ELIGIBLE_FOR_OBSERVATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_version": self.candidate.version.version_id,
            "candidate_version_manifest": self.candidate.version.to_dict(),
            "parent_version": self.candidate.version.parent_version_id,
            "status": self.status.value,
            "failed_checks": list(self.failed_checks),
            "evidence": self.evidence,
        }


class CandidateSafetyGate:
    """T1 safety gate. Passing means observe the candidate, not promote it."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    def assess(
        self,
        *,
        stable_version: DataSystemVersion,
        candidate: CandidateVersion,
        checklist: SafetyChecklist,
        evidence: Mapping[str, Any] | None = None,
    ) -> CandidateAssessment:
        if candidate.version.version_id == stable_version.version_id:
            raise ValueError("candidate and stable version must differ")
        if candidate.version.parent_version_id != stable_version.version_id:
            raise ValueError("candidate must be based on the current stable version")

        status = (
            CandidateGateStatus.ELIGIBLE_FOR_OBSERVATION
            if checklist.passed
            else CandidateGateStatus.REJECTED
        )
        evidence_value = dict(evidence or {})
        assessment = CandidateAssessment(
            candidate, status, checklist.failed_checks, evidence_value
        )
        self.store.put_json(
            kind="data_system_candidate",
            value=candidate.version.to_dict(),
            version=candidate.version.version_id,
            parents=[stable_version.version_id],
        )
        self.store.append_event(
            "candidate_safety_assessed",
            {
                **assessment.to_dict(),
                "checklist": {
                    "candidate_branch_isolated": checklist.candidate_branch_isolated,
                    "executable": checklist.executable,
                    "data_valid": checklist.data_valid,
                    "interfaces_valid": checklist.interfaces_valid,
                    "within_budget": checklist.within_budget,
                    "state_isolated": checklist.state_isolated,
                },
                "evidence": evidence_value,
            },
            event_id=f"{candidate.created_run_id}:{candidate.created_batch_id}:safety",
        )
        return assessment
