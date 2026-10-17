from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from core_system.mle_core.models import TaskRecord


AUTO_DECISIONS = ("accept", "reject")


@dataclass
class CalibrationEvidence:
    audits: int = 0
    agreements: int = 0

    @property
    def observed_agreement(self) -> float | None:
        if self.audits == 0:
            return None
        return self.agreements / self.audits

    @property
    def observed_error_rate(self) -> float | None:
        agreement = self.observed_agreement
        return None if agreement is None else 1.0 - agreement


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must describe valid observations")
    if total == 0:
        return 0.0
    if z <= 0 or not math.isfinite(z):
        raise ValueError("z must be positive and finite")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


class ReliabilityCalibrator:
    """Gate automatic quality actions using independent human audits."""

    def __init__(
        self,
        *,
        minimum_audits: int = 20,
        minimum_agreement_lower_bound: float = 0.8,
        confidence_z: float = 1.96,
    ) -> None:
        if minimum_audits <= 0:
            raise ValueError("minimum_audits must be positive")
        if not 0 < minimum_agreement_lower_bound <= 1:
            raise ValueError("minimum_agreement_lower_bound must be in (0, 1]")
        if confidence_z <= 0 or not math.isfinite(confidence_z):
            raise ValueError("confidence_z must be positive and finite")
        self.minimum_audits = minimum_audits
        self.minimum_agreement_lower_bound = minimum_agreement_lower_bound
        self.confidence_z = confidence_z
        self._evidence: dict[tuple[str, str], CalibrationEvidence] = {}

    @staticmethod
    def slice_for(record: TaskRecord) -> str:
        value = record.metadata.get("quality_slice", record.task_id)
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"Empty quality_slice for {record.record_id}")
        return normalized

    def add_audit(
        self,
        slice_name: str,
        proposed_decision: str,
        human_accept: bool,
    ) -> None:
        if proposed_decision not in AUTO_DECISIONS:
            raise ValueError("only accept/reject decisions can be independently audited")
        normalized_slice = slice_name.strip()
        if not normalized_slice:
            raise ValueError("slice_name cannot be empty")
        evidence = self._evidence.setdefault(
            (normalized_slice, proposed_decision), CalibrationEvidence()
        )
        human_decision = "accept" if human_accept else "reject"
        evidence.audits += 1
        evidence.agreements += int(proposed_decision == human_decision)

    def add_audits(
        self,
        slice_name: str,
        proposed_decision: str,
        human_accepts: Sequence[bool],
    ) -> None:
        for human_accept in human_accepts:
            self.add_audit(slice_name, proposed_decision, human_accept)

    def evidence_for(
        self, slice_name: str, proposed_decision: str
    ) -> CalibrationEvidence:
        evidence = self._evidence.get((slice_name, proposed_decision))
        if evidence is None:
            return CalibrationEvidence()
        return CalibrationEvidence(evidence.audits, evidence.agreements)

    def is_certified(self, slice_name: str, proposed_decision: str) -> bool:
        if proposed_decision not in AUTO_DECISIONS:
            return False
        evidence = self.evidence_for(slice_name, proposed_decision)
        return (
            evidence.audits >= self.minimum_audits
            and wilson_lower_bound(
                evidence.agreements, evidence.audits, self.confidence_z
            )
            >= self.minimum_agreement_lower_bound
        )

    def route(self, record: TaskRecord, proposed_decision: str) -> str:
        if proposed_decision == "defer":
            return "defer"
        if proposed_decision not in AUTO_DECISIONS:
            raise ValueError(f"Unknown quality decision: {proposed_decision}")
        if self.is_certified(self.slice_for(record), proposed_decision):
            return proposed_decision
        return "defer"

    def summary(self) -> dict[str, object]:
        slices: dict[str, dict[str, object]] = {}
        for (slice_name, decision), evidence in sorted(self._evidence.items()):
            slice_row = slices.setdefault(slice_name, {})
            slice_row[decision] = {
                "audits": evidence.audits,
                "agreements": evidence.agreements,
                "observed_agreement": evidence.observed_agreement,
                "observed_error_rate": evidence.observed_error_rate,
                "agreement_lower_bound": wilson_lower_bound(
                    evidence.agreements, evidence.audits, self.confidence_z
                ),
                "certified": self.is_certified(slice_name, decision),
            }
        return {
            "minimum_audits": self.minimum_audits,
            "minimum_agreement_lower_bound": self.minimum_agreement_lower_bound,
            "confidence_z": self.confidence_z,
            "slices": slices,
        }

    def to_dict(self) -> dict[str, object]:
        audits = []
        for (slice_name, decision), evidence in sorted(self._evidence.items()):
            # Store aggregate evidence; raw audit identities belong in the artifact ledger.
            audits.append(
                {
                    "slice": slice_name,
                    "proposed_decision": decision,
                    "audits": evidence.audits,
                    "agreements": evidence.agreements,
                }
            )
        return {
            "minimum_audits": self.minimum_audits,
            "minimum_agreement_lower_bound": self.minimum_agreement_lower_bound,
            "confidence_z": self.confidence_z,
            "audits": audits,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ReliabilityCalibrator":
        calibrator = cls(
            minimum_audits=int(value.get("minimum_audits", 20)),
            minimum_agreement_lower_bound=float(
                value.get("minimum_agreement_lower_bound", 0.8)
            ),
            confidence_z=float(value.get("confidence_z", 1.96)),
        )
        for row in value.get("audits", []):
            if "audits" not in row or "agreements" not in row:
                raise ValueError("aggregate calibration rows need audits and agreements")
            evidence = CalibrationEvidence(
                audits=int(row["audits"]), agreements=int(row["agreements"])
            )
            if evidence.agreements > evidence.audits or evidence.audits < 0:
                raise ValueError("invalid aggregate calibration evidence")
            calibrator._evidence[(str(row["slice"]), str(row["proposed_decision"]))] = evidence
        return calibrator
