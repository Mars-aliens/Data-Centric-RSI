from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from core_system.mle_core.store import ArtifactStore


class ExperienceStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    NEEDS_VALIDATION = "needs_validation"
    CONFLICTED = "conflicted"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    kind: str
    content: dict[str, Any]
    conditions: dict[str, Any]
    source_version_id: str
    source_run_id: str
    source_batch_id: str
    status: ExperienceStatus = ExperienceStatus.CANDIDATE
    success_count: int = 0
    failure_count: int = 0
    conflict_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"memory", "skill"}:
            raise ValueError("experience kind must be 'memory' or 'skill'")
        if not all(
            (
                self.experience_id,
                self.source_version_id,
                self.source_run_id,
                self.source_batch_id,
            )
        ):
            raise ValueError("experience provenance fields must be non-empty")
        if self.success_count < 0 or self.failure_count < 0:
            raise ValueError("experience outcome counts cannot be negative")


class ExperienceLifecycleStore:
    """Versioned experience registry with explicit validation and invalidation."""

    def __init__(self, store: ArtifactStore, *, failure_limit: int = 2):
        if failure_limit < 1:
            raise ValueError("failure_limit must be positive")
        self.store = store
        self.failure_limit = failure_limit
        self.records: dict[str, ExperienceRecord] = {}
        self.conflict_groups: dict[str, tuple[str, ...]] = {}
        self._restore_from_events()

    def add(self, record: ExperienceRecord) -> None:
        if record.experience_id in self.records:
            raise ValueError(f"duplicate experience: {record.experience_id}")
        if record.status != ExperienceStatus.CANDIDATE:
            raise ValueError("new experiences must start in candidate state")
        self.records[record.experience_id] = record
        self._emit("experience_added", record)

    def retrieve(self, conditions: dict[str, Any]) -> list[ExperienceRecord]:
        return [
            record
            for record in self.records.values()
            if record.status == ExperienceStatus.VERIFIED
            and all(conditions.get(key) == value for key, value in record.conditions.items())
        ]

    def validate(self, experience_id: str) -> ExperienceRecord:
        record = self._get(experience_id)
        if record.status not in {
            ExperienceStatus.CANDIDATE,
            ExperienceStatus.NEEDS_VALIDATION,
        }:
            raise ValueError(f"cannot validate experience in state {record.status.value}")
        updated = replace(record, status=ExperienceStatus.VERIFIED)
        return self._replace("experience_validated", updated)

    def record_outcome(self, experience_id: str, *, succeeded: bool) -> ExperienceRecord:
        record = self._get(experience_id)
        if record.status != ExperienceStatus.VERIFIED:
            raise ValueError("only verified experience can be reused")
        failures = record.failure_count + (0 if succeeded else 1)
        successes = record.success_count + (1 if succeeded else 0)
        status = (
            ExperienceStatus.NEEDS_VALIDATION
            if failures >= self.failure_limit
            else record.status
        )
        updated = replace(
            record,
            success_count=successes,
            failure_count=failures,
            status=status,
        )
        return self._replace("experience_outcome_recorded", updated)

    def invalidate_for_context_change(self, changed_conditions: set[str]) -> list[str]:
        invalidated: list[str] = []
        for record in tuple(self.records.values()):
            if (
                record.status == ExperienceStatus.VERIFIED
                and changed_conditions.intersection(record.conditions)
            ):
                self._replace(
                    "experience_invalidated",
                    replace(record, status=ExperienceStatus.NEEDS_VALIDATION),
                )
                invalidated.append(record.experience_id)
        return invalidated

    def mark_conflicted(self, experience_ids: list[str], *, conflict_group_id: str) -> None:
        if len(experience_ids) < 2 or len(set(experience_ids)) != len(experience_ids):
            raise ValueError("a conflict group requires at least two distinct experiences")
        records = [self._get(item) for item in experience_ids]
        if any(record.status != ExperienceStatus.VERIFIED for record in records):
            raise ValueError("only verified experiences can be marked conflicted")
        self.conflict_groups[conflict_group_id] = tuple(experience_ids)
        for record in records:
            self._replace(
                "experience_conflict_marked",
                replace(
                    record,
                    status=ExperienceStatus.CONFLICTED,
                    conflict_group_id=conflict_group_id,
                ),
            )

    def resolve_conflict(self, conflict_group_id: str, *, preferred_id: str) -> None:
        members = self.conflict_groups.get(conflict_group_id)
        if members is None or preferred_id not in members:
            raise ValueError("unknown conflict group or preferred experience")
        for experience_id in members:
            record = self._get(experience_id)
            status = (
                ExperienceStatus.NEEDS_VALIDATION
                if experience_id == preferred_id
                else ExperienceStatus.SUSPENDED
            )
            self._replace(
                "experience_conflict_resolved",
                replace(record, status=status, conflict_group_id=None),
            )
        del self.conflict_groups[conflict_group_id]

    def expire(self, experience_id: str) -> ExperienceRecord:
        record = self._get(experience_id)
        if record.status == ExperienceStatus.EXPIRED:
            return record
        return self._replace(
            "experience_expired",
            replace(record, status=ExperienceStatus.EXPIRED, conflict_group_id=None),
        )

    def _get(self, experience_id: str) -> ExperienceRecord:
        try:
            return self.records[experience_id]
        except KeyError as exc:
            raise KeyError(f"unknown experience: {experience_id}") from exc

    def _replace(self, event_type: str, record: ExperienceRecord) -> ExperienceRecord:
        self.records[record.experience_id] = record
        self._emit(event_type, record)
        return record

    def _emit(self, event_type: str, record: ExperienceRecord) -> None:
        self.store.put_json(
            kind="experience_record",
            value=record,
            version=f"experience-{record.experience_id}",
            parents=[record.source_version_id],
        )
        self.store.append_event(
            event_type,
            {"experience": record},
        )

    def _restore_from_events(self) -> None:
        lifecycle_events = {
            "experience_added",
            "experience_validated",
            "experience_outcome_recorded",
            "experience_invalidated",
            "experience_conflict_marked",
            "experience_conflict_resolved",
            "experience_expired",
        }
        for event in self.store.read_events():
            if event.get("event_type") not in lifecycle_events:
                continue
            raw = event.get("payload", {}).get("experience")
            if not isinstance(raw, dict):
                continue
            raw = dict(raw)
            raw["status"] = ExperienceStatus(raw["status"])
            record = ExperienceRecord(**raw)
            self.records[record.experience_id] = record
        for record in self.records.values():
            if record.status == ExperienceStatus.CONFLICTED and record.conflict_group_id:
                members = list(self.conflict_groups.get(record.conflict_group_id, ()))
                members.append(record.experience_id)
                self.conflict_groups[record.conflict_group_id] = tuple(members)
