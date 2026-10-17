"""Auditable requirement state for multi-round feedback experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequirementEvent:
    round_index: int
    action: str
    requirement_id: str
    text: str
    source: str
    status: str = "proposed"


@dataclass
class RequirementState:
    """Versioned requirements; mutation is only possible through ``apply``."""

    requirements: dict[str, str]
    confirmed: set[str] = field(default_factory=set)
    version: int = 0
    events: list[RequirementEvent] = field(default_factory=list)

    def apply(self, *, round_index: int, action: str, requirement_id: str,
              text: str, source: str = "human", confirm: bool = False) -> None:
        if round_index < 0 or not requirement_id.strip() or not text.strip():
            raise ValueError("round_index, requirement_id, and text must be non-empty")
        if action not in {"add", "edit", "confirm", "reject"}:
            raise ValueError(f"unknown requirement action: {action}")
        exists = requirement_id in self.requirements
        if action == "add" and exists:
            raise ValueError(f"requirement already exists: {requirement_id}")
        if action == "edit" and not exists:
            raise ValueError(f"cannot edit unknown requirement: {requirement_id}")
        if action == "reject":
            self.confirmed.discard(requirement_id)
        else:
            self.requirements[requirement_id] = text.strip()
            if confirm or action == "confirm":
                self.confirmed.add(requirement_id)
        self.version += 1
        self.events.append(RequirementEvent(
            round_index, action, requirement_id, text.strip(), source,
            "confirmed" if requirement_id in self.confirmed else "proposed",
        ))

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "requirements": dict(sorted(self.requirements.items())),
            "confirmed": sorted(self.confirmed),
            "events": [event.__dict__.copy() for event in self.events],
        }
