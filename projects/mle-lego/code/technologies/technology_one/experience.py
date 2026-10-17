from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Experience:
    experience_id: str
    problem: str
    conditions: dict[str, Any]
    reflection: str
    edits: tuple[dict[str, Any], ...]
    validation: dict[str, Any]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "problem": self.problem,
            "conditions": self.conditions,
            "reflection": self.reflection,
            "edits": list(self.edits),
            "validation": self.validation,
            "status": self.status,
        }


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    trigger: str
    procedure: tuple[str, ...]
    source_experience_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "trigger": self.trigger,
            "procedure": list(self.procedure),
            "source_experience_ids": list(self.source_experience_ids),
        }


@dataclass
class ExperienceStore:
    experiences: list[Experience] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)

    def add(self, experience: Experience) -> None:
        self.experiences.append(experience)

    def add_skill(self, skill: Skill) -> None:
        self.skills.append(skill)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiences": [item.to_dict() for item in self.experiences],
            "skills": [item.to_dict() for item in self.skills],
        }

