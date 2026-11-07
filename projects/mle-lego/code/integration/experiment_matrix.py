from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class IntegrationFeatures:
    """The three switches used by the end-to-end ablation."""

    candidate_isolation: bool
    experience_governance: bool
    delayed_feedback_selection: bool

    @property
    def name(self) -> str:
        return (
            f"isolation-{int(self.candidate_isolation)}_"
            f"experience-{int(self.experience_governance)}_"
            f"selection-{int(self.delayed_feedback_selection)}"
        )

    def to_dict(self) -> dict[str, bool | str]:
        return {
            "name": self.name,
            "candidate_isolation": self.candidate_isolation,
            "experience_governance": self.experience_governance,
            "delayed_feedback_selection": self.delayed_feedback_selection,
        }


def all_feature_configs() -> tuple[IntegrationFeatures, ...]:
    return tuple(
        IntegrationFeatures(*values)
        for values in product((False, True), repeat=3)
    )


def validate_feature_config(value: IntegrationFeatures) -> None:
    if not isinstance(value, IntegrationFeatures):
        raise TypeError("integration configuration must be IntegrationFeatures")


def feature_config_from_name(name: str) -> IntegrationFeatures:
    for config in all_feature_configs():
        if config.name == name:
            return config
    raise ValueError(
        f"unknown integration configuration {name!r}; "
        f"expected one of {[item.name for item in all_feature_configs()]}"
    )
