from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_BUDGET_KEYS = frozenset(
    {
        "llm_calls",
        "llm_tokens",
        "training_steps",
        "evaluator_calls",
        "feedback_requests",
        "wall_clock_seconds",
    }
)
REQUIRED_SPLITS = ("seed", "development", "held_out", "final_test")
PLACEHOLDER_MARKERS = ("to-be-declared", "to-be-pinned", "server/", "unset")
DATASET_STATUSES = frozenset(
    {
        "recommended-after-pii-and-terms-review",
        "terms-review-required",
        "recommended-controlled",
        "supplemental",
        "not-human-feedback",
        "paper-baseline-requires-pinning",
    }
)


@dataclass(frozen=True)
class ExternalBaselineSpec:
    key: str
    name: str
    paper_url: str
    official_code_url: str | None
    role: str
    primary_comparisons: tuple[str, ...]
    ablations: tuple[str, ...]
    required_inputs: tuple[str, ...]
    local_status: str


@dataclass(frozen=True)
class ExternalBaselineConfig:
    baseline_key: str
    task_id: str
    data_version: str
    dataset_manifest: str
    initial_checkpoint: str
    task_model: str
    evaluator: str
    seeds: tuple[int, ...]
    splits: tuple[str, ...]
    budget: dict[str, int | float]
    official_commit: str | None = None
    server_profile: str = "unset"
    final_test_unlocked: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExternalBaselineConfig":
        return cls(
            baseline_key=str(value["baseline_key"]),
            task_id=str(value["task_id"]),
            data_version=str(value["data_version"]),
            dataset_manifest=str(value["dataset_manifest"]),
            initial_checkpoint=str(value["initial_checkpoint"]),
            task_model=str(value["task_model"]),
            evaluator=str(value["evaluator"]),
            seeds=tuple(int(item) for item in value["seeds"]),
            splits=tuple(str(item) for item in value["splits"]),
            budget=dict(value["budget"]),
            official_commit=value.get("official_commit"),
            server_profile=str(value.get("server_profile", "unset")),
            final_test_unlocked=bool(value.get("final_test_unlocked", False)),
        )

    def validate(self, spec: ExternalBaselineSpec) -> None:
        if self.baseline_key != spec.key:
            raise ValueError(f"Config/spec mismatch: {self.baseline_key} != {spec.key}")
        required_values = (
            self.task_id,
            self.data_version,
            self.dataset_manifest,
            self.initial_checkpoint,
            self.task_model,
            self.evaluator,
        )
        if not all(required_values) or any(
            marker in value for value in required_values for marker in PLACEHOLDER_MARKERS
        ):
            raise ValueError("task_id, data_version, and dataset_manifest are required")
        if self.initial_checkpoint in {"", "unset", "null"}:
            raise ValueError("initial_checkpoint must be pinned before a server run")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("at least one non-negative seed is required")
        if self.splits != REQUIRED_SPLITS:
            raise ValueError(f"splits must be exactly {REQUIRED_SPLITS}")
        if self.final_test_unlocked:
            raise ValueError("final_test_unlocked must remain false in the pre-registration config")
        if self.server_profile == "unset":
            raise ValueError("server_profile must be pinned before a server run")
        if self.server_profile != "local-dry-run" and not self.official_commit:
            raise ValueError("official_commit is required for a server run")
        missing = REQUIRED_BUDGET_KEYS - set(self.budget)
        if missing:
            raise ValueError(f"budget is missing keys: {sorted(missing)}")
        if any(float(self.budget[key]) < 0 for key in REQUIRED_BUDGET_KEYS):
            raise ValueError("budget values must be non-negative")
        if self.budget["feedback_requests"] <= 0 and "feedback" in spec.required_inputs:
            raise ValueError(f"{spec.key} requires a positive feedback budget")


def load_dataset_registry(path: Path) -> list[dict[str, Any]]:
    """Load the dataset catalog and reject entries that are not auditable."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw.get("registry_version"), str) or not raw["registry_version"]:
        raise ValueError("dataset registry_version is required")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset registry must contain entries")

    required = {
        "dataset_key",
        "official_url",
        "paper_url",
        "revision",
        "license",
        "source_group_field",
        "feedback_fields",
        "supports_user_identity",
        "supports_session_replay",
        "supports_pairwise_preference",
        "final_test_source",
        "intended_role",
        "status",
    }
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError("each dataset registry entry must contain all audit fields")
        key = entry["dataset_key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"dataset_key must be unique and non-empty: {key!r}")
        seen.add(key)
        if not isinstance(entry["official_url"], str) or not entry["official_url"].startswith("https://"):
            raise ValueError(f"official_url must be an HTTPS URL: {key}")
        if entry["revision"] is not None:
            revision = entry["revision"]
            if not isinstance(revision, str) or len(revision) != 40 or any(
                character not in "0123456789abcdef" for character in revision
            ):
                raise ValueError(f"revision must be a full lowercase commit hash or null: {key}")
        if not isinstance(entry["source_group_field"], list) or not entry["source_group_field"]:
            raise ValueError(f"source_group_field is required: {key}")
        if not isinstance(entry["feedback_fields"], list) or not entry["feedback_fields"]:
            raise ValueError(f"feedback_fields is required: {key}")
        if entry["status"] not in DATASET_STATUSES:
            raise ValueError(f"unknown dataset status: {entry['status']}")
    return entries


def load_specs(path: Path) -> dict[str, ExternalBaselineSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = {}
    for item in raw["baselines"]:
        spec = ExternalBaselineSpec(
            key=str(item["key"]),
            name=str(item["name"]),
            paper_url=str(item["paper_url"]),
            official_code_url=item.get("official_code_url"),
            role=str(item["role"]),
            primary_comparisons=tuple(item["primary_comparisons"]),
            ablations=tuple(item["ablations"]),
            required_inputs=tuple(item["required_inputs"]),
            local_status=str(item["local_status"]),
        )
        if spec.key in specs:
            raise ValueError(f"Duplicate baseline spec: {spec.key}")
        specs[spec.key] = spec
    return specs


def load_configs(path: Path) -> list[ExternalBaselineConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ExternalBaselineConfig.from_dict(item) for item in raw["configs"]]
