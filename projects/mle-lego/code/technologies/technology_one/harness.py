from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable


COMPONENTS = ("generator", "selector", "filter", "memory", "skill")


@dataclass(frozen=True)
class HarnessComponent:
    name: str
    kind: str
    config: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessEdit:
    component: str
    patch: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class HarnessManifest:
    components: dict[str, HarnessComponent]
    parent_version: str | None = None
    edit_history: tuple[dict[str, Any], ...] = ()
    version: str = field(init=False)

    def __post_init__(self) -> None:
        unknown = set(self.components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown Harness components: {sorted(unknown)}")
        for component in self.components.values():
            missing = set(component.depends_on) - set(self.components)
            if missing:
                raise ValueError(
                    f"Component {component.name} depends on missing {sorted(missing)}"
                )
        payload = {
            "components": {
                name: {
                    "name": component.name,
                    "kind": component.kind,
                    "config": component.config,
                    "depends_on": list(component.depends_on),
                }
                for name, component in sorted(self.components.items())
            },
            "parent_version": self.parent_version,
            "edit_history": list(self.edit_history),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        object.__setattr__(self, "version", f"lego-harness-{digest}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "edit_history": list(self.edit_history),
            "components": {
                name: {
                    "name": component.name,
                    "kind": component.kind,
                    "config": copy.deepcopy(component.config),
                    "depends_on": list(component.depends_on),
                }
                for name, component in sorted(self.components.items())
            },
        }


def harness_manifest_from_dict(value: dict[str, Any]) -> HarnessManifest:
    """Rebuild a manifest from a persisted snapshot and verify its version."""
    if not isinstance(value, dict) or not isinstance(value.get("components"), dict):
        raise ValueError("Persisted Harness manifest is invalid")
    components: dict[str, HarnessComponent] = {}
    for name, raw in value["components"].items():
        if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
            raise ValueError(f"Persisted Harness component is invalid: {name!r}")
        components[str(name)] = HarnessComponent(
            name=str(raw.get("name", name)),
            kind=str(raw.get("kind", "")),
            config=copy.deepcopy(raw["config"]),
            depends_on=tuple(str(item) for item in raw.get("depends_on", ())),
        )
    history = value.get("edit_history", ())
    if not isinstance(history, (list, tuple)):
        raise ValueError("Persisted Harness edit history is invalid")
    manifest = HarnessManifest(
        components=components,
        parent_version=(
            None
            if value.get("parent_version") is None
            else str(value["parent_version"])
        ),
        edit_history=tuple(copy.deepcopy(history)),
    )
    expected_version = value.get("version")
    if expected_version is not None and manifest.version != expected_version:
        raise ValueError(
            "Persisted Harness version does not match its canonical contents"
        )
    return manifest


@dataclass
class HarnessVersionChain:
    current: HarnessManifest
    previous: list[HarnessManifest] = field(default_factory=list)

    def propose(self, edits: Iterable[HarnessEdit]) -> HarnessManifest:
        return apply_edits(self.current, edits)

    def accept(self, candidate: HarnessManifest) -> None:
        if candidate.parent_version != self.current.version:
            raise ValueError("Candidate does not descend from the current Harness")
        self.previous.append(self.current)
        self.current = candidate

    def rollback(self) -> HarnessManifest:
        if not self.previous:
            raise RuntimeError("No previous Harness version is available")
        self.current = self.previous.pop()
        return self.current


def initial_harness() -> HarnessManifest:
    return HarnessManifest(
        components={
            "generator": HarnessComponent(
                "generator", "data_generation", {"routes": ["account"]}
            ),
            "selector": HarnessComponent(
                "selector", "data_selection", {"min_quality": 0.85}, ("generator",)
            ),
            "filter": HarnessComponent(
                "filter", "data_filter", {"require_accepted": True}, ("selector",)
            ),
            "memory": HarnessComponent(
                "memory", "experience_memory", {"enabled": True}
            ),
            "skill": HarnessComponent(
                "skill", "executable_skill", {"enabled": True}, ("memory",)
            ),
        }
    )


def _merge_config(config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    for key, value in patch.items():
        merged[key] = copy.deepcopy(value)
    return merged


def apply_edits(
    manifest: HarnessManifest,
    edits: Iterable[HarnessEdit],
) -> HarnessManifest:
    edits = tuple(edits)
    if not edits:
        raise ValueError("At least one Harness edit is required")
    components = dict(manifest.components)
    history = list(manifest.edit_history)
    seen: set[str] = set()
    for edit in edits:
        if edit.component not in components:
            raise ValueError(f"Cannot edit unknown component: {edit.component}")
        if edit.component in seen:
            raise ValueError(f"Component edited more than once: {edit.component}")
        seen.add(edit.component)
        current = components[edit.component]
        components[edit.component] = HarnessComponent(
            name=current.name,
            kind=current.kind,
            config=_merge_config(current.config, edit.patch),
            depends_on=current.depends_on,
        )
        history.append(
            {
                "component": edit.component,
                "patch": copy.deepcopy(edit.patch),
                "rationale": edit.rationale,
            }
        )
    return HarnessManifest(
        components=components,
        parent_version=manifest.version,
        edit_history=tuple(history),
    )


def dependency_closure(
    manifest: HarnessManifest,
    components: Iterable[str],
) -> tuple[str, ...]:
    selected = set(components)
    changed = True
    while changed:
        changed = False
        for name in tuple(selected):
            for dependency in manifest.components[name].depends_on:
                if dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return tuple(name for name in COMPONENTS if name in selected)


def random_joint_edit(
    edits: dict[str, HarnessEdit],
    *,
    seed: int,
    size: int = 2,
) -> tuple[HarnessEdit, ...]:
    if size < 1 or size > len(edits):
        raise ValueError("Invalid random joint edit size")
    names = sorted(edits)
    random.Random(seed).shuffle(names)
    return tuple(edits[name] for name in names[:size])

