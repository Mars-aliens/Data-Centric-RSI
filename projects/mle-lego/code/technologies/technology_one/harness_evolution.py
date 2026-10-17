from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

from core_system.mle_core.models import Budget
from core_system.mle_core.store import ArtifactStore

from .harness import HarnessEdit, HarnessManifest, HarnessVersionChain, apply_edits, dependency_closure


EDITABLE_FIELDS: dict[str, frozenset[str]] = {
    "generator": frozenset({"routes"}),
    "selector": frozenset({"min_quality"}),
    "filter": frozenset({"require_accepted"}),
    "memory": frozenset({"enabled"}),
    "skill": frozenset({"enabled"}),
}
ROUTE_VALUES = frozenset({"account", "billing", "technical", "safety"})
MAX_EDITS = 5
EDIT_OBSERVATION_FIELDS = frozenset({
    "run_id", "batch_id", "batch_index", "task_id", "held_in_metrics",
    "available_feedback", "remaining_budget",
})
EDIT_SYSTEM_PROMPT = """You propose bounded edits to a data-processing Harness.
Return one JSON object with proposal.base_harness_version, proposal.edits, and
proposal.rationale. Each edit has component, patch, and a short rationale.
Only edit fields shown in editable_fields. Include the complete dependency
closure for changed components. Do not propose code, commands, new components,
or access to held_out/final_test. Use only prior held-in feedback; held-out
metrics are report-only and are never decision inputs. A candidate is tested
on held-in and is accepted only for a strict improvement over the active
version. Keep edits minimal and explain the operational reason, not private
chain-of-thought."""


def build_harness_edit_prompt(
    *,
    current: HarnessManifest,
    observation: dict[str, Any],
    prior_feedback: dict[str, Any],
) -> tuple[str, str]:
    unknown = set(observation) - EDIT_OBSERVATION_FIELDS
    if unknown:
        raise HarnessProposalError(
            f"edit observation contains fields outside the allowlist: {sorted(unknown)}"
        )
    if _contains_forbidden_split_key(prior_feedback):
        raise HarnessProposalError("prior feedback cannot contain held-out or final-test fields")
    payload = {
        "current_harness": current.to_dict(),
        "editable_fields": {name: sorted(fields) for name, fields in EDITABLE_FIELDS.items()},
        "allowed_generator_routes": sorted(ROUTE_VALUES),
        "observation": observation,
        "prior_feedback": prior_feedback,
    }
    return EDIT_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _contains_forbidden_split_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "held_out" in normalized or "final_test" in normalized:
                return True
            if _contains_forbidden_split_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_split_key(child) for child in value)
    return False


class HarnessProposalError(ValueError):
    pass


@dataclass(frozen=True)
class HarnessEditProposal:
    base_version: str
    edits: tuple[HarnessEdit, ...]
    rationale: str
    cost: Budget = Budget()
    model_id: str | None = None
    model_revision: str | None = None
    system_prompt_hash: str | None = None
    user_prompt_hash: str | None = None
    reflection: dict[str, Any] | None = None


def prompt_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_edit_proposal(raw: dict[str, Any], *, current: HarnessManifest) -> HarnessEditProposal:
    """Parse the bounded JSON proposal emitted by the decision model."""
    base_version = raw.get("base_harness_version")
    if base_version != current.version:
        raise HarnessProposalError(
            f"stale Harness proposal: base={base_version!r}, current={current.version!r}"
        )
    raw_edits = raw.get("edits")
    if not isinstance(raw_edits, list) or not raw_edits or len(raw_edits) > MAX_EDITS:
        raise HarnessProposalError(f"edits must contain 1..{MAX_EDITS} entries")

    edits: list[HarnessEdit] = []
    seen: set[str] = set()
    for item in raw_edits:
        if not isinstance(item, dict):
            raise HarnessProposalError("each edit must be an object")
        component = item.get("component")
        patch = item.get("patch")
        rationale = item.get("rationale", "")
        if component not in EDITABLE_FIELDS or component not in current.components:
            raise HarnessProposalError(f"component is not editable: {component!r}")
        if component in seen:
            raise HarnessProposalError(f"component edited more than once: {component}")
        seen.add(component)
        if not isinstance(patch, dict) or not patch:
            raise HarnessProposalError("patch must be a non-empty object")
        unknown_fields = set(patch) - EDITABLE_FIELDS[component]
        if unknown_fields:
            raise HarnessProposalError(
                f"fields are not editable for {component}: {sorted(unknown_fields)}"
            )
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale.split()) > 80:
            raise HarnessProposalError("each edit requires a rationale of at most 80 words")
        if component == "generator" and "routes" in patch:
            routes = patch["routes"]
            if (
                not isinstance(routes, list)
                or not routes
                or len(routes) > len(ROUTE_VALUES)
                or any(route not in ROUTE_VALUES for route in routes)
                or len(set(routes)) != len(routes)
            ):
                raise HarnessProposalError("generator.routes contains invalid route values")
        if component == "selector" and "min_quality" in patch:
            value = patch["min_quality"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise HarnessProposalError("selector.min_quality must be between 0 and 1")
        if component == "filter" and "require_accepted" in patch:
            if not isinstance(patch["require_accepted"], bool):
                raise HarnessProposalError("filter.require_accepted must be boolean")
        if component in {"memory", "skill"} and "enabled" in patch:
            if not isinstance(patch["enabled"], bool):
                raise HarnessProposalError(f"{component}.enabled must be boolean")
        edits.append(HarnessEdit(component, patch, rationale.strip()))

    required = set(dependency_closure(current, seen))
    if required != seen:
        raise HarnessProposalError(
            f"edit must include dependency closure: missing={sorted(required - seen)}"
        )
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale.split()) > 120:
        raise HarnessProposalError("proposal rationale must be at most 120 words")
    return HarnessEditProposal(current.version, tuple(edits), rationale.strip())


class HarnessEvolutionController:
    """Apply model proposals transactionally; held-out metrics never gate acceptance."""

    def __init__(
        self,
        manifest: HarnessManifest,
        *,
        store: ArtifactStore,
        run_id: str,
        min_held_in_gain: float = 0.0,
    ):
        self.chain = HarnessVersionChain(manifest)
        self.store = store
        self.run_id = run_id
        self.min_held_in_gain = min_held_in_gain
        if min_held_in_gain < 0:
            raise ValueError("min_held_in_gain cannot be negative")

    @property
    def current(self) -> HarnessManifest:
        return self.chain.current

    def consider(
        self,
        proposal: HarnessEditProposal,
        *,
        active_held_in: float,
        candidate_metrics: dict[str, float],
        held_in_metric: str,
        held_out_metric: str,
        batch_id: str,
        evaluation_cost: Budget = Budget(),
    ) -> dict[str, Any]:
        current = self.current
        if proposal.base_version != current.version:
            raise HarnessProposalError("proposal became stale before commit")
        if held_in_metric not in candidate_metrics or held_out_metric not in candidate_metrics:
            raise HarnessProposalError("candidate metrics must include held-in and held-out")
        if any("final_test" in name.lower() for name in candidate_metrics):
            raise HarnessProposalError("candidate evaluation cannot include final-test metrics")
        candidate = apply_edits(current, proposal.edits)
        candidate_held_in = float(candidate_metrics[held_in_metric])
        candidate_held_out = float(candidate_metrics[held_out_metric])
        if not all(math.isfinite(value) for value in (active_held_in, candidate_held_in, candidate_held_out)):
            raise HarnessProposalError("candidate metrics must be finite numbers")
        accepted = candidate_held_in > active_held_in + self.min_held_in_gain
        old_version = current.version
        if accepted:
            self.chain.accept(candidate)
            status = "accepted"
        else:
            status = "rejected"
        transition = {
            "run_id": self.run_id,
            "batch_id": batch_id,
            "status": status,
            "parent_version": old_version,
            "candidate_version": candidate.version,
            "active_version": self.current.version,
            "edits": [
                {"component": edit.component, "patch": edit.patch, "rationale": edit.rationale}
                for edit in proposal.edits
            ],
            "held_in_before": active_held_in,
            "held_in_candidate": candidate_held_in,
            "held_in_gain_threshold": self.min_held_in_gain,
            "held_out_candidate_report_only": candidate_held_out,
            "held_out_used_for_acceptance": False,
            "rollback_performed": not accepted,
            "rollback_version": old_version if not accepted else None,
            "policy_cost": proposal.cost,
            "policy_model": {
                "model_id": proposal.model_id,
                "model_revision": proposal.model_revision,
                "system_prompt_hash": proposal.system_prompt_hash,
                "user_prompt_hash": proposal.user_prompt_hash,
            },
            "candidate_evaluation_cost": evaluation_cost,
        }
        self.store.append_event(
            "harness_candidate_evaluated",
            transition,
            event_id=f"{self.run_id}:{batch_id}:harness-candidate-evaluated",
        )
        self.store.put_json(
            kind="harness_manifest",
            value=candidate.to_dict(),
            version=candidate.version,
            parents=[old_version],
        )
        if accepted:
            self.store.append_event(
                "harness_version_accepted",
                {"run_id": self.run_id, "batch_id": batch_id, "version": candidate.version},
                event_id=f"{self.run_id}:{batch_id}:harness-version-accepted",
            )
        else:
            self.store.append_event(
                "harness_candidate_rolled_back",
                {
                    "run_id": self.run_id,
                    "batch_id": batch_id,
                    "candidate_version": candidate.version,
                    "active_version": old_version,
                    "reason": "held_in_regression_or_no_improvement",
                },
                event_id=f"{self.run_id}:{batch_id}:harness-candidate-rolled-back",
            )
        return transition

    def evolve_batch(
        self,
        *,
        editor: Any,
        observation: dict[str, Any],
        prior_feedback: dict[str, Any],
        active_held_in: float,
        evaluate_candidate: Callable[[HarnessManifest], tuple[dict[str, float], Budget]],
        held_in_metric: str,
        held_out_metric: str,
        batch_id: str,
    ) -> dict[str, Any]:
        """Propose, evaluate, and transactionally commit one candidate version."""
        proposal = editor.propose(
            current=self.current,
            observation=observation,
            prior_feedback=prior_feedback,
        )
        candidate = apply_edits(self.current, proposal.edits)
        metrics, evaluation_cost = evaluate_candidate(candidate)
        transition = self.consider(
            proposal,
            active_held_in=active_held_in,
            candidate_metrics=metrics,
            held_in_metric=held_in_metric,
            held_out_metric=held_out_metric,
            batch_id=batch_id,
            evaluation_cost=evaluation_cost,
        )
        return transition

