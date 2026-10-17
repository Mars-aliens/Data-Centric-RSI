from __future__ import annotations

"""Run a real decision-LLM Harness evolution loop for Technology One.

This runner is deliberately separate from the deterministic component matrix.
Each batch evaluates the active Harness, evaluates one model-proposed candidate
on the same batch, and commits the candidate only when held-in macro-F1
strictly improves. Held-out metrics are recorded for diagnosis only.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from core_system.mle_core.models import Budget, to_jsonable
from core_system.mle_core.store import ArtifactStore

from .data_validation import validate_records
from .harness import (
    HarnessEdit,
    HarnessManifest,
    apply_edits,
    dependency_closure,
    harness_manifest_from_dict,
)
from .harness_evolution import (
    HarnessEditProposal,
    HarnessProposalError,
    HarnessEvolutionController,
    _contains_forbidden_split_key,
    prompt_digest,
)
from .protocol import JsonlTaskAdapter, load_protocol
from .server_adapters import (
    CommandEvaluator,
    CommandTaskModelTrainer,
    JsonCommand,
    ServerCommandError,
    budget_from_value,
    run_json_command,
)
from .server_preflight import run_preflight
from .server_runner import _command, _required_string
from .server_technical_one import _run_candidate
from .technical_one_local import (
    SPLITS,
    _groups,
    _limit_groups,
    _select_groups,
    _train_records,
    initial_manifest,
)


TECH_EDITABLE_FIELDS: dict[str, frozenset[str]] = {
    "generator": frozenset({"allowed_buckets"}),
    "selector": frozenset({"max_response_words"}),
    "filter": frozenset({"min_group_margin"}),
}
TECH_PROPOSAL_MODES = frozenset(
    {"dependency-aware", "dependency-aware-joint", "independent"}
)
TECH_DEPENDENCY_RULE = """In dependency-aware mode, choose one target
component and use exactly the legal edit set listed in the user payload;
dependency-only ancestors must also appear in proposal.edits. Do not infer a
different dependency rule from the component graph."""
TECH_JOINT_DEPENDENCY_RULE = """In dependency-aware-joint mode, choose one
target from the legal joint targets and emit its complete multi-component edit
set. A generator-only edit is invalid: every proposal must include at least
generator+selector, and a filter target must include
generator+selector+filter. Do not infer a different dependency rule from the
component graph."""
TECH_SYSTEM_PROMPT = f"""You propose one bounded edit to a Technology One data Harness.
Return one JSON object with proposal.base_harness_version, proposal.edits, and
proposal.rationale. Each edit has component, patch, and a short rationale.
Editable fields are generator.allowed_buckets, selector.max_response_words,
and filter.min_group_margin. {TECH_DEPENDENCY_RULE} Do not edit the
training-group budget, model, evaluator,
code, commands, memory rules, or final-test policy. Use only held-in evidence
and prior eligible experience. You may also return a bounded reflection object
with problem, evidence, lesson, conditions, and source_batch. Held-out and
final-test data are never decision inputs. Keep the change minimal and do not
include private chain-of-thought. Every proposed field must differ from the
active configuration. Do not repeat a configuration listed in prior feedback."""
TECH_OBSERVATION_FIELDS = frozenset(
    {
        "run_id",
        "batch_id",
        "batch_index",
        "task_id",
        "active_harness_version",
        "held_in_metrics",
        "available_feedback",
        "remaining_budget",
        "attempted_configuration_count",
        "recent_attempted_configurations",
    }
)
ENDURANCE_STATE_VERSION = "technology-one-evolution-state-v2"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _configuration_fingerprint(
    profile: dict[str, Any],
    *,
    seed: int,
    max_train_groups: int,
    max_eval_groups: int,
    train_group_budget: int,
    proposal_mode: str,
    budget_overrides: dict[str, int | float] | None = None,
) -> str:
    payload = {
        "dataset": profile.get("dataset"),
        "decision_model": profile.get("decision_model"),
        "task_model": profile.get("task_model"),
        "evaluator": profile.get("evaluator"),
        "budget": profile.get("budget"),
        "seed": seed,
        "max_train_groups": max_train_groups,
        "max_eval_groups": max_eval_groups,
        "train_group_budget": train_group_budget,
        "proposal_mode": proposal_mode,
        "budget_overrides": budget_overrides or {},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _harness_config_fingerprint(manifest: HarnessManifest) -> str:
    """Fingerprint effective configuration without version or edit-history noise."""
    payload = {
        name: {
            "kind": component.kind,
            "config": component.config,
            "depends_on": list(component.depends_on),
        }
        for name, component in sorted(manifest.components.items())
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _harness_config_signature(manifest: HarnessManifest) -> dict[str, Any]:
    """Compact, policy-visible representation of a configuration attempt."""
    return {
        name: component.config
        for name, component in sorted(manifest.components.items())
        if name in TECH_EDITABLE_FIELDS
    }


def _rejection_reason_counts(transitions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for transition in transitions:
        if transition.get("status") != "rejected":
            continue
        reason = str(transition.get("rejection_reason", "unspecified_rejection"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _proposal_to_dict(proposal: HarnessEditProposal | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "base_version": proposal.base_version,
        "edits": [
            {
                "component": edit.component,
                "patch": edit.patch,
                "rationale": edit.rationale,
            }
            for edit in proposal.edits
        ],
        "rationale": proposal.rationale,
        "cost": dict(proposal.cost.__dict__),
        "model_id": proposal.model_id,
        "model_revision": proposal.model_revision,
        "system_prompt_hash": proposal.system_prompt_hash,
        "user_prompt_hash": proposal.user_prompt_hash,
        "reflection": proposal.reflection,
    }


def _proposal_from_dict(value: dict[str, Any] | None) -> HarnessEditProposal | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("edits"), list):
        raise ValueError("Persisted Harness proposal is invalid")
    return HarnessEditProposal(
        base_version=str(value["base_version"]),
        edits=tuple(
            HarnessEdit(
                component=str(item["component"]),
                patch=dict(item["patch"]),
                rationale=str(item.get("rationale", "")),
            )
            for item in value["edits"]
        ),
        rationale=str(value.get("rationale", "")),
        cost=budget_from_value(value.get("cost")),
        model_id=value.get("model_id"),
        model_revision=value.get("model_revision"),
        system_prompt_hash=value.get("system_prompt_hash"),
        user_prompt_hash=value.get("user_prompt_hash"),
        reflection=value.get("reflection"),
    )


class TechnologyOneProposalFailure(HarnessProposalError):
    """A decision proposal stayed invalid after the bounded schema retry."""

    def __init__(
        self,
        message: str,
        *,
        cost: Budget,
        model_id: str,
        model_revision: str,
        system_prompt_hash: str,
        user_prompt_hash: str,
    ) -> None:
        super().__init__(message)
        self.cost = cost
        self.model_id = model_id
        self.model_revision = model_revision
        self.system_prompt_hash = system_prompt_hash
        self.user_prompt_hash = user_prompt_hash


def _contains_forbidden(value: Any) -> bool:
    return _contains_forbidden_split_key(value)


def _technology_edit_component_sets(
    current: HarnessManifest, *, proposal_mode: str
) -> dict[str, list[str]]:
    """Return the exact component sets the policy may emit.

    The policy should not have to reconstruct dependency closure from prose.
    Keeping this derived from the active manifest also makes the prompt reflect
    any future change to the Harness dependency graph.
    """
    editable = tuple(name for name in TECH_EDITABLE_FIELDS if name in current.components)
    if proposal_mode == "independent":
        return {name: [name] for name in editable}
    if proposal_mode == "dependency-aware-joint":
        return {
            name: list(dependency_closure(current, (name,)))
            for name in editable
            if len(dependency_closure(current, (name,))) >= 2
        }
    return {
        name: list(dependency_closure(current, (name,)))
        for name in editable
    }


def build_technology_one_edit_prompt(
    *,
    current: HarnessManifest,
    observation: dict[str, Any],
    prior_feedback: dict[str, Any],
    proposal_mode: str = "dependency-aware",
) -> tuple[str, str]:
    if proposal_mode not in TECH_PROPOSAL_MODES:
        raise HarnessProposalError(f"unsupported proposal mode: {proposal_mode}")
    unknown = set(observation) - TECH_OBSERVATION_FIELDS
    if unknown:
        raise HarnessProposalError(
            f"edit observation contains fields outside the allowlist: {sorted(unknown)}"
        )
    if _contains_forbidden(prior_feedback):
        raise HarnessProposalError(
            "prior feedback cannot contain held-out or final-test fields"
        )
    edit_component_sets = _technology_edit_component_sets(
        current, proposal_mode=proposal_mode
    )
    payload = {
        "current_harness": current.to_dict(),
        "editable_fields": {
            name: sorted(fields) for name, fields in TECH_EDITABLE_FIELDS.items()
        },
        "dependency_graph": {
            name: list(component.depends_on)
            for name, component in sorted(current.components.items())
        },
        "proposal_mode": proposal_mode,
        "legal_edit_component_sets": edit_component_sets,
        "proposal_contract": (
            "Choose one target component and emit exactly its listed component "
            "set in proposal.edits, with no extra components."
            if proposal_mode in {"dependency-aware", "dependency-aware-joint"}
            else "Emit exactly one edit for exactly one listed component."
        ),
        "allowed_bucket_values": list(range(10)),
        "observation": observation,
        "prior_feedback": prior_feedback,
    }
    if proposal_mode == "independent":
        system_prompt = TECH_SYSTEM_PROMPT.replace(
            TECH_DEPENDENCY_RULE,
            "Edit exactly one component and do not include any dependency-only "
            "ancestors.",
        )
    elif proposal_mode == "dependency-aware-joint":
        system_prompt = TECH_SYSTEM_PROMPT.replace(
            TECH_DEPENDENCY_RULE,
            TECH_JOINT_DEPENDENCY_RULE,
        )
    else:
        system_prompt = TECH_SYSTEM_PROMPT
    return system_prompt, json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _parse_technology_edit(
    raw: dict[str, Any], *, current: HarnessManifest,
    proposal_mode: str = "dependency-aware",
) -> HarnessEditProposal:
    if proposal_mode not in TECH_PROPOSAL_MODES:
        raise HarnessProposalError(f"unsupported proposal mode: {proposal_mode}")
    if raw.get("base_harness_version") != current.version:
        raise HarnessProposalError("proposal does not target the active Harness version")
    raw_edits = raw.get("edits")
    if not isinstance(raw_edits, list) or not raw_edits or len(raw_edits) > 3:
        raise HarnessProposalError("edits must contain 1..3 entries")

    edits: list[HarnessEdit] = []
    seen: set[str] = set()
    for item in raw_edits:
        if not isinstance(item, dict):
            raise HarnessProposalError("each edit must be an object")
        component = item.get("component")
        patch = item.get("patch")
        rationale = item.get("rationale", "")
        if component not in TECH_EDITABLE_FIELDS or component in seen:
            raise HarnessProposalError(f"invalid or repeated component: {component!r}")
        if component not in current.components or not isinstance(patch, dict) or not patch:
            raise HarnessProposalError("edit component or patch is invalid")
        unknown = set(patch) - TECH_EDITABLE_FIELDS[component]
        if unknown:
            raise HarnessProposalError(f"non-editable fields: {sorted(unknown)}")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale.split()) > 80:
            raise HarnessProposalError("each edit needs a rationale of at most 80 words")

        if component == "generator":
            buckets = patch.get("allowed_buckets")
            if (
                not isinstance(buckets, list)
                or not buckets
                or len(buckets) > 10
                or len(set(buckets)) != len(buckets)
                or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 10 for item in buckets)
            ):
                raise HarnessProposalError("generator.allowed_buckets is invalid")
        elif component == "selector":
            words = patch.get("max_response_words")
            if isinstance(words, bool) or not isinstance(words, int) or not 8 <= words <= 320:
                raise HarnessProposalError("selector.max_response_words is invalid")
        elif component == "filter":
            margin = patch.get("min_group_margin")
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= 4:
                raise HarnessProposalError("filter.min_group_margin is invalid")

        seen.add(component)
        edits.append(HarnessEdit(component, patch, rationale.strip()))

    required = set(dependency_closure(current, seen))
    if proposal_mode in {"dependency-aware", "dependency-aware-joint"} and required != seen:
        raise HarnessProposalError(
            f"edit must include dependency closure: missing={sorted(required - seen)}"
        )
    if proposal_mode == "independent" and len(seen) != 1:
        raise HarnessProposalError("independent proposal must edit exactly one component")
    if proposal_mode == "dependency-aware-joint" and len(seen) < 2:
        raise HarnessProposalError(
            "dependency-aware-joint proposal must edit at least two components"
        )
    candidate = apply_edits(current, edits)
    if all(
        candidate.components[name].config == current.components[name].config
        for name in current.components
    ):
        raise HarnessProposalError(
            "proposal is a no-op: effective Harness configuration is unchanged"
        )
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale.split()) > 120:
        raise HarnessProposalError("proposal rationale is too long")
    return HarnessEditProposal(current.version, tuple(edits), rationale.strip())


def _parse_reflection(raw: dict[str, Any], *, batch_id: str) -> dict[str, Any] | None:
    value = raw.get("reflection")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HarnessProposalError("reflection must be an object")
    problem = str(value.get("problem", "")).strip()
    lesson = str(value.get("lesson", "")).strip()
    if not problem or not lesson or len(problem.split()) > 40 or len(lesson.split()) > 80:
        raise HarnessProposalError("reflection problem/lesson is invalid or too long")
    evidence = value.get("evidence", {})
    conditions = value.get("conditions", {})
    if not isinstance(evidence, dict) or not isinstance(conditions, dict):
        raise HarnessProposalError("reflection evidence and conditions must be objects")
    if _contains_forbidden(evidence) or _contains_forbidden(conditions):
        raise HarnessProposalError("reflection cannot contain held-out or final-test data")
    return {
        "reflection_id": str(value.get("reflection_id", f"{batch_id}-reflection")),
        "source_batch": str(value.get("source_batch", batch_id)),
        "problem": problem,
        "evidence": evidence,
        "lesson": lesson,
        "conditions": conditions,
        "eligible_from_batch": int(value.get("eligible_from_batch", 1)),
    }


class TechnologyOneHarnessEditor:
    def __init__(
        self,
        command: JsonCommand,
        *,
        model_id: str,
        model_revision: str,
        command_retries: int = 0,
        retry_backoff_seconds: float = 5.0,
        proposal_mode: str = "dependency-aware",
    ):
        if command_retries < 0:
            raise ValueError("command_retries cannot be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if proposal_mode not in TECH_PROPOSAL_MODES:
            raise ValueError(f"unsupported proposal mode: {proposal_mode}")
        self.command = command
        self.model_id = model_id
        self.model_revision = model_revision
        self.command_retries = command_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.proposal_mode = proposal_mode
        self.total_command_retries = 0

    def _call_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        for retry in range(self.command_retries + 1):
            try:
                return run_json_command(self.command, payload)
            except ServerCommandError:
                if retry >= self.command_retries:
                    raise
                self.total_command_retries += 1
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**retry))
        raise AssertionError("policy retry loop did not return or raise")

    def propose(
        self,
        *,
        current: HarnessManifest,
        observation: dict[str, Any],
        prior_feedback: dict[str, Any],
    ) -> HarnessEditProposal:
        system_prompt, user_prompt = build_technology_one_edit_prompt(
            current=current,
            observation=observation,
            prior_feedback=prior_feedback,
            proposal_mode=self.proposal_mode,
        )
        total_cost = Budget()
        last_error = "unknown proposal validation error"
        final_user_prompt = user_prompt
        for attempt in range(2):
            if attempt:
                final_user_prompt = user_prompt + (
                    "\n\nSchema correction: the previous proposal failed local "
                    f"validation with: {last_error}. Return only a corrected proposal. "
                    "Use the exact legal_edit_component_sets from the user payload. "
                    "In dependency-aware mode, choose one target component and set "
                    "proposal.edits to that target's complete listed set; for example, "
                    "a selector target requires [generator, selector], and a filter "
                    "target requires [generator, selector, filter]. In "
                    "dependency-aware-joint mode, generator-only is invalid and "
                    "the proposal must use one of the multi-component sets. In independent "
                    "mode, use exactly one component. Keep every patch within the "
                    "declared allowlist and do not repeat a prior configuration."
                )
            response = self._call_policy(
                {
                    "protocol": "lego-harness-edit-v1",
                    "model": {"id": self.model_id, "revision": self.model_revision},
                    "prompt": {"system": system_prompt, "user": final_user_prompt},
                },
            )
            attempt_cost = budget_from_value(response.get("cost"))
            total_cost = total_cost.add(attempt_cost)
            try:
                raw = response.get("proposal")
                if not isinstance(raw, dict):
                    raise HarnessProposalError("decision model did not return proposal")
                if attempt_cost.llm_calls < 1:
                    raise HarnessProposalError("decision proposal must report llm_calls >= 1")
                parsed = _parse_technology_edit(
                    raw, current=current, proposal_mode=self.proposal_mode
                )
                reflection = _parse_reflection(
                    response, batch_id=str(observation["batch_id"])
                )
            except HarnessProposalError as exc:
                last_error = str(exc)
                if attempt == 0:
                    continue
                raise TechnologyOneProposalFailure(
                    last_error,
                    cost=total_cost,
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                    system_prompt_hash=prompt_digest(system_prompt),
                    user_prompt_hash=prompt_digest(final_user_prompt),
                ) from exc
            return replace(
                parsed,
                cost=total_cost,
                model_id=self.model_id,
                model_revision=self.model_revision,
                system_prompt_hash=prompt_digest(system_prompt),
                user_prompt_hash=prompt_digest(final_user_prompt),
                reflection=reflection,
            )
        raise AssertionError("proposal retry loop did not return or raise")


def _group_hash(group_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(group_ids)).encode()).hexdigest()


def _feedback_view(transition: dict[str, Any]) -> dict[str, Any]:
    """Keep only information that may be shown to the next decision call."""
    return {
        "batch_id": transition["batch_id"],
        "status": transition["status"],
        "parent_version": transition["parent_version"],
        "candidate_version": transition["candidate_version"],
        "active_version": transition["active_version"],
        "edits": transition["edits"],
        "held_in_before": transition["held_in_before"],
        "held_in_candidate": transition["held_in_candidate"],
    }


def run_server_technology_one_evolution(
    profile_path: Path,
    *,
    root: Path,
    output_dir: Path,
    run_id: str,
    seed: int | None = None,
    batches: int | None = None,
    max_train_groups: int = 3000,
    max_eval_groups: int = 1000,
    train_group_budget: int | None = None,
    resume: bool = False,
    command_retries: int = 0,
    retry_backoff_seconds: float = 5.0,
    budget_overrides: dict[str, int | float] | None = None,
    proposal_mode: str = "dependency-aware",
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    if command_retries < 0:
        raise ValueError("command_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")
    if proposal_mode not in TECH_PROPOSAL_MODES:
        raise ValueError(f"unsupported proposal mode: {proposal_mode}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preflight = run_preflight(profile_path, root)
    if preflight["errors"]:
        raise ValueError("Server preflight failed: " + "; ".join(preflight["errors"]))
    protocol = load_protocol(root / profile["dataset"]["protocol"])
    records = JsonlTaskAdapter(protocol).load(root / profile["dataset"]["train_jsonl"])
    validate_records(records, protocol)
    run_seed = int(profile["seeds"][0] if seed is None else seed)
    batch_count = int(
        profile.get("stream", {}).get("max_batches", 3)
        if batches is None
        else batches
    )
    if batch_count < 2:
        raise ValueError("evolution requires at least two batches")
    by_split = {
        split: _groups(record for record in records if record.split == split)
        for split in SPLITS
    }
    train_groups = _limit_groups(by_split["seed"], max_train_groups)
    held_in_groups = _limit_groups(by_split["held_in"], max_eval_groups)
    held_out_groups = _limit_groups(by_split["held_out"], max_eval_groups)
    held_in_records = [record for group in held_in_groups.values() for record in group]
    held_out_records = [record for group in held_out_groups.values() for record in group]
    current = initial_manifest(
        train_group_budget=train_group_budget
        or int(profile["stream"].get("train_group_budget", 1000))
    )
    requested_budget = int(current.components["selector"].config["train_group_budget"])
    if len(train_groups) < requested_budget:
        raise ValueError("train group pool is smaller than the requested batch budget")

    run_root = output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "endurance_state.json"
    progress_path = run_root / "progress.json"
    summary_path = run_root / "summary.json"
    fingerprint = _configuration_fingerprint(
        profile,
        seed=run_seed,
        max_train_groups=max_train_groups,
        max_eval_groups=max_eval_groups,
        train_group_budget=requested_budget,
        proposal_mode=proposal_mode,
        budget_overrides=budget_overrides,
    )

    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            resume
            and existing_summary.get("run_id") == run_id
            and int(existing_summary.get("seed", -1)) == run_seed
            and int(existing_summary.get("batch_count", -1)) == batch_count
            and existing_summary.get("configuration_fingerprint") == fingerprint
        ):
            return existing_summary
        raise ValueError(
            "summary.json already exists for this run; use a new run_id"
        )

    store = ArtifactStore(run_root)
    budget_values: dict[str, int | float] = {
        "llm_calls": int(profile["budget"].get("llm_calls", 0)),
        "llm_tokens": int(profile["budget"].get("llm_tokens", 0)),
        "training_steps": int(profile["budget"].get("training_steps", 0)),
        "training_seconds": float(profile["budget"].get("training_seconds", 0.0)),
        "evaluator_calls": int(profile["budget"].get("evaluator_calls", 0)),
        "feedback_requests": int(profile["budget"].get("feedback_requests", 0)),
        "wall_clock_seconds": float(profile["budget"].get("wall_clock_seconds", 0.0)),
    }
    if budget_overrides:
        unknown_budget_fields = set(budget_overrides) - set(budget_values)
        if unknown_budget_fields:
            raise ValueError(
                f"Unknown budget override fields: {sorted(unknown_budget_fields)}"
            )
        budget_values.update(budget_overrides)
    budget_limit = Budget(
        llm_calls=int(budget_values["llm_calls"]),
        llm_tokens=int(budget_values["llm_tokens"]),
        training_steps=int(budget_values["training_steps"]),
        training_seconds=float(budget_values["training_seconds"]),
        evaluator_calls=int(budget_values["evaluator_calls"]),
        feedback_requests=int(budget_values["feedback_requests"]),
        wall_clock_seconds=float(budget_values["wall_clock_seconds"]),
    )

    if resume:
        if not state_path.exists():
            raise ValueError(f"cannot resume without {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state_version") != ENDURANCE_STATE_VERSION:
            raise ValueError("unsupported evolution endurance state version")
        if state.get("run_id") != run_id or int(state.get("seed", -1)) != run_seed:
            raise ValueError("resume state belongs to another run or seed")
        if int(state.get("target_batch_count", -1)) != batch_count:
            raise ValueError("resume requires the original target batch count")
        if state.get("configuration_fingerprint") != fingerprint:
            raise ValueError("resume configuration differs from the persisted run")
        current = harness_manifest_from_dict(state["active_harness"])
        batch_summaries = list(state.get("batches", []))
        transitions = list(state.get("transitions", []))
        feedback_history = list(state.get("feedback_history", []))
        reflection_history = list(state.get("reflections", []))
        attempted_configuration_hashes = set(
            str(value) for value in state.get("attempted_configuration_hashes", [])
        )
        attempted_configuration_records = list(
            state.get("attempted_configuration_records", [])
        )
        proposal = _proposal_from_dict(state.get("proposal"))
        pending_transition = state.get("pending_transition")
        total_cost = budget_from_value(state.get("budget"))
        next_batch_index = int(state.get("next_batch_index", 0))
        phase = str(state.get("phase", "ready_for_batch"))
        prior_elapsed = float(state.get("elapsed_wall_clock_seconds", 0.0))
        reliability = dict(state.get("reliability", {}))
        reliability["resume_count"] = int(reliability.get("resume_count", 0)) + 1
    else:
        if state_path.exists():
            raise ValueError(
                f"partial run state already exists at {state_path}; pass --resume"
            )
        batch_summaries: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        feedback_history: list[dict[str, Any]] = []
        reflection_history: list[dict[str, Any]] = []
        attempted_configuration_hashes: set[str] = set()
        attempted_configuration_records: list[dict[str, Any]] = []
        proposal: HarnessEditProposal | None = None
        pending_transition: dict[str, Any] | None = None
        total_cost = Budget()
        next_batch_index = 0
        phase = "ready_for_batch"
        prior_elapsed = 0.0
        reliability = {
            "resume_count": 0,
            "command_retry_count": 0,
            "recoverable_candidate_failure_count": 0,
            "duplicate_candidate_count": 0,
            "candidate_evaluation_skipped_count": 0,
            "proposal_noop_count": 0,
            "checkpoint_write_count": 0,
        }

    initial_harness_version = (
        str(state.get("initial_harness_version"))
        if resume
        else current.version
    )
    controller = HarnessEvolutionController(
        current, store=store, run_id=run_id, min_held_in_gain=0.0
    )
    editor = TechnologyOneHarnessEditor(
        _command(profile, "policy"),
        model_id=_required_string(profile, "decision_model.model_id"),
        model_revision=_required_string(profile, "decision_model.revision"),
        command_retries=command_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        proposal_mode=proposal_mode,
    )

    def elapsed_wall_clock() -> float:
        return prior_elapsed + (time.perf_counter() - invocation_started)

    def check_wall_clock_budget() -> None:
        limit = budget_limit.wall_clock_seconds
        if limit >= 0 and elapsed_wall_clock() > limit:
            raise ValueError(
                "evolution wall-clock budget exceeded: "
                f"used={elapsed_wall_clock():.1f}, limit={limit:.1f}"
            )

    def charge(cost: Budget) -> None:
        nonlocal total_cost
        updated = total_cost.add(cost)
        if updated.exceeds(budget_limit):
            raise ValueError(
                "evolution budget exceeded: "
                f"used={updated}, limit={budget_limit}"
            )
        total_cost = updated

    def sync_editor_retries() -> None:
        if editor.total_command_retries:
            reliability["command_retry_count"] = int(
                reliability.get("command_retry_count", 0)
            ) + editor.total_command_retries
            editor.total_command_retries = 0

    def save_state(state_phase: str, next_index: int) -> None:
        reliability["checkpoint_write_count"] = int(
            reliability.get("checkpoint_write_count", 0)
        ) + 1
        state_value = {
            "state_version": ENDURANCE_STATE_VERSION,
            "run_id": run_id,
            "seed": run_seed,
            "target_batch_count": batch_count,
            "configuration_fingerprint": fingerprint,
            "phase": state_phase,
            "next_batch_index": next_index,
            "initial_harness_version": initial_harness_version,
            "active_harness": controller.current.to_dict(),
            "proposal": _proposal_to_dict(proposal),
            "pending_transition": pending_transition,
            "budget": dict(total_cost.__dict__),
            "elapsed_wall_clock_seconds": elapsed_wall_clock(),
            "batches": batch_summaries,
            "transitions": transitions,
            "feedback_history": feedback_history,
            "reflections": reflection_history,
            "attempted_configuration_hashes": sorted(attempted_configuration_hashes),
            "attempted_configuration_records": attempted_configuration_records,
            "reliability": reliability,
            "final_test_used": False,
        }
        _atomic_write_json(state_path, state_value)
        _atomic_write_json(
            progress_path,
            {
                "run_id": run_id,
                "seed": run_seed,
                "status": "complete" if state_phase == "complete" else "running",
                "phase": state_phase,
                "completed_batches": len(batch_summaries),
                "target_batches": batch_count,
                "next_batch_index": next_index,
                "active_harness_version": controller.current.version,
                "accepted": sum(
                    1 for item in transitions if item.get("status") == "accepted"
                ),
                "rejected": sum(
                    1 for item in transitions if item.get("status") == "rejected"
                ),
                "attempted_configuration_count": len(attempted_configuration_hashes),
                "duplicate_candidate_count": int(
                    reliability.get("duplicate_candidate_count", 0)
                ),
                "candidate_evaluation_skipped_count": int(
                    reliability.get("candidate_evaluation_skipped_count", 0)
                ),
                "elapsed_wall_clock_seconds": elapsed_wall_clock(),
                "reliability": reliability,
                "final_test_used": False,
            },
        )

    def run_candidate_with_retries(
        *, batch_id: str, label: str, operation: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        for retry in range(command_retries + 1):
            try:
                return operation()
            except ServerCommandError as exc:
                if retry >= command_retries:
                    raise
                reliability["command_retry_count"] = int(
                    reliability.get("command_retry_count", 0)
                ) + 1
                store.append_event(
                    "evolution_command_retry",
                    {
                        "run_id": run_id,
                        "batch_id": batch_id,
                        "label": label,
                        "retry": retry + 1,
                        "error": str(exc),
                    },
                    event_id=f"{run_id}:{batch_id}:{label}:retry-{retry + 1}",
                )
                if retry_backoff_seconds:
                    time.sleep(retry_backoff_seconds * (2**retry))
        raise AssertionError("candidate retry loop did not return or raise")

    def propose_for_next_batch(source_batch: dict[str, Any], target_index: int) -> None:
        nonlocal proposal, pending_transition
        active_result = source_batch["active_result"]
        source_index = int(source_batch["batch_index"])
        observation = {
            "run_id": run_id,
            "batch_id": source_batch["batch_id"],
            "batch_index": source_index,
            "task_id": protocol.task_id,
            "active_harness_version": controller.current.version,
            "held_in_metrics": active_result["held_in"],
            "available_feedback": len(feedback_history),
            "remaining_budget": {
                "llm_calls": max(
                    0,
                    int(budget_limit.llm_calls)
                    - total_cost.llm_calls,
                ),
                "llm_tokens": max(
                    0,
                    int(budget_limit.llm_tokens)
                    - total_cost.llm_tokens,
                ),
            },
            "attempted_configuration_count": len(attempted_configuration_hashes),
            "recent_attempted_configurations": attempted_configuration_records[-16:],
        }
        prior_feedback = {
            "history": feedback_history[-4:],
            "reflections": reflection_history[-4:],
        }
        try:
            proposal = editor.propose(
                current=controller.current,
                observation=observation,
                prior_feedback=prior_feedback,
            )
        except TechnologyOneProposalFailure as exc:
            sync_editor_retries()
            charge(exc.cost)
            if "no-op" in str(exc).lower():
                reliability["proposal_noop_count"] = int(
                    reliability.get("proposal_noop_count", 0)
                ) + 1
            failure_batch_id = f"{run_id}-batch-{target_index:04d}"
            pending_transition = {
                "run_id": run_id,
                "batch_id": failure_batch_id,
                "status": "rejected",
                "parent_version": controller.current.version,
                "candidate_version": controller.current.version,
                "active_version": controller.current.version,
                "edits": [],
                "held_in_before": float(active_result["held_in"]["macro_f1"]),
                "held_in_candidate": None,
                "held_in_gain_threshold": 0.0,
                "held_out_candidate_report_only": None,
                "held_out_used_for_acceptance": False,
                "rollback_performed": True,
                "rollback_version": controller.current.version,
                "proposal_validation_failed": True,
                "rejection_reason": f"decision_proposal_invalid: {exc}",
                "policy_cost": exc.cost,
                "policy_model": {
                    "model_id": exc.model_id,
                    "model_revision": exc.model_revision,
                    "system_prompt_hash": exc.system_prompt_hash,
                    "user_prompt_hash": exc.user_prompt_hash,
                },
                "candidate_evaluation_cost": Budget(),
            }
            transitions.append(pending_transition)
            store.append_event(
                "harness_edit_proposal_rejected",
                pending_transition,
                event_id=f"{run_id}:{failure_batch_id}:proposal-rejected",
            )
            proposal = None
        except ServerCommandError as exc:
            sync_editor_retries()
            store.append_event(
                "decision_endpoint_unavailable",
                {
                    "run_id": run_id,
                    "source_batch_id": source_batch["batch_id"],
                    "target_batch_index": target_index,
                    "error": str(exc),
                },
                event_id=f"{run_id}:{source_batch['batch_id']}:policy-unavailable",
            )
            save_state("proposal_pending", target_index)
            raise
        else:
            sync_editor_retries()
            charge(proposal.cost)
            if proposal.reflection is not None:
                reflection_history.append(proposal.reflection)
                store.append_event(
                    "reflection_recorded",
                    {
                        "batch_id": source_batch["batch_id"],
                        "reflection": proposal.reflection,
                        "eligible_from_batch": target_index,
                    },
                    event_id=(
                        f"{run_id}:{source_batch['batch_id']}:reflection-recorded"
                    ),
                )
            store.append_event(
                "harness_edit_proposed",
                {
                    "batch_id": source_batch["batch_id"],
                    "base_version": proposal.base_version,
                    "edits": [
                        {"component": edit.component, "patch": edit.patch}
                        for edit in proposal.edits
                    ],
                    "cost": proposal.cost,
                    "model_id": proposal.model_id,
                    "model_revision": proposal.model_revision,
                },
                event_id=f"{run_id}:{source_batch['batch_id']}:harness-edit-proposed",
            )

    if not resume:
        save_state("ready_for_batch", 0)
    elif phase == "complete":
        if len(batch_summaries) != batch_count:
            raise ValueError("completed resume state has an incomplete batch ledger")
    elif phase == "proposal_pending":
        if not batch_summaries or next_batch_index >= batch_count:
            raise ValueError("proposal-pending resume state is inconsistent")
        check_wall_clock_budget()
        propose_for_next_batch(batch_summaries[-1], next_batch_index)
        phase = "ready_for_batch"
        save_state(phase, next_batch_index)
    elif phase != "ready_for_batch":
        raise ValueError(f"unknown resume phase: {phase}")

    for batch_index in range(next_batch_index, batch_count):
        check_wall_clock_budget()
        batch_id = f"{run_id}-batch-{batch_index:04d}"
        batch_root = run_root / f"batch-{batch_index:04d}"
        active_manifest = controller.current
        batch_seed = run_seed + batch_index * 1009
        active_records, active_group_ids = _select_groups(
            active_manifest, train_groups, requested_budget, batch_seed
        )
        try:
            active_result = run_candidate_with_retries(
                batch_id=batch_id,
                label="active",
                operation=lambda: _run_candidate(
                    root=batch_root,
                    label="active",
                    manifest=active_manifest,
                    train_records=_train_records(active_manifest, active_records),
                    train_group_ids=active_group_ids,
                    held_in_records=held_in_records,
                    held_out_records=held_out_records,
                    profile=profile,
                    seed=batch_seed,
                ),
            )
        except ServerCommandError as exc:
            store.append_event(
                "evolution_batch_interrupted",
                {
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "phase": "active_evaluation",
                    "error": str(exc),
                    "resume_from_batch": batch_index,
                },
                event_id=f"{run_id}:{batch_id}:active-interrupted",
            )
            save_state("ready_for_batch", batch_index)
            raise
        charge(
            Budget(
                training_steps=int(active_result["budget"].get("training_steps", 0)),
                training_seconds=float(active_result["budget"].get("training_seconds", 0.0)),
                evaluator_calls=int(active_result["budget"].get("evaluator_calls", 0)),
            )
        )
        batch_record: dict[str, Any] = {
            "batch_id": batch_id,
            "batch_index": batch_index,
            "active_harness_version_before": active_manifest.version,
            "active_result": active_result,
            "active_train_group_hash": _group_hash(active_group_ids),
            "candidate_result": None,
            "transition": None,
        }

        if pending_transition is not None and pending_transition["batch_id"] == batch_id:
            batch_record["transition"] = pending_transition
            pending_transition = None

        if proposal is not None:
            candidate_manifest = apply_edits(active_manifest, proposal.edits)
            candidate_config_hash = _harness_config_fingerprint(candidate_manifest)
            duplicate_candidate = candidate_config_hash in attempted_configuration_hashes
            if duplicate_candidate:
                reliability["duplicate_candidate_count"] = int(
                    reliability.get("duplicate_candidate_count", 0)
                ) + 1
                reliability["candidate_evaluation_skipped_count"] = int(
                    reliability.get("candidate_evaluation_skipped_count", 0)
                ) + 1
            else:
                attempted_configuration_hashes.add(candidate_config_hash)
                attempted_configuration_records.append(
                    {
                        "hash": candidate_config_hash,
                        "configuration": _harness_config_signature(candidate_manifest),
                    }
                )
            try:
                if duplicate_candidate:
                    raise ValueError("candidate configuration was already attempted")
                candidate_records, candidate_group_ids = _select_groups(
                    candidate_manifest, train_groups, requested_budget, batch_seed
                )
            except ValueError as exc:
                candidate_result = None
                candidate_group_ids = []
                transition = {
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "status": "rejected",
                    "parent_version": active_manifest.version,
                    "candidate_version": candidate_manifest.version,
                    "active_version": active_manifest.version,
                    "edits": [
                        {"component": edit.component, "patch": edit.patch, "rationale": edit.rationale}
                        for edit in proposal.edits
                    ],
                    "held_in_before": float(active_result["held_in"]["macro_f1"]),
                    "held_in_candidate": None,
                    "held_out_candidate_report_only": None,
                    "held_out_used_for_acceptance": False,
                    "rollback_performed": True,
                    "rollback_version": active_manifest.version,
                    "candidate_harness_config_hash": candidate_config_hash,
                    "candidate_evaluation_skipped": duplicate_candidate,
                    "rejection_reason": (
                        "duplicate_configuration"
                        if duplicate_candidate
                        else f"candidate_validation_failed: {exc}"
                    ),
                    "policy_cost": proposal.cost,
                    "policy_model": {
                        "model_id": proposal.model_id,
                        "model_revision": proposal.model_revision,
                        "system_prompt_hash": proposal.system_prompt_hash,
                        "user_prompt_hash": proposal.user_prompt_hash,
                    },
                    "candidate_evaluation_cost": Budget(),
                }
                store.append_event(
                    (
                        "harness_duplicate_candidate_skipped"
                        if duplicate_candidate
                        else "harness_candidate_validation_failed"
                    ),
                    transition,
                    event_id=(
                        f"{run_id}:{batch_id}:duplicate-candidate-skipped"
                        if duplicate_candidate
                        else f"{run_id}:{batch_id}:candidate-validation-failed"
                    ),
                )
            else:
                try:
                    candidate_result = run_candidate_with_retries(
                        batch_id=batch_id,
                        label="candidate",
                        operation=lambda: _run_candidate(
                            root=batch_root,
                            label="candidate",
                            manifest=candidate_manifest,
                            train_records=_train_records(
                                candidate_manifest, candidate_records
                            ),
                            train_group_ids=candidate_group_ids,
                            held_in_records=held_in_records,
                            held_out_records=held_out_records,
                            profile=profile,
                            seed=batch_seed,
                        ),
                    )
                except ServerCommandError as exc:
                    reliability["recoverable_candidate_failure_count"] = int(
                        reliability.get("recoverable_candidate_failure_count", 0)
                    ) + 1
                    candidate_result = None
                    transition = {
                        "run_id": run_id,
                        "batch_id": batch_id,
                        "status": "rejected",
                        "parent_version": active_manifest.version,
                        "candidate_version": candidate_manifest.version,
                        "active_version": active_manifest.version,
                        "edits": [
                            {
                                "component": edit.component,
                                "patch": edit.patch,
                                "rationale": edit.rationale,
                            }
                            for edit in proposal.edits
                        ],
                        "held_in_before": float(
                            active_result["held_in"]["macro_f1"]
                        ),
                        "held_in_candidate": None,
                        "held_in_gain_threshold": 0.0,
                        "held_out_candidate_report_only": None,
                        "held_out_used_for_acceptance": False,
                        "rollback_performed": True,
                        "rollback_version": active_manifest.version,
                        "candidate_harness_config_hash": candidate_config_hash,
                        "candidate_execution_failed": True,
                        "rejection_reason": f"candidate_execution_failed: {exc}",
                        "policy_cost": proposal.cost,
                        "candidate_evaluation_cost": Budget(),
                    }
                    store.append_event(
                        "harness_candidate_execution_failed",
                        transition,
                        event_id=f"{run_id}:{batch_id}:candidate-execution-failed",
                    )
                else:
                    candidate_cost = candidate_result["budget"]
                    candidate_budget = Budget(
                        training_steps=int(
                            candidate_cost.get("training_steps", 0)
                        ),
                        training_seconds=float(
                            candidate_cost.get("training_seconds", 0.0)
                        ),
                        evaluator_calls=int(
                            candidate_cost.get("evaluator_calls", 0)
                        ),
                    )
                    charge(candidate_budget)
                    candidate_metrics = {
                        "held_in_macro_f1": float(
                            candidate_result["held_in"]["macro_f1"]
                        ),
                        "held_out_macro_f1": float(
                            candidate_result["held_out"]["macro_f1"]
                        ),
                    }
                    transition = controller.consider(
                        proposal,
                        active_held_in=float(
                            active_result["held_in"]["macro_f1"]
                        ),
                        candidate_metrics=candidate_metrics,
                        held_in_metric="held_in_macro_f1",
                        held_out_metric="held_out_macro_f1",
                        batch_id=batch_id,
                        evaluation_cost=candidate_budget,
                    )
                    transition["candidate_harness_config_hash"] = candidate_config_hash
                    if transition["status"] == "rejected":
                        transition["rejection_reason"] = (
                            "held_in_no_improvement"
                            if transition["held_in_candidate"]
                            == transition["held_in_before"]
                            else "held_in_regression"
                        )
            transitions.append(transition)
            batch_record["candidate_result"] = candidate_result
            batch_record["candidate_train_group_hash"] = _group_hash(candidate_group_ids)
            batch_record["transition"] = transition

        batch_summaries.append(batch_record)
        proposal = None

        if batch_index < batch_count - 1:
            if transitions:
                feedback_history.append(_feedback_view(transitions[-1]))
            next_batch_index = batch_index + 1
            phase = "proposal_pending"
            save_state(phase, next_batch_index)
            propose_for_next_batch(batch_record, next_batch_index)
            phase = "ready_for_batch"
            save_state(phase, next_batch_index)
        else:
            next_batch_index = batch_count
            phase = "complete"
            save_state(phase, next_batch_index)

    if len(batch_summaries) != batch_count:
        raise ValueError(
            f"evolution ended with {len(batch_summaries)}/{batch_count} batches"
        )
    final_elapsed = elapsed_wall_clock()
    final_budget = dict(total_cost.__dict__)
    final_budget["wall_clock_seconds"] = final_elapsed
    summary = {
        "experiment": "technology-one-server-harness-evolution",
        "run_id": run_id,
        "dataset": profile["dataset"],
        "decision_policy": (
            "decision-llm-harness-edit-with-held-in-acceptance"
            f"; proposal_mode={proposal_mode}"
        ),
        "proposal_mode": proposal_mode,
        "decision_llm_used": True,
        "task_model": profile["task_model"],
        "evaluator": profile["evaluator"],
        "seed": run_seed,
        "batch_count": batch_count,
        "requested_train_group_budget": requested_budget,
        "natural_support": True,
        "acceptance_metric": "held_in_macro_f1",
        "held_out_used_for_acceptance": False,
        "final_test_used": False,
        "status": "complete",
        "run_status": "complete",
        "configuration_fingerprint": fingerprint,
        "initial_harness_version": initial_harness_version,
        "final_harness_version": controller.current.version,
        "accepted_version_count": sum(1 for item in transitions if item["status"] == "accepted"),
        "rejected_candidate_count": sum(1 for item in transitions if item["status"] == "rejected"),
        "rollback_count": sum(1 for item in transitions if item.get("rollback_performed", False)),
        "rejection_reason_counts": _rejection_reason_counts(transitions),
        "attempted_configuration_count": len(attempted_configuration_hashes),
        "attempted_configuration_hashes": sorted(attempted_configuration_hashes),
        "budget": final_budget,
        "budget_limit": dict(budget_limit.__dict__),
        "transitions": to_jsonable(transitions),
        "reflections": to_jsonable(reflection_history),
        "batches": to_jsonable(batch_summaries),
        "reliability": to_jsonable(reliability),
        "resume_supported": True,
        "state_checkpoint": str(state_path),
        "timing": {"wall_clock_seconds": final_elapsed},
        "interpretation": (
            "Decision-LLM-generated bounded Harness edits are evaluated on a "
            "candidate branch and accepted only by held-in improvement. Held-out "
            "metrics are report-only and final test remains locked."
        ),
    }
    _atomic_write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-batch decision-LLM Harness evolution for Technology One."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batches", type=int)
    parser.add_argument("--max-train-groups", type=int, default=3000)
    parser.add_argument("--max-eval-groups", type=int, default=1000)
    parser.add_argument("--train-group-budget", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same run-id from its atomic endurance_state.json.",
    )
    parser.add_argument(
        "--command-retries",
        type=int,
        default=0,
        help="Retry transient policy/trainer/evaluator command failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--proposal-mode",
        choices=sorted(TECH_PROPOSAL_MODES),
        default="dependency-aware",
        help="Decision-LLM proposal constraint used for the Technology One comparison.",
    )
    for field in (
        "llm_calls",
        "llm_tokens",
        "training_steps",
        "training_seconds",
        "evaluator_calls",
        "feedback_requests",
        "wall_clock_seconds",
    ):
        parser.add_argument(
            f"--budget-{field.replace('_', '-')}",
            type=float if field.endswith("seconds") else int,
            default=None,
        )
    args = parser.parse_args()
    print(
        json.dumps(
            run_server_technology_one_evolution(
                args.profile,
                root=args.root,
                output_dir=args.output_dir,
                run_id=args.run_id,
                seed=args.seed,
                batches=args.batches,
                max_train_groups=args.max_train_groups,
                max_eval_groups=args.max_eval_groups,
                train_group_budget=args.train_group_budget,
                resume=args.resume,
                command_retries=args.command_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                proposal_mode=args.proposal_mode,
                budget_overrides={
                    field: value
                    for field in (
                        "llm_calls",
                        "llm_tokens",
                        "training_steps",
                        "training_seconds",
                        "evaluator_calls",
                        "feedback_requests",
                        "wall_clock_seconds",
                    )
                    if (value := getattr(args, f"budget_{field}")) is not None
                },
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
