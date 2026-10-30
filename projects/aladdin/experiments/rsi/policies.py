"""Versioned policy state and conservative feedback-driven updates."""

from __future__ import annotations

import copy
from typing import Any


def initial_policy_state() -> dict[str, Any]:
    return {
        "synthesis": {"version": 0, "guidance": []},
        "cleaning": {
            "version": 0,
            "near_duplicate_threshold": 0.92,
            "repeated_ngram_threshold": 0.65,
        },
        "selection": {"version": 0, "impact_weight": 0.5},
    }


def update_policy_state(
    state: dict[str, Any],
    feedback_events: list[dict[str, Any]],
    *,
    update_synthesis: bool = True,
    update_cleaning: bool = True,
    update_selection: bool = True,
    selector_learning_rate: float = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < selector_learning_rate <= 1.0:
        raise ValueError("selector_learning_rate must be in (0, 1]")
    updated = copy.deepcopy(state)
    before = copy.deepcopy(state)
    applied: dict[str, list[str]] = {"synthesis": [], "cleaning": [], "selection": []}

    if update_synthesis:
        guidance = list(updated["synthesis"].get("guidance", []))
        known = {str(value).strip().casefold() for value in guidance}
        for event in feedback_events:
            requirement = str(event.get("new_requirement") or "").strip()
            if requirement and requirement.casefold() not in known:
                guidance.append(requirement)
                known.add(requirement.casefold())
                applied["synthesis"].append(str(event.get("query_id") or ""))
        if guidance != updated["synthesis"].get("guidance", []):
            updated["synthesis"]["guidance"] = guidance[-8:]
            updated["synthesis"]["version"] += 1

    if update_cleaning:
        repetitive_events = [
            event
            for event in feedback_events
            if event.get("issue_code") == "repetitive"
        ]
        if len(repetitive_events) >= 2:
            before_threshold = float(updated["cleaning"]["repeated_ngram_threshold"])
            after_threshold = max(0.25, before_threshold - min(0.15, 0.05 * (len(repetitive_events) // 2)))
            if after_threshold < before_threshold:
                updated["cleaning"]["repeated_ngram_threshold"] = after_threshold
                updated["cleaning"]["version"] += 1
                applied["cleaning"] = [str(event.get("query_id") or "") for event in repetitive_events]

    if update_selection and feedback_events:
        old_weight = float(updated["selection"].get("impact_weight", 0.5))
        actionable_rate = sum(bool(event.get("actionable")) for event in feedback_events) / len(feedback_events)
        new_weight = min(2.0, max(0.0, old_weight + selector_learning_rate * (actionable_rate - 0.5)))
        if abs(new_weight - old_weight) > 1e-12:
            updated["selection"]["impact_weight"] = new_weight
            updated["selection"]["version"] += 1
            applied["selection"] = [str(event.get("query_id") or "") for event in feedback_events]

    changes = {}
    for component in ("synthesis", "cleaning", "selection"):
        if updated[component] != before[component]:
            changes[component] = {"before": before[component], "after": updated[component]}
    event = {
        "schema_version": 1,
        "event_type": "policy_update",
        "status": "updated" if changes else "no_op",
        "feedback_query_ids": [str(item.get("query_id") or "") for item in feedback_events],
        "applied_feedback_query_ids": applied,
        "updates_enabled": {
            "synthesis": update_synthesis,
            "cleaning": update_cleaning,
            "selection": update_selection,
        },
        "changes": changes,
        "state_before": before,
        "state_after": updated,
    }
    return updated, event
