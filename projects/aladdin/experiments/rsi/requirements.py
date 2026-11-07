"""Controlled requirement-release schedules and objective compliance checks."""

from __future__ import annotations

from typing import Any


CRITERIA = {"contains", "max_chars", "starts_with"}


def validate_requirement_schedule(schedule: Any, rounds: int) -> list[dict[str, Any]]:
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("requirements comparison needs a non-empty requirement_schedule")
    ids = set()
    normalized = []
    for item in schedule:
        if not isinstance(item, dict):
            raise ValueError("each requirement schedule entry must be an object")
        requirement_id = str(item.get("requirement_id") or "").strip()
        text = str(item.get("text") or "").strip()
        release_round = item.get("release_round")
        criterion = item.get("criterion")
        if not requirement_id or requirement_id in ids:
            raise ValueError("requirement IDs must be non-empty and unique")
        if not text:
            raise ValueError(f"requirement {requirement_id} needs non-empty text")
        if not isinstance(release_round, int) or isinstance(release_round, bool) or not 0 <= release_round < rounds:
            raise ValueError(f"requirement {requirement_id} release_round must be within configured rounds")
        if not isinstance(criterion, dict) or criterion.get("type") not in CRITERIA:
            raise ValueError(f"requirement {requirement_id} needs a supported objective criterion")
        kind = criterion["type"]
        expected = criterion.get("value")
        if kind == "max_chars" and (not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0):
            raise ValueError(f"requirement {requirement_id} max_chars value must be a positive integer")
        if kind in {"contains", "starts_with"} and (not isinstance(expected, str) or not expected.strip()):
            raise ValueError(f"requirement {requirement_id} criterion value must be non-empty text")
        ids.add(requirement_id)
        normalized.append({**item, "requirement_id": requirement_id, "text": text})
    return sorted(normalized, key=lambda item: (item["release_round"], item["requirement_id"]))


def evaluate_requirement_compliance(
    records: list[dict[str, Any]], requirements: list[dict[str, Any]]
) -> dict[str, Any]:
    per_requirement = {}
    for requirement in requirements:
        criterion = requirement["criterion"]
        kind, expected = criterion["type"], criterion["value"]
        passed = 0
        for record in records:
            value = str(record.get(criterion.get("field", "output")) or "").strip()
            if kind == "contains":
                matched = expected.casefold() in value.casefold()
            elif kind == "starts_with":
                matched = value.casefold().startswith(expected.casefold())
            else:
                matched = len(value) <= expected
            passed += int(matched)
        per_requirement[requirement["requirement_id"]] = {
            "release_round": requirement["release_round"],
            "compliant": passed,
            "evaluated": len(records),
            "rate": passed / len(records) if records else None,
        }
    total = sum(row["compliant"] for row in per_requirement.values())
    denominator = sum(row["evaluated"] for row in per_requirement.values())
    return {
        "per_requirement": per_requirement,
        "joint_compliance_rate": total / denominator if denominator else None,
        "requirement_coverage_rate": sum(row["rate"] is not None for row in per_requirement.values()) / len(requirements) if requirements else None,
        "requirement_count": len(requirements),
        "candidate_count": len(records),
    }
