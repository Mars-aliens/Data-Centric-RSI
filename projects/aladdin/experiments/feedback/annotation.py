"""Interactive collection of new feedback with measured review time."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


JUDGMENTS = {
    "1": "usable",
    "2": "needs_revision",
    "3": "unusable",
    "4": "cannot_judge",
}
ISSUES = {
    "1": "incorrect",
    "2": "irrelevant",
    "3": "incomplete",
    "4": "unclear",
    "5": "format",
    "6": "repetitive",
    "7": "other",
    "0": "none",
}
CLARITY = {"1": "clear", "2": "partly_unclear", "3": "unclear", "0": "not_applicable"}


def _ask_choice(
    prompt: str, choices: dict[str, str], input_fn: Callable[[str], str]
) -> str:
    allowed = "/".join(choices)
    while True:
        answer = input_fn(f"{prompt} [{allowed}]: ").strip()
        if answer in choices:
            return choices[answer]
        print(f"请输入以下选项之一：{allowed}")


def _show_candidate(candidate: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    context = candidate.get("context") or []
    if context:
        print("上下文：")
        for turn in context:
            if isinstance(turn, dict):
                role = str(turn.get("role") or "user")
                content = str(turn.get("content") or "")
            else:
                role, content = "对话", str(turn)
            print(f"{role}: {content}")
    print(f"输入：{candidate.get('input', '')}")
    print(f"候选回答：{candidate.get('output', '')}")
    print("=" * 72)


def collect_one_feedback(
    query: dict[str, Any],
    *,
    annotator_id: str,
    input_fn: Callable[[str], str] = input,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if not annotator_id.strip():
        raise ValueError("annotator_id must be non-empty")
    candidate = query.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("query is missing its candidate object")
    started = datetime.now(timezone.utc).isoformat()
    start_clock = clock()
    _show_candidate(candidate)
    judgment = _ask_choice(
        "判断：1 可用，2 需要修改，3 不可用，4 无法判断", JUDGMENTS, input_fn
    )
    issue = "none"
    if judgment in {"needs_revision", "unusable"}:
        issue = _ask_choice(
            "主要问题：1 不正确，2 不相关，3 不完整，4 不清楚，5 格式，6 重复，7 其他，0 无", 
            ISSUES,
            input_fn,
        )
    clarity = _ask_choice(
        "要求清晰度：1 清晰，2 部分不清晰，3 不清晰，0 不适用", CLARITY, input_fn
    )
    corrected = input_fn("可选：请直接填写修改后的回答，留空跳过： ").strip()
    requirement = input_fn("可选：本例揭示的新要求，留空表示没有： ").strip()
    ended = datetime.now(timezone.utc).isoformat()
    duration = max(0.0, clock() - start_clock)
    return {
        "schema_version": 1,
        "feedback_source": "live_human",
        "query_id": query["query_id"],
        "record_id": query["record_id"],
        "source_group_id": query.get("source_group_id"),
        "arm_id": query.get("arm_id"),
        "round": query.get("round"),
        "selection_method": query.get("selection_method"),
        "annotator_id": annotator_id.strip(),
        "judgment": judgment,
        "issue_code": issue,
        "requirement_clarity": clarity,
        "corrected_output": corrected or None,
        "new_requirement": requirement or None,
        "actionable": bool(corrected or requirement),
        "started_at_utc": started,
        "ended_at_utc": ended,
        "duration_seconds": duration,
        "candidate_was_quarantined": bool(query.get("candidate_was_quarantined")),
    }


def collect_interactive_feedback(
    queries: list[dict[str, Any]],
    *,
    annotator_id: str,
    budget_seconds_per_arm: float,
    output_path: str | Path,
    input_fn: Callable[[str], str] = input,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Resume a shuffled annotation queue up to a measured per-arm time target."""
    if not math.isfinite(budget_seconds_per_arm) or budget_seconds_per_arm <= 0:
        raise ValueError("budget_seconds_per_arm must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = read_feedback(destination) if destination.exists() else []
    queries_by_id = {str(query.get("query_id") or ""): query for query in queries}
    if "" in queries_by_id or len(queries_by_id) != len(queries):
        raise ValueError("annotation queue must have unique non-empty query IDs")
    existing_by_id = {str(event["query_id"]): event for event in existing}
    if not set(existing_by_id).issubset(queries_by_id):
        raise ValueError("feedback file contains a query not present in the annotation queue")
    for query_id, event in existing_by_id.items():
        if str(event.get("arm_id") or "") != str(queries_by_id[query_id].get("arm_id") or ""):
            raise ValueError(f"feedback arm does not match query {query_id}")
    elapsed_by_arm: dict[str, float] = {}
    for event in existing:
        arm_id = str(event.get("arm_id") or "")
        elapsed_by_arm[arm_id] = elapsed_by_arm.get(arm_id, 0.0) + float(event["duration_seconds"])
    events: list[dict[str, Any]] = list(existing)
    for query in queries:
        query_id = str(query.get("query_id") or "")
        if query_id in existing_by_id:
            continue
        arm_id = str(query.get("arm_id") or "")
        if not arm_id:
            raise ValueError("every query must identify an experiment arm")
        estimated_seconds = float(query.get("estimated_review_seconds") or 0.0)
        elapsed = elapsed_by_arm.get(arm_id, 0.0)
        if elapsed >= budget_seconds_per_arm or (
            estimated_seconds > 0 and elapsed + estimated_seconds > budget_seconds_per_arm
        ):
            continue
        event = collect_one_feedback(
            query, annotator_id=annotator_id, input_fn=input_fn, clock=clock
        )
        elapsed_by_arm[arm_id] = elapsed_by_arm.get(arm_id, 0.0) + event["duration_seconds"]
        events.append(event)
        with destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    return events


def read_feedback(path: str | Path, *, expected_source: str = "live_human") -> list[dict[str, Any]]:
    events = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            if event.get("feedback_source") != expected_source:
                raise ValueError(f"non-live or unexpected feedback source at {path}:{line_number}")
            if event.get("judgment") not in set(JUDGMENTS.values()):
                raise ValueError(f"invalid judgment at {path}:{line_number}")
            duration = event.get("duration_seconds")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(f"invalid review duration at {path}:{line_number}")
            events.append(event)
    query_ids = [str(event.get("query_id") or "") for event in events]
    if not all(query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("feedback file contains missing or duplicate query IDs")
    return events
