from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TRACE_SCHEMA = "mle-lego-t3-feedback-trace-v1"
POLICIES = ("single_source_batch", "arrival_as_current", "source_attributed_window")


def _metric_index(summary: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    metrics: dict[tuple[int, str], dict[str, Any]] = {}
    for batch in summary.get("batch_history", []):
        for metric in batch.get("released_metrics", []):
            key = (int(metric["source_batch_index"]), str(metric["model_version_id"]))
            metrics[key] = dict(metric)
    for metric in summary.get("terminal_metrics", []):
        key = (int(metric["source_batch_index"]), str(metric["model_version_id"]))
        metrics[key] = dict(metric)
    return metrics


def trace_from_t1_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete":
        raise ValueError("T3 requires a completed T1 summary")
    metrics = _metric_index(summary)
    by_source: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (source_batch, version_id), metric in metrics.items():
        by_source[source_batch][version_id] = metric

    candidate_by_batch = {
        int(item["batch_index"]): item
        for item in summary.get("candidate_history", [])
        if isinstance(item, dict) and isinstance(item.get("batch_index"), int)
    }
    stable_version_by_batch = {
        int(item["batch_index"]): item.get("model_version_id")
        for item in summary.get("batch_history", [])
        if isinstance(item, dict) and isinstance(item.get("model_version_id"), str)
    }
    candidate_version_ids = {
        str(item["candidate_version_id"])
        for item in summary.get("candidate_history", [])
        if isinstance(item, dict) and isinstance(item.get("candidate_version_id"), str)
    }
    failed_candidate_by_batch = {
        int(item["batch_index"]): item
        for item in summary.get("candidate_history", [])
        if isinstance(item, dict)
        and isinstance(item.get("batch_index"), int)
        and item.get("status") not in {"eligible_for_observation", "eligible"}
    }

    comparisons: list[dict[str, Any]] = []
    feedback_events: list[dict[str, Any]] = []
    for source_batch in sorted(by_source):
        stable_version = stable_version_by_batch.get(source_batch)
        candidate_versions = candidate_version_ids.intersection(by_source[source_batch])
        if not stable_version or stable_version not in by_source[source_batch]:
            continue
        failed_item = failed_candidate_by_batch.get(source_batch)
        if not candidate_versions and failed_item is None:
            continue
        candidate_version = sorted(candidate_versions)[0] if candidate_versions else None
        stable_metric = by_source[source_batch][stable_version]
        candidate_metric = by_source[source_batch].get(candidate_version) if candidate_version else None
        stable_score = float(stable_metric["macro_f1"])
        candidate_score = (
            float(candidate_metric["macro_f1"]) if candidate_metric is not None else None
        )
        available = max(
            int(stable_metric["available_from_batch"]),
            int(candidate_metric["available_from_batch"])
            if candidate_metric is not None
            else int(stable_metric["available_from_batch"]),
        )
        comparison = {
            "source_batch_index": source_batch,
            "stable_version_id": stable_version,
            "candidate_version_id": candidate_version or failed_item.get("candidate_version_id"),
            "stable_score": stable_score,
            "candidate_score": candidate_score,
            "gain": candidate_score - stable_score if candidate_score is not None else None,
            "available_from_batch": available,
            "candidate_status": (
                failed_item.get("status", "failed")
                if candidate_metric is None
                else candidate_by_batch.get(source_batch, {}).get("status", "unknown")
            ),
        }
        comparisons.append(comparison)
        feedback_events.append(
            {
                "feedback_id": f"stable-{source_batch}-{stable_version}",
                "source_batch_index": source_batch,
                "available_from_batch": int(stable_metric["available_from_batch"]),
                "target_version_id": stable_version,
                "score": stable_score,
            }
        )
        if candidate_metric is not None:
            feedback_events.append(
                {
                    "feedback_id": f"candidate-{source_batch}-{candidate_version}",
                    "source_batch_index": source_batch,
                    "available_from_batch": int(candidate_metric["available_from_batch"]),
                    "target_version_id": candidate_version,
                    "score": candidate_score,
                }
            )
    if not comparisons:
        raise ValueError("T1 summary contains no source-aligned stable/candidate metric pairs")
    return {
        "schema": TRACE_SCHEMA,
        "source_run_id": summary.get("run_id"),
        "source_task_manifest_sha256": summary.get("task_manifest_sha256"),
        "source_config_sha256": summary.get("config_sha256"),
        "held_out_used": summary.get("held_out_used"),
        "final_test_used": summary.get("final_test_used"),
        "comparisons": comparisons,
        "feedback_events": sorted(
            feedback_events,
            key=lambda item: (
                item["available_from_batch"], item["source_batch_index"], item["feedback_id"]
            ),
        ),
    }


def evaluate_trace(
    trace: dict[str, Any],
    *,
    window_batches: int = 3,
    min_mean_gain: float = 0.0,
    max_regression: float = 0.05,
) -> dict[str, Any]:
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"trace schema must be {TRACE_SCHEMA}")
    if trace.get("held_out_used") is not False or trace.get("final_test_used") is not False:
        raise ValueError("T3 trace must not use held-out or final-test information")
    if window_batches < 1 or min_mean_gain < 0 or max_regression < 0:
        raise ValueError("invalid T3 decision thresholds")
    comparisons = sorted(trace.get("comparisons", []), key=lambda item: item["source_batch_index"])
    events = sorted(
        trace.get("feedback_events", []),
        key=lambda item: (item["available_from_batch"], item["source_batch_index"], item["feedback_id"]),
    )
    if not comparisons:
        raise ValueError("trace must contain paired source-batch comparisons")
    for item in comparisons:
        values = (item.get("stable_score"), item.get("candidate_score"), item.get("gain"))
        if item.get("candidate_status") in {"failed", "execution_error", "rejected"}:
            if not isinstance(item.get("stable_score"), (int, float)) or not math.isfinite(item["stable_score"]):
                raise ValueError("stable comparison score must be finite")
            continue
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("comparison scores must be finite numbers")
    first = comparisons[0]
    first_action = "accept_candidate" if first["gain"] > min_mean_gain else "keep_stable"

    current_version = first["stable_version_id"]
    arrival_misattributed: list[str] = []
    first_arrival_decision: str | None = None
    latest_stable_score: float | None = None
    for event in events:
        if event.get("target_version_id") != current_version:
            arrival_misattributed.append(str(event.get("feedback_id")))
        if event.get("target_version_id") == first["stable_version_id"]:
            latest_stable_score = float(event["score"])
        elif first.get("candidate_version_id") and event.get("target_version_id") == first["candidate_version_id"]:
            if first_arrival_decision is None and latest_stable_score is not None:
                first_arrival_decision = (
                    "accept_candidate"
                    if float(event["score"]) - latest_stable_score > min_mean_gain
                    else "keep_stable"
                )
                if first_arrival_decision == "accept_candidate":
                    current_version = first["candidate_version_id"]

    window = comparisons[:window_batches]
    gains = [float(item["gain"]) for item in window if item.get("gain") is not None]
    mean_gain = sum(gains) / len(gains) if gains else None
    failed = any(str(item.get("candidate_status", "")).lower() in {"failed", "execution_error", "rejected"} for item in window)
    if failed:
        window_action = "reject_failed_candidate"
        reason = "candidate_failed_during_observation"
        reported_mean_gain = None
    elif len(window) < window_batches:
        window_action = "insufficient_source_aligned_batches"
        reason = "insufficient_source_aligned_batches"
        reported_mean_gain = mean_gain
    elif len(window) < window_batches or len(gains) < window_batches:
        window_action = "insufficient_source_aligned_batches"
        reason = "insufficient_source_aligned_batches"
    elif mean_gain is not None and mean_gain > min_mean_gain and min(gains) >= -max_regression:
        window_action = "accept_candidate"
        reason = "paired_source_window_passed"
        reported_mean_gain = mean_gain
    else:
        window_action = "keep_stable"
        reason = "paired_source_window_did_not_pass"
        reported_mean_gain = mean_gain

    return {
        "schema": "mle-lego-t3-evaluation-v1",
        "source_run_id": trace.get("source_run_id"),
        "source_task_manifest_sha256": trace.get("source_task_manifest_sha256"),
        "source_config_sha256": trace.get("source_config_sha256"),
        "policies": {
            "single_source_batch": {
                "action": first_action,
                "observed_source_batches": 1,
                "gain": first["gain"],
                "decision_available_from_batch": first["available_from_batch"],
            },
            "arrival_as_current": {
                "action": first_arrival_decision or "insufficient_arrivals",
                "misattributed_feedback_count": len(arrival_misattributed),
                "misattributed_feedback_ids": arrival_misattributed,
                "warning": "Diagnostic unsafe baseline: feedback is evaluated against the currently active version, not its source version.",
            },
            "source_attributed_window": {
                "action": window_action,
                "reason": reason,
                "observed_source_batches": len(window),
                "window_batches": window_batches,
                "decision_available_from_batch": max(
                    (int(item["available_from_batch"]) for item in window), default=None
                ),
                "mean_gain": reported_mean_gain,
                "worst_batch_gain": min(gains) if gains else None,
                "candidate_failed": failed,
            },
        },
        "source_aligned_comparisons": comparisons,
        "feedback_event_count": len(events),
        "held_out_used": False,
        "final_test_used": False,
        "interpretation": "Offline replay recommendation only; this evaluator does not promote or switch the live Task Model.",
    }


def evaluate_future_window(
    trace: dict[str, Any],
    *,
    decision_window_batches: int = 3,
    outcome_window_batches: int = 3,
    min_mean_gain: float = 0.0,
    max_regression: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a selection recommendation on later, non-overlapping source batches."""
    evaluation = evaluate_trace(
        trace,
        window_batches=decision_window_batches,
        min_mean_gain=min_mean_gain,
        max_regression=max_regression,
    )
    comparisons = sorted(trace["comparisons"], key=lambda item: item["source_batch_index"])
    decision_window = comparisons[:decision_window_batches]
    outcome_window = comparisons[
        decision_window_batches : decision_window_batches + outcome_window_batches
    ]
    complete_outcome = len(outcome_window) == outcome_window_batches and all(
        item.get("candidate_score") is not None
        and item.get("candidate_status") not in {"failed", "execution_error", "rejected"}
        for item in outcome_window
    )
    outcome_gains = [float(item["gain"]) for item in outcome_window] if complete_outcome else []
    outcome_mean = sum(outcome_gains) / len(outcome_gains) if outcome_gains else None
    outcome_success = (
        outcome_mean is not None
        and outcome_mean > min_mean_gain
        and min(outcome_gains) >= -max_regression
    ) if complete_outcome else None
    decision_available = max(
        (int(item["available_from_batch"]) for item in decision_window), default=None
    )
    outcome_available = max(
        (int(item["available_from_batch"]) for item in outcome_window), default=None
    ) if complete_outcome else None

    recommendations: dict[str, Any] = {}
    for name in ("single_source_batch", "arrival_as_current", "source_attributed_window"):
        action = evaluation["policies"][name]["action"]
        can_score = outcome_success is not None and action in {"accept_candidate", "keep_stable"}
        recommendations[name] = {
            "action": action,
            "outcome_success": outcome_success,
            "false_accept": bool(action == "accept_candidate" and outcome_success is False)
            if can_score else None,
            "false_reject": bool(action == "keep_stable" and outcome_success is True)
            if can_score else None,
            "scorable": can_score,
        }

    return {
        "schema": "mle-lego-future-window-evaluation-v1",
        "source_run_id": trace.get("source_run_id"),
        "decision_window_batches": decision_window_batches,
        "outcome_window_batches": outcome_window_batches,
        "decision_source_batches": [item["source_batch_index"] for item in decision_window],
        "outcome_source_batches": [item["source_batch_index"] for item in outcome_window],
        "decision_available_from_batch": decision_available,
        "outcome_available_from_batch": outcome_available,
        "outcome_complete": complete_outcome,
        "outcome_mean_gain": outcome_mean,
        "outcome_worst_batch_gain": min(outcome_gains) if outcome_gains else None,
        "outcome_success": outcome_success,
        "recommendations": recommendations,
        "held_out_used": False,
        "final_test_used": False,
        "interpretation": (
            "Offline prospective validation on a non-overlapping future source window; "
            "does not perform online promotion or rollback."
        ),
    }


def load_t1_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read T1 summary: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("T1 summary must be a JSON object")
    return value
