from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


ARMS = (
    "no_experience",
    "reflection_only",
    "executable_unmanaged",
    "lifecycle_governed",
)


def _regime(records: list[dict[str, Any]]) -> dict[str, str]:
    if any(item.get("target") not in {"preferred", "rejected", "tie"} for item in records):
        raise ValueError("experience policy received an unsupported released label")
    counts = Counter(str(item["target"]) for item in records)
    nonzero = [count for count in counts.values() if count]
    ratio = max(nonzero) / min(nonzero) if nonzero else 1.0
    band = "balanced" if ratio < 1.5 else "moderate" if ratio < 2.5 else "high"
    dominant = sorted(counts, key=lambda label: (-counts[label], label))[0] if counts else "none"
    return {"dominant_label": dominant, "imbalance_band": band}


def _cap_majority(records: list[dict[str, Any]], ratio: int = 2) -> list[dict[str, Any]]:
    counts = Counter(str(item["target"]) for item in records)
    nonzero = [count for count in counts.values() if count]
    if len(nonzero) < 2:
        return list(records)
    cap = min(nonzero) * ratio
    kept_by_label: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for item in reversed(records):
        label = str(item["target"])
        if kept_by_label[label] >= cap:
            continue
        kept_by_label[label] += 1
        kept.append(item)
    return list(reversed(kept))


class ExperienceTrainingPolicy:
    """Deterministic T2 experiment arms over cumulative released training data."""

    def __init__(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {', '.join(ARMS)}")
        self.arm = arm

    def run_signature(self) -> dict[str, str]:
        return {"technology": "T2", "arm": self.arm, "policy_version": "t2-experience-v1"}

    def run_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"arm": self.arm, "policy_version": "t2-experience-v1", **state}

    def prepare(
        self,
        records: list[dict[str, Any]],
        *,
        newly_released: list[dict[str, Any]],
        released_metrics: list[dict[str, Any]],
        batch_index: int,
        state: dict[str, Any],
        task_manifest: dict[str, Any],
        run_id: str,
        source_version_id: str,
        source_batch_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        next_state = dict(state)
        next_state.setdefault("reflections", [])
        next_state.setdefault("experience", None)
        next_state.setdefault("revalidation_count", 0)
        observed = _regime(newly_released) if newly_released else None
        experience = next_state.get("experience")
        experience_before = dict(experience) if isinstance(experience, dict) else None
        conditions_before = (
            dict(experience_before.get("conditions", {}))
            if experience_before is not None
            else None
        )
        status_before = experience_before.get("status") if experience_before else None
        activated = False
        context_changed = False
        if newly_released and self.arm == "lifecycle_governed" and experience is not None:
            current_conditions = {
                "dataset_revision": task_manifest.get("dataset_revision"),
                **observed,
            }
            if experience.get("pending_conditions") == current_conditions:
                if experience.get("pending_condition_batch_index") != batch_index:
                    experience = {
                        **experience,
                        "pending_condition_batches": experience.get("pending_condition_batches", 0) + 1,
                        "pending_condition_batch_index": batch_index,
                    }
                    next_state["experience"] = experience
            elif experience["conditions"] != current_conditions:
                context_changed = True
                experience = {
                    **experience,
                    "status": "needs_validation",
                    "pending_conditions": current_conditions,
                    "pending_condition_batches": 1,
                    "pending_condition_batch_index": batch_index,
                }
                next_state["experience"] = experience

        if self.arm == "reflection_only" and newly_released:
            counts = Counter(item["target"] for item in newly_released)
            next_state["reflections"] = [
                *next_state["reflections"],
                {
                    "batch_index": batch_index,
                    "label_counts": dict(sorted(counts.items())),
                    "action": "none",
                },
            ]
            if len(next_state["reflections"]) > 32:
                removed = len(next_state["reflections"]) - 32
                next_state["reflections"] = next_state["reflections"][-32:]
                next_state["archived_reflection_count"] = (
                    next_state.get("archived_reflection_count", 0) + removed
                )

        if self.arm in {"executable_unmanaged", "lifecycle_governed"} and newly_released:
            if experience is None:
                experience = {
                    "experience_id": f"t2-replay-cap-{batch_index:06d}",
                    "kind": "skill",
                    "content": {"operation": "cap_majority_per_label_ratio", "max_ratio": 2},
                    "conditions": {
                        "dataset_revision": task_manifest.get("dataset_revision"),
                        **observed,
                    },
                    "source_batch_index": batch_index,
                    "source_batch_id": source_batch_id,
                    "source_run_id": run_id,
                    "source_version_id": source_version_id,
                    "source_batch_count": 1,
                    "pending_conditions": (
                        {
                            "dataset_revision": task_manifest.get("dataset_revision"),
                            **observed,
                        }
                        if self.arm == "lifecycle_governed"
                        else None
                    ),
                    "pending_condition_batches": 1 if self.arm == "lifecycle_governed" else 0,
                    "pending_condition_batch_index": batch_index,
                    "status": "verified" if self.arm == "executable_unmanaged" else "candidate",
                }
                next_state["experience"] = experience

            if self.arm == "lifecycle_governed":
                if experience["source_batch_count"] < 2 and experience["source_batch_index"] != batch_index:
                    experience = {**experience, "source_batch_count": experience["source_batch_count"] + 1}
                if (
                    experience["status"] in {"candidate", "needs_validation"}
                    and experience["source_batch_count"] >= 2
                    and not context_changed
                    and experience.get("pending_condition_batches", 0) >= 2
                ):
                    selected = _cap_majority(records, 2)
                    labels_before = {item["target"] for item in records}
                    labels_after = {item["target"] for item in selected}
                    if labels_before == labels_after and len(selected) > 0:
                        experience = {
                            **experience,
                            "conditions": {
                                "dataset_revision": task_manifest.get("dataset_revision"),
                                **observed,
                            },
                            "pending_conditions": None,
                            "pending_condition_batches": 0,
                            "pending_condition_batch_index": None,
                            "status": "verified",
                            "validation": {
                                "method": "label_coverage_after_selection",
                                "validated_at_batch": batch_index,
                                "evidence_batches": experience["source_batch_count"],
                            },
                        }
                        next_state["revalidation_count"] += 1
                next_state["experience"] = experience

            activated = experience["status"] == "verified" and (
                self.arm != "lifecycle_governed"
                or experience["conditions"] == {
                    "dataset_revision": task_manifest.get("dataset_revision"),
                    **observed,
                }
            )

        selected_records = records
        if activated:
            selected_records = _cap_majority(records, 2)

        audit = {
            "arm": self.arm,
            "batch_index": batch_index,
            "observed_regime": observed,
            "conditions_before": conditions_before,
            "conditions_observed": (
                {"dataset_revision": task_manifest.get("dataset_revision"), **observed}
                if observed is not None
                else None
            ),
            "context_changed": context_changed,
            "experience_status_before": status_before,
            "experience_id": experience.get("experience_id") if experience else None,
            "experience_status": experience.get("status") if experience else None,
            "experience_source_batch_index": (
                experience.get("source_batch_index") if experience else None
            ),
            "experience_source_version_id": (
                experience.get("source_version_id") if experience else None
            ),
            "experience_applied": activated,
            "released_metrics_observed": len(released_metrics),
            "records_before": len(records),
            "records_selected": len(selected_records),
            "records_dropped": len(records) - len(selected_records),
            "record_ids_before": [item["record_id"] for item in records],
            "record_ids_selected": [item["record_id"] for item in selected_records],
            "record_ids_selected_sha256": hashlib.sha256(
                json.dumps(
                    sorted(item["record_id"] for item in selected_records),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        return selected_records, audit, next_state
