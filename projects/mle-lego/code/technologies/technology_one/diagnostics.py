from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .harness import HarnessEdit, HarnessManifest


@dataclass(frozen=True)
class FailureDiagnostic:
    batch_id: str
    problem: str
    evidence: dict[str, Any]
    suspected_components: tuple[str, ...]
    candidate_edits: dict[str, HarnessEdit] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "problem": self.problem,
            "evidence": self.evidence,
            "suspected_components": list(self.suspected_components),
            "candidate_edits": {
                name: {
                    "component": edit.component,
                    "patch": edit.patch,
                    "rationale": edit.rationale,
                }
                for name, edit in self.candidate_edits.items()
            },
        }


def diagnose_batch(
    *,
    batch_id: str,
    manifest: HarnessManifest,
    source_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
) -> FailureDiagnostic | None:
    if selected_records:
        return None
    routes = {
        str(record.get("metadata", {}).get("route"))
        for record in source_records
    }
    configured_routes = set(manifest.components["generator"].config.get("routes", []))
    missing_routes = sorted(routes - configured_routes)
    low_quality = [
        float(record.get("metadata", {}).get("quality", 0.0))
        for record in source_records
    ]
    threshold = float(manifest.components["selector"].config.get("min_quality", 0.0))
    needed_threshold = min(low_quality) if low_quality else threshold
    edits: dict[str, HarnessEdit] = {}
    if missing_routes:
        edits["generator"] = HarnessEdit(
            "generator",
            {"routes": sorted(configured_routes | set(missing_routes))},
            "Add source routes observed in the failed batch.",
        )
    if low_quality and min(low_quality) < threshold:
        edits["selector"] = HarnessEdit(
            "selector",
            {"min_quality": needed_threshold},
            "Lower the selector threshold to retain the observed valid records.",
        )
    if not edits:
        return FailureDiagnostic(
            batch_id=batch_id,
            problem="No records selected and no safe component diagnosis was found.",
            evidence={"source_count": len(source_records)},
            suspected_components=(),
        )
    return FailureDiagnostic(
        batch_id=batch_id,
        problem="The current Harness selected no records from an available batch.",
        evidence={
            "source_count": len(source_records),
            "selected_count": len(selected_records),
            "missing_routes": missing_routes,
            "configured_routes": sorted(configured_routes),
            "selector_threshold": threshold,
        },
        suspected_components=tuple(edits),
        candidate_edits=edits,
    )

