from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import FeedbackRecord, utc_now
from .store import ArtifactStore


class InMemoryFeedbackProvider:
    """Deterministic local feedback provider for replay experiments."""

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        sequence: list[Any] | None = None,
    ):
        self.answers = answers or {}
        self.sequence = list(sequence or [])
        self.request_count = 0

    def request(self, *, target_artifact_id: str, target_version: str) -> FeedbackRecord:
        self.request_count += 1
        if self.request_count <= len(self.sequence):
            answer = self.sequence[self.request_count - 1]
        else:
            answer = self.answers.get(
                target_artifact_id,
                {
                    "decision": "accept",
                    "accepted_record_ids": [
                        "candidate-account",
                        "candidate-billing",
                        "candidate-access",
                        "candidate-invoice",
                    ],
                    "labels": {
                        "candidate-account": "account",
                        "candidate-billing": "billing",
                        "candidate-access": "account",
                        "candidate-invoice": "billing",
                    },
                },
            )
        return FeedbackRecord(
            feedback_id=f"feedback-{self.request_count:04d}",
            target_artifact_id=target_artifact_id,
            target_version=target_version,
            feedback_type="preference",
            value=answer,
            provider_type="human_replay",
            provider_version="local-replay-v0",
            permitted_use=("development_update", "training_data_selection"),
            requested_at=utc_now(),
            received_at=utc_now(),
        )


@dataclass
class FeedbackCatalog:
    """Local feedback registry for replayed or manually entered judgments."""

    records: dict[str, FeedbackRecord] = field(default_factory=dict)

    def add(
        self,
        *,
        feedback_id: str,
        target_artifact_id: str,
        target_version: str,
        feedback_type: str,
        value: Any,
        provider_type: str = "human",
        provider_version: str = "local-manual-v0",
        permitted_use: tuple[str, ...] = ("development_update",),
        requested_at: str | None = None,
        received_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        if feedback_id in self.records:
            raise ValueError(f"Feedback already exists: {feedback_id}")
        record = FeedbackRecord(
            feedback_id=feedback_id,
            target_artifact_id=target_artifact_id,
            target_version=target_version,
            feedback_type=feedback_type,
            value=value,
            provider_type=provider_type,
            provider_version=provider_version,
            permitted_use=permitted_use,
            requested_at=requested_at or utc_now(),
            received_at=received_at or utc_now(),
            metadata=metadata or {},
        )
        self.records[feedback_id] = record
        return record

    def for_artifact(self, artifact_id: str) -> list[FeedbackRecord]:
        return [
            record
            for record in self.records.values()
            if record.target_artifact_id == artifact_id
        ]

    def write(self, store: ArtifactStore, *, version: str = "feedback-v0") -> None:
        artifact = store.put_json(
            kind="feedback",
            value=list(self.records.values()),
            version=version,
        )
        store.append_event(
            "feedback_catalog_written",
            {"artifact": artifact, "count": len(self.records)},
        )
