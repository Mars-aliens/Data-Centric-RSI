from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .models import ArtifactRef, to_jsonable, utc_now


class ArtifactStore:
    """Immutable JSON artifacts plus an append-only event ledger."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.events_path = self.root / "events.jsonl"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _canonical_bytes(self, value: Any) -> bytes:
        return json.dumps(
            to_jsonable(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def put_json(
        self,
        *,
        kind: str,
        value: Any,
        version: str,
        parents: list[str] | None = None,
    ) -> ArtifactRef:
        payload = self._canonical_bytes(value)
        content_hash = hashlib.sha256(payload).hexdigest()
        artifact_id = f"{kind}-{content_hash[:16]}"
        path = self.artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            envelope = {
                "artifact_id": artifact_id,
                "kind": kind,
                "version": version,
                "content_hash": content_hash,
                "created_at": utc_now(),
                "parents": parents or [],
                "value": to_jsonable(value),
            }
            path.write_text(
                json.dumps(envelope, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            version=version,
            content_hash=content_hash,
        )

    def get_json(self, artifact_id: str) -> Any:
        path = self.artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return json.loads(path.read_text(encoding="utf-8"))["value"]

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "timestamp": utc_now(),
            "payload": to_jsonable(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write_run_snapshot(self, run_id: str, value: Any) -> None:
        runs_dir = self.root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{run_id}.json"
        path.write_text(
            json.dumps(to_jsonable(value), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
