from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion
from core_system.mle_core.store import ArtifactStore

from .candidate_gate import (
    CandidateGateStatus,
    CandidateSafetyGate,
    SafetyChecklist,
)
from .candidate_proposal import read_candidate_manifest
from .candidate_sandbox import (
    KubernetesJobCandidateSandbox,
    SandboxMount,
    SandboxResult,
    hash_directory,
)
from .long_horizon_task import LABELS


_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_COST_FIELDS = {
    "training_steps",
    "training_records_seen",
    "training_seconds",
    "prediction_records",
    "prediction_seconds",
    "llm_calls",
    "llm_tokens",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _checkpoint_version(checkpoint: Mapping[str, Any]) -> str:
    version = checkpoint.get("task_model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("checkpoint must contain a task_model_version")
    return version


class LongHorizonCandidateShadow:
    """Run one candidate branch beside the trusted long-horizon runner."""

    def __init__(
        self,
        *,
        candidate_id: str,
        candidate_dir: str | Path,
        output_dir: str | Path,
        sandbox: KubernetesJobCandidateSandbox,
        command: Sequence[str],
        store: ArtifactStore,
        component_versions: Mapping[str, str] | None = None,
        task_config_overrides: Mapping[str, Any] | None = None,
        environment: Mapping[str, str] | None = None,
        max_reported_training_steps: int | None = None,
        require_proposal_manifest: bool = False,
    ):
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError("candidate_id must contain only letters, numbers, '.', '_' or '-'")
        self.candidate_id = candidate_id
        self.candidate_dir = Path(candidate_dir).resolve(strict=True)
        if not self.candidate_dir.is_dir():
            raise ValueError("candidate_dir must be a directory")
        manifest_path = self.candidate_dir / "candidate_proposal.json"
        self.proposal_manifest: dict[str, Any] | None = None
        self.proposal_manifest_sha256: str | None = None
        if manifest_path.exists() or require_proposal_manifest:
            self.proposal_manifest = read_candidate_manifest(self.candidate_dir)
            if self.proposal_manifest.get("candidate_id") != candidate_id:
                raise ValueError("candidate_id does not match candidate_proposal.json")
            self.proposal_manifest_sha256 = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        self.require_proposal_manifest = require_proposal_manifest
        self.candidate_artifact_hash = hash_directory(self.candidate_dir)
        if getattr(sandbox, "backend_name", None) != "kubernetes":
            raise ValueError(
                "candidate sandbox backend must be kubernetes; "
                "local and non-Kubernetes candidate execution is not allowed"
            )
        self.sandbox = sandbox
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.command = tuple(command)
        if not self.command or any(not item for item in self.command):
            raise ValueError("candidate command must be a non-empty argv sequence")
        self.store = store
        self.gate = CandidateSafetyGate(store)
        self.environment = dict(environment or {})
        self.max_reported_training_steps = max_reported_training_steps
        self.component_versions = dict(component_versions or {})
        allowed_component_versions = {
            "data_version",
            "task_model_version",
            "harness_version",
            "memory_version",
            "skill_version",
            "evaluator_version",
        }
        unknown_component_versions = set(self.component_versions) - allowed_component_versions
        if unknown_component_versions:
            raise ValueError(
                f"unsupported candidate component version fields: {sorted(unknown_component_versions)}"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.component_versions.values()
        ):
            raise ValueError("candidate component versions must be non-empty strings")
        self.task_config_overrides = dict(task_config_overrides or {})
        if {"base_model_id", "base_model_path", "base_model_revision"}.intersection(
            self.task_config_overrides
        ):
            raise ValueError("candidate overrides cannot replace the frozen base model identity")
        self.candidate_hash = _hash_json(
            {
                "candidate_artifact_sha256": self.candidate_artifact_hash,
                "command": list(self.command),
                "component_versions": self.component_versions,
                "sandbox_backend": self.sandbox.backend_name,
                "sandbox_image": self.sandbox.image,
                "task_config_overrides": self.task_config_overrides,
            }
        )
        self.latest_path = self.root / "latest.json"
        self.attempts_dir = self.root / "attempts"
        self.attempts_dir.mkdir(exist_ok=True)
        self._stable_version: DataSystemVersion | None = None
        self._candidate_version: CandidateVersion | None = None
        self._gate_status: str | None = None
        self._terminal_failure: str | None = None
        self._proposal_base_version_binding: str | None = None
        self._trials: dict[str, dict[str, Any]] = {}
        self._restore_metadata()

    def run_signature(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "candidate_artifact_sha256": self.candidate_artifact_hash,
            "proposal_manifest_sha256": self.proposal_manifest_sha256,
            "require_proposal_manifest": self.require_proposal_manifest,
            "sandbox_backend": self.sandbox.backend_name,
            "sandbox_image": self.sandbox.image,
            "sandbox_configuration": (
                self.sandbox.run_configuration()
                if isinstance(self.sandbox, KubernetesJobCandidateSandbox)
                else {}
            ),
            "fault_injection": {
                str(key): value
                for key, value in sorted(self.sandbox.fault_injection.items())
            },
            "sandbox_limits": {
                "timeout_seconds": self.sandbox.limits.timeout_seconds,
                "memory_gib": self.sandbox.limits.memory_gib,
                "cpu_cores": self.sandbox.limits.cpu_cores,
                "pids_limit": self.sandbox.limits.pids_limit,
                "tmpfs_mib": self.sandbox.limits.tmpfs_mib,
                "gpu_devices": self.sandbox.limits.gpu_devices,
            },
            "command": list(self.command),
            "component_versions": self.component_versions,
            "task_config_overrides_sha256": _hash_json(self.task_config_overrides),
            "environment_sha256": _hash_json(self.environment),
        }

    def _restore_metadata(self) -> None:
        if not self.latest_path.is_file():
            return
        state = json.loads(self.latest_path.read_text(encoding="utf-8"))
        if state.get("candidate_id") != self.candidate_id:
            raise ValueError("candidate state belongs to a different candidate_id")
        if state.get("candidate_hash") != self.candidate_hash:
            raise ValueError("candidate artifacts changed since the run was created")
        self._gate_status = state.get("gate_status")
        self._terminal_failure = state.get("terminal_failure")
        self._proposal_base_version_binding = state.get("proposal_base_version_binding")
        raw_trials = state.get("trials", {})
        if not isinstance(raw_trials, dict):
            raise ValueError("candidate state trial ledger is invalid")
        self._trials = {
            str(key): value for key, value in raw_trials.items() if isinstance(value, dict)
        }
        stable = state.get("stable_version")
        candidate = state.get("candidate_version")
        if isinstance(stable, dict):
            self._stable_version = DataSystemVersion.from_dict(stable)
        if isinstance(candidate, dict):
            self._candidate_version = CandidateVersion(
                candidate_id=candidate["candidate_id"],
                version=DataSystemVersion.from_dict(candidate["version"]),
                created_run_id=candidate["created_run_id"],
                created_batch_id=candidate["created_batch_id"],
            )

    def _save_metadata(
        self,
        *,
        checkpoint: dict[str, Any] | None = None,
        trial: dict[str, Any] | None = None,
    ) -> None:
        current: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "gate_status": self._gate_status,
            "terminal_failure": self._terminal_failure,
            "proposal_manifest_sha256": self.proposal_manifest_sha256,
            "proposal_base_version_binding": self._proposal_base_version_binding,
            "stable_version": (
                self._stable_version.to_dict() if self._stable_version is not None else None
            ),
            "candidate_version": (
                {
                    "candidate_id": self._candidate_version.candidate_id,
                    "version": self._candidate_version.version.to_dict(),
                    "created_run_id": self._candidate_version.created_run_id,
                    "created_batch_id": self._candidate_version.created_batch_id,
                }
                if self._candidate_version is not None
                else None
            ),
        }
        if checkpoint is not None:
            current["checkpoint"] = checkpoint
        elif self.latest_path.is_file():
            previous = json.loads(self.latest_path.read_text(encoding="utf-8"))
            current["checkpoint"] = previous.get("checkpoint")
        trials = dict(self._trials)
        if trial is not None:
            trials[str(trial["batch_index"])] = trial
        current["trials"] = trials
        _atomic_json(self.latest_path, current)
        self._trials = trials

    def _version_pair(
        self, *, checkpoint: dict[str, Any], task_manifest_hash: str, run_id: str, batch_id: str
    ) -> tuple[DataSystemVersion, CandidateVersion]:
        model_version = _checkpoint_version(checkpoint)
        stable_id = f"stable-{_hash_json({'task': task_manifest_hash, 'model': model_version})[:16]}"
        stable = DataSystemVersion(
            version_id=stable_id,
            data_version=task_manifest_hash,
            task_model_version=model_version,
            harness_version="fixed-harness-v1",
            memory_version="empty-memory-v1",
            skill_version="empty-skill-v1",
            evaluator_version="helpsteer2-prequential-v1",
        )
        if self.proposal_manifest is not None:
            declared_parent = self.proposal_manifest["base_version_id"]
            if declared_parent not in {"runtime-stable", stable.version_id}:
                raise ValueError(
                    "candidate proposal base_version_id does not match the runtime stable version"
                )
            if (
                self._proposal_base_version_binding is not None
                and self._proposal_base_version_binding != stable.version_id
            ):
                raise ValueError("candidate proposal is already bound to a different stable version")
            self._proposal_base_version_binding = stable.version_id
        versions = {
            "data_version": stable.data_version,
            "task_model_version": stable.task_model_version,
            "harness_version": self.candidate_hash,
            "memory_version": stable.memory_version,
            "skill_version": stable.skill_version,
            "evaluator_version": stable.evaluator_version,
        }
        versions.update(self.component_versions)
        candidate_version = DataSystemVersion(
            version_id=f"candidate-{self.candidate_id}-{self.candidate_hash[:12]}",
            parent_version_id=stable.version_id,
            tool_versions=(("candidate-container", self.sandbox.image),),
            **versions,
        )
        return stable, CandidateVersion(
            candidate_id=self.candidate_id,
            version=candidate_version,
            created_run_id=run_id,
            created_batch_id=batch_id,
        )

    @staticmethod
    def _public_records(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw = batch.get("records")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("candidate batch must contain records")
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in raw:
            if not isinstance(record, Mapping):
                raise ValueError("candidate batch record must be an object")
            if "target" in record or "scores" in record:
                raise ValueError("candidate input must not contain labels or scores")
            record_id, input_text = record.get("record_id"), record.get("input")
            if not isinstance(record_id, str) or not record_id or record_id in seen:
                raise ValueError("candidate batch has an invalid or duplicate record_id")
            if not isinstance(input_text, str) or not input_text:
                raise ValueError(f"candidate input is missing for {record_id}")
            seen.add(record_id)
            result.append({"record_id": record_id, "input": input_text})
        return result

    @staticmethod
    def _released_feedback(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in items:
            target = item.get("target")
            if target not in LABELS:
                raise ValueError("released feedback contains an unsupported target")
            result.append(
                {
                    "feedback_id": item["feedback_id"],
                    "record_id": item["record_id"],
                    "source_batch_index": item["source_batch_index"],
                    "input": item["input"],
                    "target": target,
                }
            )
        return result

    def _checkpoint(self, stable_checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        if self.latest_path.is_file():
            latest = json.loads(self.latest_path.read_text(encoding="utf-8"))
            current = latest.get("checkpoint")
            if isinstance(current, dict):
                return current
        return dict(stable_checkpoint)

    def _remove_attempt_for_adapter(self, adapter_path: str | Path) -> None:
        attempts_root = self.attempts_dir.resolve()
        adapter = Path(adapter_path).resolve()
        if attempts_root not in adapter.parents:
            return
        attempt_dir = adapter.parent
        if attempts_root not in attempt_dir.parents or not attempt_dir.exists():
            return
        shutil.rmtree(attempt_dir)

    def _remove_attempt_directory(self, attempt_dir: Path) -> None:
        attempts_root = self.attempts_dir.resolve()
        resolved = attempt_dir.resolve()
        if attempts_root not in resolved.parents or not resolved.exists():
            return
        shutil.rmtree(resolved)

    @staticmethod
    def _validate_cost(raw: object, *, max_training_steps: int | None) -> dict[str, int | float]:
        if not isinstance(raw, dict):
            raise ValueError("candidate cost must be an object")
        unknown = set(raw) - _COST_FIELDS
        if unknown:
            raise ValueError(f"candidate cost contains unknown fields: {sorted(unknown)}")
        cost: dict[str, int | float] = {}
        for key, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"candidate cost {key} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"candidate cost {key} must be finite and non-negative")
            if key in {
                "training_steps",
                "training_records_seen",
                "prediction_records",
                "llm_calls",
                "llm_tokens",
            }:
                if not float(value).is_integer():
                    raise ValueError(f"candidate cost {key} must be an integer")
                value = int(value)
            cost[key] = value
        if max_training_steps is not None and cost.get("training_steps", 0) > max_training_steps:
            raise ValueError("candidate exceeded its per-batch training-step budget")
        return cost

    @staticmethod
    def _normalize_checkpoint(
        raw: object,
        *,
        output_dir: Path,
        previous: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("candidate result must include a checkpoint object")
        checkpoint = dict(raw)
        if checkpoint.get("base_model_path") != "/models/task":
            raise ValueError("candidate checkpoint must use the mounted task-model path")
        if checkpoint.get("base_model_revision") != previous.get("base_model_revision"):
            raise ValueError("candidate checkpoint changed the frozen base-model revision")
        if checkpoint.get("backend") != previous.get("backend"):
            raise ValueError("candidate checkpoint changed the task-model backend")
        parent_version = _checkpoint_version(previous)
        next_version = _checkpoint_version(checkpoint)
        if next_version != parent_version:
            expected_parent = {
                "task_model_version": parent_version,
                "adapter_hash": previous.get("adapter_hash"),
            }
            if checkpoint.get("parent_checkpoint") != expected_parent:
                raise ValueError("candidate checkpoint parent lineage is invalid")
        adapter = checkpoint.get("adapter_path")
        if not isinstance(adapter, str):
            raise ValueError("candidate checkpoint must include adapter_path")
        if adapter == "/parent-checkpoint":
            actual = Path(str(previous["adapter_path"])).resolve(strict=True)
        elif adapter.startswith("/candidate-output/"):
            relative = Path(adapter.removeprefix("/candidate-output/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("candidate checkpoint path escapes candidate output")
            actual = (output_dir / relative).resolve(strict=True)
            if output_dir.resolve() not in actual.parents:
                raise ValueError("candidate checkpoint path escapes candidate output")
        else:
            raise ValueError("candidate checkpoint must reference its read-only parent or output")
        if not actual.is_dir():
            raise ValueError("candidate checkpoint adapter directory is missing")
        actual_hash = hash_directory(actual)
        if checkpoint.get("adapter_hash") != actual_hash:
            raise ValueError("candidate checkpoint adapter hash does not match its files")
        checkpoint["adapter_path"] = str(actual)
        return checkpoint

    def run_batch(
        self,
        *,
        run_id: str,
        batch: Mapping[str, Any],
        released_feedback: Sequence[Mapping[str, Any]],
        seed_examples: Sequence[Mapping[str, Any]],
        stable_checkpoint: Mapping[str, Any],
        task_config: Mapping[str, Any],
        task_manifest_hash: str,
    ) -> dict[str, Any]:
        batch_id = str(batch.get("batch_id", ""))
        batch_index = int(batch.get("batch_index", -1))
        if not batch_id or batch_index < 0:
            raise ValueError("candidate batch requires batch_id and non-negative batch_index")
        if str(batch_index) in self._trials:
            return self._trials[str(batch_index)]
        if self._terminal_failure is not None or self._gate_status == CandidateGateStatus.REJECTED.value:
            return {
                "candidate_id": self.candidate_id,
                "candidate_version_id": (
                    self._candidate_version.version.version_id
                    if self._candidate_version is not None
                    else None
                ),
                "status": "not_run_terminal",
                "failure_code": self._terminal_failure or "safety_gate_rejected",
                "predictions": None,
            }

        records = self._public_records(batch)
        feedback = self._released_feedback(released_feedback)
        checkpoint = self._checkpoint(stable_checkpoint)
        adapter_path = Path(str(checkpoint.get("adapter_path", ""))).resolve(strict=True)
        base_model_path = Path(str(task_config.get("base_model_path", ""))).resolve(strict=True)
        stable_adapter_path = Path(str(stable_checkpoint["adapter_path"])).resolve(strict=True)
        stable_hash_before = hash_directory(stable_adapter_path)
        parent_hash_before = hash_directory(adapter_path)

        if self._candidate_version is None:
            try:
                stable_version, candidate_version = self._version_pair(
                    checkpoint=dict(stable_checkpoint),
                    task_manifest_hash=task_manifest_hash,
                    run_id=run_id,
                    batch_id=batch_id,
                )
            except ValueError as exc:
                self._gate_status = CandidateGateStatus.REJECTED.value
                self._terminal_failure = f"proposal_parent_mismatch: {exc}"
                trial = {
                    "candidate_id": self.candidate_id,
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "status": "rejected",
                    "failure_code": "proposal_parent_mismatch",
                    "error": str(exc),
                    "predictions": None,
                }
                self._save_metadata(trial=trial)
                _atomic_json(self.root / "trials" / f"batch-{batch_index:06d}.json", trial)
                self.store.append_event(
                    "candidate_proposal_rejected",
                    trial,
                    event_id=f"{run_id}:{batch_id}:{self.candidate_id}:parent-mismatch",
                )
                return trial
            self._stable_version = stable_version
            self._candidate_version = candidate_version
        else:
            stable_version = self._stable_version
            candidate_version = self._candidate_version
            assert stable_version is not None

        output_dir = self.attempts_dir / f"batch-{batch_index:06d}-{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=False)

        payload = {
            "protocol": "lego-t1-candidate-step-v1",
            "candidate_id": self.candidate_id,
            "candidate_version_id": candidate_version.version.version_id,
            "parent_version_id": candidate_version.version.parent_version_id,
            "batch_id": batch_id,
            "batch_index": batch_index,
            "records": records,
            "seed_examples": [dict(item) for item in seed_examples],
            "released_feedback": feedback,
            "checkpoint": {
                **checkpoint,
                "base_model_path": "/models/task",
                "adapter_path": "/parent-checkpoint",
            },
            "config": {
                **dict(task_config),
                **self.task_config_overrides,
                "base_model_path": "/models/task",
            },
            "seed": int(task_config.get("seed", 17)) + batch_index + 1,
        }
        input_hash = _hash_json(payload)
        result = self.sandbox.execute(
            command=self.command,
            payload=payload,
            mounts=(
                SandboxMount(self.candidate_dir, "/candidate", read_only=True),
                SandboxMount(adapter_path, "/parent-checkpoint", read_only=True),
                SandboxMount(base_model_path, "/models/task", read_only=True),
            ),
            output_dir=output_dir,
            environment=self.environment,
        )
        stable_hash_after = hash_directory(stable_adapter_path)
        parent_hash_after = hash_directory(adapter_path)
        response = result.response
        response_error: str | None = None
        predictions: dict[str, str] | None = None
        next_checkpoint: dict[str, Any] | None = None
        cost: dict[str, int | float] = {}
        schema_valid = False
        data_valid = False
        lineage_valid = False
        try:
            if result.status != "completed" or response is None:
                raise ValueError(result.error or result.failure_code or "candidate execution failed")
            if response.get("protocol") != "lego-t1-candidate-step-v1":
                raise ValueError("candidate returned an unsupported protocol")
            if response.get("candidate_id") != self.candidate_id:
                raise ValueError("candidate response ID does not match the run")
            if response.get("parent_version_id") != candidate_version.version.parent_version_id:
                raise ValueError("candidate response parent version does not match its manifest")
            schema_valid = True
            raw_predictions = response.get("predictions")
            if not isinstance(raw_predictions, list):
                raise ValueError("candidate predictions must be a list")
            predictions = {}
            for item in raw_predictions:
                if not isinstance(item, dict):
                    raise ValueError("candidate prediction must be an object")
                record_id, label = item.get("record_id"), item.get("prediction")
                if not isinstance(record_id, str) or record_id in predictions:
                    raise ValueError("candidate predictions have invalid or duplicate record IDs")
                if label not in LABELS:
                    raise ValueError(f"candidate returned an unsupported label: {label!r}")
                predictions[record_id] = label
            expected_ids = {item["record_id"] for item in records}
            if set(predictions) != expected_ids:
                raise ValueError("candidate predictions do not cover exactly the current batch")
            data_valid = True
            cost = self._validate_cost(
                response.get("cost"),
                max_training_steps=self.max_reported_training_steps,
            )
            if cost.get("prediction_records") != len(records):
                raise ValueError("candidate prediction record cost does not match the batch")
            next_checkpoint = self._normalize_checkpoint(
                response.get("checkpoint"),
                output_dir=output_dir,
                previous=checkpoint,
            )
            lineage_valid = True
        except (KeyError, OSError, TypeError, ValueError) as exc:
            response_error = str(exc)
            predictions = None

        budget_ok = (
            result.status == "completed"
            and result.elapsed_seconds <= self.sandbox.limits.timeout_seconds
            and response_error is None
        )
        isolated = stable_hash_before == stable_hash_after and parent_hash_before == parent_hash_after
        evidence = {
            **result.to_dict(),
            "sandbox_backend": result.backend,
            "candidate_artifact_sha256": self.candidate_artifact_hash,
            "proposal_manifest_sha256": self.proposal_manifest_sha256,
            "proposal_base_version_id": (
                self.proposal_manifest.get("base_version_id")
                if self.proposal_manifest is not None
                else None
            ),
            "proposal_base_version_binding": self._proposal_base_version_binding,
            "input_sha256": input_hash,
            "stable_parent_hash_before": stable_hash_before,
            "stable_parent_hash_after": stable_hash_after,
            "stable_parent_unchanged": isolated,
            "candidate_parent_hash_before": parent_hash_before,
            "candidate_parent_hash_after": parent_hash_after,
            "candidate_output_dir": str(output_dir),
            "candidate_output_limit_mib": self.sandbox.limits.tmpfs_mib,
            "container_isolation": bool(
                result.backend == "kubernetes"
                and result.backend == self.sandbox.backend_name
                and getattr(self.sandbox, "isolation_verified", False)
                and getattr(self.sandbox, "isolation_evidence", {}).get("container_isolation")
            ),
            "isolation_evidence": getattr(self.sandbox, "isolation_evidence", {}),
            "response_error": response_error,
            "reported_cost": cost,
            "mount_policy": {
                "candidate_read_only": True,
                "parent_checkpoint_read_only": True,
                "stable_checkpoint_read_only": True,
                "task_model_read_only": True,
                "candidate_output_only_writable": True,
                "candidate_output_storage": {
                    "type": "bounded_tmpfs",
                    "max_mib": self.sandbox.limits.tmpfs_mib,
                    "persisted_after_exit_by_trusted_host": True,
                },
                "network": "disabled",
            },
        }
        checklist = SafetyChecklist(
            sandboxed=(
                result.backend == "kubernetes"
                and evidence["container_isolation"]
                and result.failure_code != "sandbox_unavailable"
            ),
            executable=result.status == "completed" and result.returncode == 0,
            data_valid=data_valid,
            interfaces_valid=schema_valid and lineage_valid,
            within_budget=budget_ok,
            state_isolated=isolated,
        )

        if self._gate_status is None:
            assert stable_version is not None and candidate_version is not None
            assessment = self.gate.assess(
                stable_version=stable_version,
                candidate=candidate_version,
                checklist=checklist,
                evidence=evidence,
            )
            self._gate_status = assessment.status.value
            if not assessment.eligible:
                self._terminal_failure = ",".join(assessment.failed_checks)
            gate_result = assessment.to_dict()
        else:
            if not checklist.passed:
                self._gate_status = CandidateGateStatus.REJECTED.value
                self._terminal_failure = (
                    response_error
                    or result.failure_code
                    or ",".join(checklist.failed_checks)
                )
            gate_result = {
                "status": self._gate_status,
                "failed_checks": (
                    [] if checklist.passed else list(checklist.failed_checks)
                ),
                "candidate_id": self.candidate_id,
                "candidate_version": candidate_version.version.version_id,
                "parent_version": candidate_version.version.parent_version_id,
                "evidence": evidence,
            }
            if not checklist.passed:
                self.store.append_event(
                    "candidate_shadow_failed",
                    {**gate_result, "terminal_failure": self._terminal_failure},
                    event_id=f"{run_id}:{batch_id}:{self.candidate_id}:failed",
                )

        accepted_predictions = (
            predictions
            if self._gate_status == CandidateGateStatus.ELIGIBLE_FOR_OBSERVATION.value
            and self._terminal_failure is None
            and checklist.passed
            else None
        )
        trial = {
            "candidate_id": self.candidate_id,
            "candidate_version_id": candidate_version.version.version_id,
            "parent_version_id": candidate_version.version.parent_version_id,
            "batch_id": batch_id,
            "batch_index": batch_index,
            "status": (
                "eligible_for_observation"
                if accepted_predictions is not None
                else "rejected"
            ),
            "gate": gate_result,
            "predictions": (
                [
                    {"record_id": record_id, "prediction": label}
                    for record_id, label in sorted(accepted_predictions.items())
                ]
                if accepted_predictions is not None
                else None
            ),
            "checkpoint_version": (
                _checkpoint_version(next_checkpoint) if next_checkpoint is not None else None
            ),
            "candidate_checkpoint_retained": (
                next_checkpoint is not None
                and accepted_predictions is not None
                and output_dir.resolve() in Path(next_checkpoint["adapter_path"]).resolve().parents
            ),
            "cost": cost,
        }
        self._save_metadata(
            checkpoint=(
                next_checkpoint
                if next_checkpoint is not None and accepted_predictions is not None
                else None
            ),
            trial=trial,
        )
        if next_checkpoint is not None and accepted_predictions is not None:
            previous_adapter = str(checkpoint["adapter_path"])
            next_adapter = str(next_checkpoint["adapter_path"])
            if Path(previous_adapter).resolve() != Path(next_adapter).resolve():
                self._remove_attempt_for_adapter(previous_adapter)
            if output_dir.resolve() not in Path(next_adapter).resolve().parents:
                self._remove_attempt_directory(output_dir)
        else:
            self._remove_attempt_directory(output_dir)
        _atomic_json(self.root / "trials" / f"batch-{batch_index:06d}.json", trial)
        self.store.append_event(
            "candidate_shadow_batch_completed",
            {key: value for key, value in trial.items() if key != "predictions"},
            event_id=f"{run_id}:{batch_id}:{self.candidate_id}:shadow",
        )
        return trial

    def run_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "candidate_artifact_sha256": self.candidate_artifact_hash,
            "proposal_manifest_sha256": self.proposal_manifest_sha256,
            "proposal_base_version_id": (
                self.proposal_manifest.get("base_version_id")
                if self.proposal_manifest is not None
                else None
            ),
            "proposal_base_version_binding": self._proposal_base_version_binding,
            "gate_status": self._gate_status,
            "terminal_failure": self._terminal_failure,
            "candidate_version": (
                self._candidate_version.version.to_dict()
                if self._candidate_version is not None
                else None
            ),
        }
