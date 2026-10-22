from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from core_system.mle_core.interfaces import Evaluator, TaskModelTrainer
from core_system.mle_core.models import Budget, TaskRecord, to_jsonable
from core_system.mle_core.store import ArtifactStore

from .harness_evolution import (
    HarnessEditProposal,
    HarnessProposalError,
    parse_edit_proposal,
    prompt_digest,
)


class ServerCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class JsonCommand:
    argv: tuple[str, ...]
    timeout_seconds: float = 3600.0
    cwd: str | None = None

    @classmethod
    def from_value(cls, value: Sequence[str] | str, *, timeout_seconds: float = 3600.0, cwd: str | None = None) -> "JsonCommand":
        if isinstance(value, str):
            raise ValueError("Commands must be argv arrays, not shell strings")
        argv = tuple(str(item) for item in value)
        if not argv or any(not item for item in argv):
            raise ValueError("Command argv must contain a non-empty executable")
        return cls(argv=argv, timeout_seconds=timeout_seconds, cwd=cwd)


def run_json_command(command: JsonCommand, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command.argv),
            input=json.dumps(payload, ensure_ascii=True),
            stdout=subprocess.PIPE,
            # Child progress is emitted on stderr. Inheriting it keeps the
            # live progress visible and still preserves the JSON stdout pipe.
            stderr=None,
            check=False,
            cwd=command.cwd,
            env=os.environ.copy(),
            text=True,
            timeout=command.timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServerCommandError(f"Server command failed to start or timed out: {command.argv}") from exc
    if completed.returncode != 0:
        raise ServerCommandError(
            f"Server command exited {completed.returncode}: {command.argv}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ServerCommandError(
            f"Server command must emit one JSON object on stdout: {command.argv}; stdout={completed.stdout[-4000:]}"
        ) from exc
    if not isinstance(value, dict):
        raise ServerCommandError(f"Server command output must be a JSON object: {command.argv}")
    return value


def budget_from_value(value: dict[str, Any] | None) -> Budget:
    value = value or {}
    fields = {
        "llm_calls",
        "llm_tokens",
        "training_steps",
        "training_seconds",
        "evaluator_calls",
        "feedback_requests",
        "wall_clock_seconds",
    }
    # Trainers may return record counts as diagnostic metadata alongside the
    # formal Budget fields. Keep those diagnostics out of accounting while
    # continuing to reject genuinely unknown cost keys.
    diagnostic_fields = {"training_records_seen"}
    unknown = set(value) - fields - diagnostic_fields
    if unknown:
        raise ValueError(f"Unknown budget fields: {sorted(unknown)}")
    return Budget(**{field: value.get(field, 0) for field in fields})


def _decision_model_payload(
    model_id: str,
    model_revision: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = {
        "provider",
        "api_url",
        "api_url_env",
        "api_key_env",
        "temperature",
        "top_p",
        "max_output_tokens",
        "json_mode",
    }
    value = {"id": model_id, "revision": model_revision}
    if settings:
        value.update({key: settings[key] for key in allowed if key in settings})
    return value


def _record_payload(records: list[TaskRecord]) -> list[dict[str, Any]]:
    return [to_jsonable(record) for record in records]


class CommandHarnessEditor:
    """Ask the decision model for one constrained, versioned Harness patch."""

    def __init__(
        self,
        command: JsonCommand,
        *,
        model_id: str,
        model_revision: str,
        model_settings: dict[str, Any] | None = None,
    ):
        self.command = command
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_settings = dict(model_settings or {})

    def propose(
        self,
        *,
        current: Any,
        observation: dict[str, Any],
        prior_feedback: dict[str, Any],
    ) -> HarnessEditProposal:
        from .harness_evolution import build_harness_edit_prompt

        system_prompt, user_prompt = build_harness_edit_prompt(
            current=current,
            observation=observation,
            prior_feedback=prior_feedback,
        )
        response = run_json_command(
            self.command,
            {
                "protocol": "lego-harness-edit-v1",
                "model": _decision_model_payload(
                    self.model_id, self.model_revision, self.model_settings
                ),
                "prompt": {"system": system_prompt, "user": user_prompt},
            },
        )
        raw = response.get("proposal")
        if not isinstance(raw, dict):
            raise ServerCommandError("Harness edit response must contain a proposal object")
        cost = budget_from_value(response.get("cost"))
        if cost.llm_calls < 1:
            raise ServerCommandError("Harness edit response must report at least one LLM call")
        try:
            parsed = parse_edit_proposal(raw, current=current)
        except HarnessProposalError as exc:
            raise ServerCommandError(f"Invalid Harness edit proposal: {exc}") from exc
        return HarnessEditProposal(
            base_version=parsed.base_version,
            edits=parsed.edits,
            rationale=parsed.rationale,
            cost=cost,
            model_id=self.model_id,
            model_revision=self.model_revision,
            system_prompt_hash=prompt_digest(system_prompt),
            user_prompt_hash=prompt_digest(user_prompt),
        )


class CommandTaskModelTrainer(TaskModelTrainer):
    def __init__(
        self,
        command: JsonCommand,
        *,
        name: str = "server_task_model",
        version: str = "1",
        initial_checkpoint_path: str | None = None,
        default_config: dict[str, Any] | None = None,
    ):
        self.command = command
        self.name = name
        self.version = version
        self.initial_checkpoint_path = initial_checkpoint_path
        self.default_config = dict(default_config or {})

    def train(
        self,
        *,
        train_records: list[TaskRecord],
        initial_checkpoint: dict[str, Any] | None,
        config: dict[str, Any],
        seed: int,
        store: ArtifactStore,
    ) -> tuple[dict[str, Any], Budget]:
        response = run_json_command(
            self.command,
            {
                "protocol": "lego-trainer-v1",
                "records": _record_payload(train_records),
                "initial_checkpoint": initial_checkpoint,
                "initial_checkpoint_path": self.initial_checkpoint_path,
                "config": {**self.default_config, **config},
                "seed": seed,
                "artifact_store_root": str(store.root),
            },
        )
        checkpoint = response.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ServerCommandError("Trainer response must contain a checkpoint JSON object")
        return checkpoint, budget_from_value(response.get("cost"))


class CommandEvaluator(Evaluator):
    def __init__(self, command: JsonCommand, *, name: str = "server_evaluator", version: str = "1"):
        self.command = command
        self.name = name
        self.version = version

    def evaluate(
        self,
        *,
        checkpoint: dict[str, Any],
        records: list[TaskRecord],
        config: dict[str, Any],
    ) -> tuple[dict[str, float], Budget]:
        response = run_json_command(
            self.command,
            {
                "protocol": "lego-evaluator-v1",
                "checkpoint": checkpoint,
                "records": _record_payload(records),
                "config": config,
            },
        )
        metrics = response.get("metrics")
        if not isinstance(metrics, dict) or any(not isinstance(key, str) for key in metrics):
            raise ServerCommandError("Evaluator response must contain a metrics object")
        return {str(key): float(value) for key, value in metrics.items()}, budget_from_value(response.get("cost"))
