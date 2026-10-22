from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .long_horizon_task import LABELS, PreparedLongHorizonTask
from .server_adapters import JsonCommand, run_json_command


_STATE_VERSION = "long-horizon-qwen-run-v1"
_TRAIN_COST_FIELDS = {"training_steps", "training_records_seen", "training_seconds"}
_PREDICT_COST_FIELDS = {"prediction_records", "prediction_seconds"}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_version(checkpoint: Mapping[str, Any]) -> str:
    version = checkpoint.get("task_model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Qwen checkpoint must include a non-empty task_model_version")
    return version


def _validated_cost(
    raw: object, allowed: set[str], *, required: set[str]
) -> dict[str, int | float]:
    if not isinstance(raw, dict):
        raise ValueError("command cost must be an object")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"command returned unknown cost fields: {sorted(unknown)}")
    missing = required - set(raw)
    if missing:
        raise ValueError(f"command omitted required cost fields: {sorted(missing)}")
    result: dict[str, int | float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"command cost {key!r} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"command cost {key!r} must be finite and non-negative")
        if key in {"training_steps", "training_records_seen", "prediction_records"}:
            if not float(value).is_integer():
                raise ValueError(f"command cost {key!r} must be an integer")
            value = int(value)
        result[key] = value
    return result


def _aggregate_confusion(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    matrix = {target: {predicted: 0 for predicted in LABELS} for target in LABELS}
    for metric in metrics:
        raw = metric.get("confusion_matrix")
        if not isinstance(raw, dict):
            raise ValueError("prequential metric is missing its confusion matrix")
        for target in LABELS:
            for predicted in LABELS:
                value = raw.get(target, {}).get(predicted)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("prequential confusion matrix is invalid")
                matrix[target][predicted] += value
    count = sum(sum(row.values()) for row in matrix.values())
    if not count:
        return None
    f1_values = []
    for label in LABELS:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[target][label] for target in LABELS if target != label)
        false_negative = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "accuracy": sum(matrix[label][label] for label in LABELS) / count,
        "macro_f1": sum(f1_values) / len(LABELS),
        "num_records": count,
        "confusion_matrix": matrix,
    }


class LongHorizonQwenRunner:
    """Trusted fixed-flow runner: predict first, then train only on released labels."""

    def __init__(
        self,
        task: PreparedLongHorizonTask,
        *,
        command: JsonCommand,
        config: dict[str, Any],
        output_dir: str | Path,
        run_id: str,
        seed: int = 17,
        prediction_batch_size: int = 8,
        command_runner: Callable[[JsonCommand, dict[str, Any]], dict[str, Any]] = run_json_command,
        on_batch: Callable[[dict[str, Any]], None] | None = None,
        candidate_shadow: Any | None = None,
    ):
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.task = task
        self.command = command
        self.config = copy.deepcopy(config)
        self.root = Path(output_dir)
        self.run_id = run_id
        self.seed = int(seed)
        self.prediction_batch_size = prediction_batch_size
        self.command_runner = command_runner
        self.on_batch = on_batch
        self.candidate_shadow = candidate_shadow
        self.state_path = self.root / "run_state.json"
        self.attempt_log_path = self.root / "command_attempts.jsonl"
        self._manifest_hash = _canonical_hash(task.agent_view.manifest)
        self._command_spec = {
            "argv": list(command.argv),
            "cwd": command.cwd,
            "timeout_seconds": command.timeout_seconds,
        }
        config_identity = {
            "config": self.config,
            "prediction_batch_size": self.prediction_batch_size,
            "command": self._command_spec,
        }
        if self.candidate_shadow is not None:
            config_identity["candidate_shadow"] = self.candidate_shadow.run_signature()
        self._config_hash = _canonical_hash(config_identity)
        self.state = self._load_or_initialize()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema": _STATE_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "task_manifest_sha256": self._manifest_hash,
            "config_sha256": self._config_hash,
            "model_config": copy.deepcopy(self.config),
            "command": self._command_spec,
            "status": "running",
            "completed_batches": 0,
            "total_batches": self.task.scheduler.batch_count,
            "task_state": self.task.state_dict(),
            "initial_checkpoint": None,
            "active_checkpoint": None,
            "released_feedback": [],
            "batch_history": [],
            "candidate_history": [],
            "total_cost": {
                "training_steps": 0,
                "training_records_seen": 0,
                "training_seconds": 0.0,
                "prediction_records": 0,
                "prediction_seconds": 0.0,
            },
            "terminal_metrics": [],
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.state_path.is_file():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"run state is unreadable: {self.state_path}") from exc
            expected = {
                "schema": _STATE_VERSION,
                "run_id": self.run_id,
                "seed": self.seed,
                "task_manifest_sha256": self._manifest_hash,
                "config_sha256": self._config_hash,
                "total_batches": self.task.scheduler.batch_count,
            }
            mismatches = [key for key, value in expected.items() if state.get(key) != value]
            if mismatches:
                raise ValueError(f"run state does not match this invocation: {mismatches}")
            state.setdefault("candidate_history", [])
            scheduler_index = int(state["task_state"]["scheduler"]["next_batch_index"])
            if (
                state.get("completed_batches") != scheduler_index
                or len(state.get("batch_history", ())) != scheduler_index
            ):
                raise ValueError("run state batch history and scheduler cursor are inconsistent")
            if scheduler_index and state.get("active_checkpoint") is None:
                raise ValueError("run state has completed batches without an active checkpoint")
            self.task.restore_state(state["task_state"])
            return state

        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                f"output directory contains artifacts but no run state: {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        state = self._initial_state()
        self._save_state(state)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _attempt_root(self, operation: str) -> Path:
        path = self.root / "checkpoint-attempts" / operation / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _append_attempt(self, value: dict[str, Any]) -> None:
        with self.attempt_log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        attempt_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        self._append_attempt(
            {
                "attempt_id": attempt_id,
                "operation": request["operation"],
                "status": "started",
                "started_at": started_at,
                "request_sha256": _canonical_hash(request),
            }
        )
        try:
            response = self.command_runner(self.command, request)
        except Exception as exc:
            self._append_attempt(
                {
                    "attempt_id": attempt_id,
                    "operation": request["operation"],
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )
            raise
        self._append_attempt(
            {
                "attempt_id": attempt_id,
                "operation": request["operation"],
                "status": "completed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "response_sha256": _canonical_hash(response),
                "cost": response.get("cost", {}),
            }
        )
        return response

    def _train(
        self,
        records: list[dict[str, Any]],
        *,
        parent: dict[str, Any] | None,
        operation: str,
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, int | float]]:
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "train",
            "records": records,
            "config": self.config,
            "seed": seed,
            "initial_checkpoint": parent,
            "artifact_store_root": str(self._attempt_root(operation)),
        }
        response = self._invoke(request)
        checkpoint = response.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("Qwen train response must contain a checkpoint object")
        _checkpoint_version(checkpoint)
        expected_parent = None
        if parent is not None:
            expected_parent = {
                "task_model_version": _checkpoint_version(parent),
                "adapter_hash": parent.get("adapter_hash"),
            }
        if checkpoint.get("parent_checkpoint") != expected_parent:
            raise ValueError("Qwen checkpoint parent provenance does not match the requested parent")
        json.dumps(checkpoint, ensure_ascii=False)
        return checkpoint, _validated_cost(
            response.get("cost", {}),
            _TRAIN_COST_FIELDS,
            required={"training_steps", "training_records_seen", "training_seconds"},
        )

    def _predict(
        self, checkpoint: dict[str, Any], records: list[dict[str, Any]]
    ) -> tuple[dict[str, str], dict[str, int | float]]:
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "predict",
            "checkpoint": checkpoint,
            "records": [
                {"record_id": item["record_id"], "input": item["input"]}
                for item in records
            ],
            "batch_size": self.prediction_batch_size,
        }
        response = self._invoke(request)
        raw_predictions = response.get("predictions")
        if not isinstance(raw_predictions, list):
            raise ValueError("Qwen predict response must contain a predictions list")
        by_id: dict[str, str] = {}
        for item in raw_predictions:
            if not isinstance(item, dict):
                raise ValueError("each prediction result must be an object")
            record_id, prediction = item.get("record_id"), item.get("prediction")
            if not isinstance(record_id, str) or record_id in by_id:
                raise ValueError("prediction results contain an invalid or duplicate record_id")
            if prediction not in LABELS:
                raise ValueError(f"prediction result has unsupported label: {prediction!r}")
            by_id[record_id] = prediction
        expected = {item["record_id"] for item in records}
        if set(by_id) != expected:
            raise ValueError(
                "Qwen predictions must cover exactly the current batch: "
                f"missing={sorted(expected - set(by_id))}, extra={sorted(set(by_id) - expected)}"
            )
        return by_id, _validated_cost(
            response.get("cost", {}),
            _PREDICT_COST_FIELDS,
            required={"prediction_records", "prediction_seconds"},
        )

    @staticmethod
    def _add_cost(
        total: dict[str, int | float], cost: Mapping[str, int | float]
    ) -> None:
        for key, value in cost.items():
            total[key] = total.get(key, 0) + value

    def _write_summary(self, status: str) -> dict[str, Any]:
        released_metrics: dict[tuple[int, str], dict[str, Any]] = {}
        for batch in self.state["batch_history"]:
            for metric in batch["released_metrics"]:
                key = (metric["source_batch_index"], metric["model_version_id"])
                released_metrics[key] = metric
        for metric in self.state["terminal_metrics"]:
            key = (metric["source_batch_index"], metric["model_version_id"])
            released_metrics[key] = metric
        ordered_metrics = [released_metrics[key] for key in sorted(released_metrics)]
        stable_versions = {
            str(batch["model_version_id"])
            for batch in self.state["batch_history"]
            if batch.get("model_version_id")
        }
        candidate_versions = {
            str(item["candidate_version_id"])
            for item in self.state.get("candidate_history", [])
            if item.get("candidate_version_id")
        }
        stable_metrics = [
            metric for metric in ordered_metrics
            if metric["model_version_id"] in stable_versions
        ]
        candidate_metrics = [
            metric for metric in ordered_metrics
            if metric["model_version_id"] in candidate_versions
        ]
        candidate_metrics_by_version = {
            version_id: _aggregate_confusion(
                [metric for metric in candidate_metrics if metric["model_version_id"] == version_id]
            )
            for version_id in sorted(candidate_versions)
        }
        attempt_audit = self._attempt_audit()
        summary = {
            "status": status,
            "run_id": self.run_id,
            "seed": self.seed,
            "completed_batches": self.state["completed_batches"],
            "target_batches": self.state["total_batches"],
            "batch_history": self.state["batch_history"],
            "terminal_metrics": self.state["terminal_metrics"],
            "prequential_metrics": _aggregate_confusion(stable_metrics),
            "prequential_scored_batches": len(stable_metrics),
            "candidate_prequential_metrics": candidate_metrics_by_version,
            "candidate_history": self.state.get("candidate_history", []),
            "candidate_shadow": (
                self.candidate_shadow.run_metadata()
                if self.candidate_shadow is not None
                else None
            ),
            "task_manifest_sha256": self._manifest_hash,
            "config_sha256": self._config_hash,
            "model_config": self.config,
            "command": self._command_spec,
            "active_model_version": (
                _checkpoint_version(self.state["active_checkpoint"])
                if self.state["active_checkpoint"] is not None
                else None
            ),
            "released_feedback_records": len(self.state["released_feedback"]),
            "pending_feedback_records": self._pending_feedback_count(),
            "cost": self.state["total_cost"],
            "command_attempt_audit": attempt_audit,
            "held_out_used": False,
            "final_test_used": False,
        }
        temporary = self.root / f".summary.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.root / "summary.json")
        return summary

    def _attempt_audit(self) -> dict[str, Any]:
        attempts: dict[str, list[dict[str, Any]]] = {}
        if self.attempt_log_path.is_file():
            for line_number, line in enumerate(
                self.attempt_log_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"command attempt log is corrupt at line {line_number}"
                    ) from exc
                attempts.setdefault(str(item["attempt_id"]), []).append(item)

        counts = {"started": 0, "completed": 0, "failed": 0, "interrupted": 0}
        costs: dict[str, dict[str, int | float]] = {"train": {}, "predict": {}}
        invalid_cost_count = 0
        for entries in attempts.values():
            terminal = [item for item in entries if item.get("status") in {"completed", "failed"}]
            counts["started"] += 1
            if not terminal:
                counts["interrupted"] += 1
                continue
            result = terminal[-1]
            counts[result["status"]] += 1
            if result["status"] != "completed":
                continue
            operation = result.get("operation")
            if operation not in {"train", "predict"} or not isinstance(result.get("cost"), dict):
                invalid_cost_count += 1
                continue
            required_costs = (
                _TRAIN_COST_FIELDS
                if operation == "train"
                else _PREDICT_COST_FIELDS
            )
            if not required_costs.issubset(result["cost"]):
                invalid_cost_count += 1
            for key, value in result["cost"].items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    invalid_cost_count += 1
                    continue
                costs[operation][key] = costs[operation].get(key, 0) + value
        return {
            "counts": counts,
            "reported_cost_by_operation": costs,
            "invalid_or_missing_cost_response_count": invalid_cost_count,
            "failed_attempts_without_reported_cost": counts["failed"],
        }

    def _pending_feedback_count(self) -> int:
        oracle_state = self.state["task_state"]["feedback_oracle"]
        released = set(oracle_state.get("released_feedback_ids", ()))
        total = sum(
            len(batch.records) for batch in self.task.scheduler.batches
        )
        return max(0, total - len(released))

    def _write_predictions(
        self, batch_index: int, records: list[dict[str, Any]], predictions: Mapping[str, str]
    ) -> tuple[str, str]:
        directory = self.root / "predictions"
        directory.mkdir(parents=True, exist_ok=True)
        relative_path = Path("predictions") / f"batch-{batch_index:06d}.jsonl"
        final_path = self.root / relative_path
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{final_path.name}.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            {
                                "record_id": record["record_id"],
                                "prediction": predictions[record["record_id"]],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, final_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
        return str(relative_path), digest

    def _initialize_model(self) -> None:
        seed_records = [item.to_dict() for item in self.task.agent_view.seed_examples]
        checkpoint, cost = self._train(
            seed_records,
            parent=None,
            operation="initial-seed",
            seed=self.seed,
        )
        state = copy.deepcopy(self.state)
        state["initial_checkpoint"] = checkpoint
        state["active_checkpoint"] = checkpoint
        self._add_cost(state["total_cost"], cost)
        self.state = state
        self._save_state(state)

    def run(self) -> dict[str, Any]:
        if self.state["status"] == "complete":
            return self._write_summary("complete")
        if self.state["active_checkpoint"] is None:
            self._initialize_model()

        while True:
            prior_task_state = copy.deepcopy(self.state["task_state"])
            try:
                current = self.task.next_agent_input()
            except StopIteration:
                break

            try:
                state = copy.deepcopy(self.state)
                released = [item.to_dict() for item in current.released_feedback]
                known_feedback = {
                    item["feedback_id"] for item in state["released_feedback"]
                }
                duplicate_feedback = known_feedback.intersection(
                    item["feedback_id"] for item in released
                )
                if duplicate_feedback:
                    raise ValueError(
                        f"feedback was released more than once: {sorted(duplicate_feedback)}"
                    )
                state["released_feedback"].extend(released)

                prediction_checkpoint = state["active_checkpoint"]
                prediction_version_id = _checkpoint_version(prediction_checkpoint)
                predictions, predict_cost = self._predict(
                    prediction_checkpoint,
                    [record.to_dict() for record in current.batch.records],
                )
                self.task.feedback_oracle.submit_predictions(
                    source_batch_index=current.batch.batch_index,
                    model_version_id=prediction_version_id,
                    predictions=predictions,
                )

                candidate_trial: dict[str, Any] | None = None
                if self.candidate_shadow is not None:
                    try:
                        candidate_trial = self.candidate_shadow.run_batch(
                            run_id=self.run_id,
                            batch=current.batch.to_dict(),
                            released_feedback=state["released_feedback"],
                            seed_examples=[
                                item.to_dict() for item in self.task.agent_view.seed_examples
                            ],
                            stable_checkpoint=prediction_checkpoint,
                            task_config=self.config,
                            task_manifest_hash=self._manifest_hash,
                        )
                        candidate_predictions = candidate_trial.get("predictions")
                        candidate_version_id = candidate_trial.get("candidate_version_id")
                        if (
                            candidate_trial.get("status") == "eligible_for_observation"
                            and isinstance(candidate_version_id, str)
                            and isinstance(candidate_predictions, list)
                        ):
                            candidate_by_id = {
                                item["record_id"]: item["prediction"]
                                for item in candidate_predictions
                            }
                            self.task.feedback_oracle.submit_predictions(
                                source_batch_index=current.batch.batch_index,
                                model_version_id=candidate_version_id,
                                predictions=candidate_by_id,
                            )
                    except Exception as exc:
                        candidate_trial = {
                            "candidate_id": getattr(
                                self.candidate_shadow, "candidate_id", "unknown"
                            ),
                            "status": "execution_error",
                            "failure_code": type(exc).__name__,
                            "error": str(exc)[:1000],
                            "predictions": None,
                        }

                active_checkpoint = prediction_checkpoint
                train_cost: dict[str, int | float] = {}
                if released:
                    training_records = [
                        item.to_dict() for item in self.task.agent_view.seed_examples
                    ] + [
                        {
                            "record_id": item["record_id"],
                            "input": item["input"],
                            "target": item["target"],
                        }
                        for item in state["released_feedback"]
                    ]
                    active_checkpoint, train_cost = self._train(
                        training_records,
                        parent=prediction_checkpoint,
                        operation=f"batch-{current.batch.batch_index:06d}",
                        seed=self.seed + current.batch.batch_index + 1,
                    )

                batch_result = {
                    "batch_index": current.batch.batch_index,
                    "batch_id": current.batch.batch_id,
                    "input_record_count": len(current.batch.records),
                    "released_feedback_ids": [item["feedback_id"] for item in released],
                    "released_metrics": [item.to_dict() for item in current.released_metrics],
                    "trained_this_batch": bool(released),
                    "model_version_id": prediction_version_id,
                    "active_model_version_after_batch": _checkpoint_version(active_checkpoint),
                    "cost": {"training": train_cost, "prediction": predict_cost},
                }
                if candidate_trial is not None:
                    batch_result["candidate_trial"] = {
                        key: value
                        for key, value in candidate_trial.items()
                        if key != "predictions"
                    }
                prediction_records = [record.to_dict() for record in current.batch.records]
                prediction_path, prediction_hash = self._write_predictions(
                    current.batch.batch_index, prediction_records, predictions
                )
                batch_result["prediction_artifact"] = prediction_path
                batch_result["prediction_sha256"] = prediction_hash
            except BaseException:
                self.task.restore_state(prior_task_state)
                raise

            state["active_checkpoint"] = active_checkpoint
            state["batch_history"].append(batch_result)
            if candidate_trial is not None:
                state.setdefault("candidate_history", []).append(
                    {
                        key: value
                        for key, value in candidate_trial.items()
                        if key != "predictions"
                    }
                )
            state["completed_batches"] += 1
            state["task_state"] = self.task.state_dict()
            self._add_cost(state["total_cost"], train_cost)
            self._add_cost(state["total_cost"], predict_cost)
            self.state = state
            self._save_state(state)
            self._write_summary("running")
            if self.on_batch is not None:
                self.on_batch(batch_result)

        terminal_metrics = self.task.feedback_oracle.finalize_prequential_metrics()
        state = copy.deepcopy(self.state)
        state["terminal_metrics"] = [item.to_dict() for item in terminal_metrics]
        state["task_state"] = self.task.state_dict()
        state["status"] = "complete"
        self.state = state
        self._save_state(state)
        return self._write_summary("complete")
