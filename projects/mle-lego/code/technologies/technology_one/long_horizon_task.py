from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


LABELS = ("preferred", "rejected", "tie")
SCORE_FIELDS = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
DATASET_REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
TASK_DEFINITION_VERSION = "mle-lego-long-horizon-task-v1"


class TaskIntegrityError(ValueError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prompt_key(value: Any) -> str:
    prompt = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not prompt:
        raise TaskIntegrityError("prompt must be non-empty")
    return prompt


@dataclass(frozen=True)
class LongHorizonTaskConfig:
    stream_seed: int = 17
    groups_per_batch: int = 50
    feedback_delay_batches: int = 2
    max_batches: int | None = None
    seed_percent: int = 10
    development_percent: int = 60
    held_in_percent: int = 15
    held_out_percent: int = 15
    dataset_revision: str = DATASET_REVISION

    def __post_init__(self) -> None:
        if self.groups_per_batch < 1:
            raise ValueError("groups_per_batch must be positive")
        if self.feedback_delay_batches not in {0, 2, 5}:
            raise ValueError("feedback_delay_batches must be one of 0, 2, 5")
        if self.max_batches is not None and self.max_batches < 1:
            raise ValueError("max_batches must be positive when set")
        percentages = (
            self.seed_percent,
            self.development_percent,
            self.held_in_percent,
            self.held_out_percent,
        )
        if any(value < 0 for value in percentages) or sum(percentages) != 100:
            raise ValueError("train split percentages must be non-negative and sum to 100")
        if self.development_percent == 0 or self.seed_percent == 0:
            raise ValueError("seed and development splits must both be non-empty")
        if not self.dataset_revision:
            raise ValueError("dataset_revision must be non-empty")


@dataclass(frozen=True)
class LabeledExample:
    record_id: str
    source_group_id: str
    input: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "source_group_id": self.source_group_id,
            "input": self.input,
            "target": self.target,
        }


@dataclass(frozen=True)
class ObservedExample:
    record_id: str
    source_group_id: str
    input: str
    source_batch_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_group_id": self.source_group_id,
            "input": self.input,
            "source_batch_index": self.source_batch_index,
        }


@dataclass(frozen=True)
class StreamBatch:
    batch_index: int
    batch_id: str
    source_group_ids: tuple[str, ...]
    records: tuple[ObservedExample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "batch_id": self.batch_id,
            "source_group_ids": list(self.source_group_ids),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ReleasedFeedback:
    feedback_id: str
    record_id: str
    source_group_id: str
    source_batch_index: int
    available_from_batch: int
    input: str
    target: str
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "record_id": self.record_id,
            "source_group_id": self.source_group_id,
            "source_batch_index": self.source_batch_index,
            "available_from_batch": self.available_from_batch,
            "input": self.input,
            "target": self.target,
            "scores": dict(self.scores),
        }


@dataclass(frozen=True)
class PrequentialMetric:
    source_batch_index: int
    available_from_batch: int
    model_version_id: str
    accuracy: float
    macro_f1: float
    num_records: int
    per_class_count: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_batch_index": self.source_batch_index,
            "available_from_batch": self.available_from_batch,
            "model_version_id": self.model_version_id,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "num_records": self.num_records,
            "per_class_count": dict(self.per_class_count),
            "confusion_matrix": {
                target: dict(predictions)
                for target, predictions in self.confusion_matrix.items()
            },
        }


@dataclass(frozen=True)
class AgentBatchInput:
    batch: StreamBatch
    released_feedback: tuple[ReleasedFeedback, ...]
    released_metrics: tuple[PrequentialMetric, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch.to_dict(),
            "released_feedback": [item.to_dict() for item in self.released_feedback],
            "released_metrics": [item.to_dict() for item in self.released_metrics],
        }


@dataclass(frozen=True)
class AgentTaskView:
    task_id: str
    manifest: dict[str, Any]
    seed_examples: tuple[LabeledExample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "manifest": dict(self.manifest),
            "seed_examples": [item.to_dict() for item in self.seed_examples],
        }


@dataclass(frozen=True)
class _GoldRecord:
    record_id: str
    source_group_id: str
    input: str
    target: str
    scores: dict[str, float]
    score_mean: float


@dataclass(frozen=True)
class _Group:
    group_id: str
    source_split: str
    split: str
    prompt: str
    records: tuple[_GoldRecord, ...]


def _score_row(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for field_name in SCORE_FIELDS:
        try:
            value = float(row[field_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskIntegrityError(f"missing or invalid HelpSteer2 score {field_name!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 4.0:
            raise TaskIntegrityError(f"HelpSteer2 score {field_name!r} must be in [0, 4]")
        values[field_name] = value
    return values


def _train_split(group_id: str, config: LongHorizonTaskConfig) -> str:
    bucket = int(_sha256(group_id.encode("utf-8"))[:8], 16) % 100
    thresholds = (
        (config.seed_percent, "seed"),
        (config.seed_percent + config.development_percent, "development"),
        (
            config.seed_percent + config.development_percent + config.held_in_percent,
            "held_in",
        ),
    )
    for threshold, split in thresholds:
        if bucket < threshold:
            return split
    return "held_out"


def _make_groups(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_split: str,
    config: LongHorizonTaskConfig,
) -> tuple[list[_Group], list[dict[str, Any]], int]:
    grouped: dict[str, list[tuple[Mapping[str, Any], dict[str, float]]]] = {}
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        if "prompt" not in row or "response" not in row:
            raise TaskIntegrityError("each HelpSteer2 row must contain prompt and response")
        prompt = _prompt_key(row["prompt"])
        response = str(row["response"]).strip()
        if not response:
            raise TaskIntegrityError("response must be non-empty")
        scores = _score_row(row)
        grouped.setdefault(prompt, []).append((row, scores))
        canonical_rows.append(
            {
                "source_split": source_split,
                "prompt": prompt,
                "response": response,
                "scores": scores,
            }
        )

    result: list[_Group] = []
    singleton_count = 0
    for prompt in sorted(grouped, key=lambda value: _sha256(value.encode("utf-8"))):
        items = grouped[prompt]
        if len(items) < 2:
            singleton_count += 1
            continue
        group_id = f"prompt-{_sha256(prompt.encode('utf-8'))[:16]}"
        scored = [
            (str(row["response"]).strip(), scores, sum(scores.values()) / len(SCORE_FIELDS))
            for row, scores in items
        ]
        scored.sort(key=lambda item: (item[0], tuple(item[1][key] for key in SCORE_FIELDS)))
        means = [item[2] for item in scored]
        highest, lowest = max(means), min(means)
        labels = (
            ["tie"] * len(means)
            if highest == lowest
            else [
                "preferred" if value == highest else "rejected" if value == lowest else "tie"
                for value in means
            ]
        )
        occurrences: dict[bytes, int] = {}
        gold_records: list[_GoldRecord] = []
        for (response, scores, mean), label in zip(scored, labels, strict=True):
            signature = _canonical_json({"response": response, "scores": scores})
            occurrence = occurrences.get(signature, 0)
            occurrences[signature] = occurrence + 1
            record_id = "helpsteer2-" + _sha256(
                _canonical_json(
                    {"group_id": group_id, "signature": signature.decode("utf-8"), "occurrence": occurrence}
                )
            )[:20]
            gold_records.append(
                _GoldRecord(
                    record_id,
                    group_id,
                    f"Prompt: {prompt}\nResponse: {response}",
                    label,
                    dict(scores),
                    mean,
                )
            )
        split = "final_test" if source_split == "validation" else _train_split(group_id, config)
        result.append(
            _Group(group_id, source_split, split, prompt, tuple(gold_records))
        )
    return result, canonical_rows, singleton_count


class HelpSteer2BatchScheduler:
    """Trusted runner-side cursor; only the requested current batch is returned."""

    def __init__(self, batches: tuple[StreamBatch, ...], *, next_batch_index: int = 0):
        self._batches = batches
        self._next_batch_index = next_batch_index
        if not 0 <= next_batch_index <= len(batches):
            raise ValueError("next_batch_index is outside the stream")

    @property
    def batch_count(self) -> int:
        return len(self._batches)

    @property
    def batches(self) -> tuple[StreamBatch, ...]:
        return self._batches

    def next_batch(self) -> StreamBatch:
        if self._next_batch_index >= len(self._batches):
            raise StopIteration("development stream is complete")
        batch = self._batches[self._next_batch_index]
        self._next_batch_index += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"next_batch_index": self._next_batch_index}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        next_index = int(state["next_batch_index"])
        if not 0 <= next_index <= len(self._batches):
            raise ValueError("scheduler state is outside the stream")
        self._next_batch_index = next_index


class FeedbackOracle:
    """Trusted label store with causal release and prequential scoring."""

    def __init__(
        self,
        *,
        batches: tuple[StreamBatch, ...],
        gold_records: Mapping[str, _GoldRecord],
        feedback_delay_batches: int,
        state: Mapping[str, Any] | None = None,
    ):
        self._batches = batches
        self._gold = dict(gold_records)
        self._delay = feedback_delay_batches
        self._schedule: dict[int, list[ReleasedFeedback]] = {}
        self._batch_record_ids: dict[int, tuple[str, ...]] = {}
        for batch in batches:
            self._batch_record_ids[batch.batch_index] = tuple(
                record.record_id for record in batch.records
            )
            for record in batch.records:
                gold = self._gold[record.record_id]
                available = batch.batch_index + feedback_delay_batches + 1
                feedback = ReleasedFeedback(
                    feedback_id=f"feedback-{record.record_id}",
                    record_id=record.record_id,
                    source_group_id=record.source_group_id,
                    source_batch_index=batch.batch_index,
                    available_from_batch=available,
                    input=gold.input,
                    target=gold.target,
                    scores=dict(gold.scores),
                )
                self._schedule.setdefault(available, []).append(feedback)
        self._released_feedback_ids: set[str] = set()
        self._last_started_batch = -1
        self._finalized = False
        self._predictions: dict[tuple[int, str], dict[str, str]] = {}
        self._released_metric_keys: set[tuple[int, str]] = set()
        if state is not None:
            self.restore_state(state)

    @property
    def feedback_delay_batches(self) -> int:
        return self._delay

    def release_for_batch(self, batch_index: int) -> tuple[ReleasedFeedback, ...]:
        if self._finalized:
            raise ValueError("feedback oracle has been finalized")
        if batch_index != self._last_started_batch + 1:
            raise ValueError("feedback batches must be opened sequentially without skips")
        if batch_index >= len(self._batches):
            raise ValueError("cannot release feedback outside the development stream")
        self._last_started_batch = batch_index
        due = self._schedule.get(batch_index, ())
        new_items = tuple(
            item for item in due if item.feedback_id not in self._released_feedback_ids
        )
        self._released_feedback_ids.update(item.feedback_id for item in new_items)
        return new_items

    def submit_predictions(
        self,
        *,
        source_batch_index: int,
        model_version_id: str,
        predictions: Mapping[str, str],
    ) -> None:
        if self._finalized:
            raise ValueError("feedback oracle has been finalized")
        if not 0 <= source_batch_index < len(self._batches):
            raise ValueError("source_batch_index is outside the development stream")
        if source_batch_index != self._last_started_batch:
            raise ValueError("predictions must be submitted while their source batch is active")
        if not model_version_id.strip():
            raise ValueError("model_version_id must be non-empty")
        expected = set(self._batch_record_ids[source_batch_index])
        actual = set(predictions)
        if actual != expected:
            raise ValueError(
                f"predictions must cover exactly the current batch; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        invalid = {value for value in predictions.values() if value not in LABELS}
        if invalid:
            raise ValueError(f"predictions contain unsupported labels: {sorted(invalid)}")
        key = (source_batch_index, model_version_id)
        if key in self._predictions:
            raise ValueError("predictions already submitted for this version and batch")
        self._predictions[key] = dict(predictions)

    def release_prequential_metrics(
        self, batch_index: int
    ) -> tuple[PrequentialMetric, ...]:
        if batch_index != self._last_started_batch:
            raise ValueError("metrics can only be released for the currently open batch")
        released: list[PrequentialMetric] = []
        for (source_batch, version_id), predictions in sorted(self._predictions.items()):
            key = (source_batch, version_id)
            if key in self._released_metric_keys:
                continue
            record_ids = self._batch_record_ids[source_batch]
            feedback_ids = {f"feedback-{record_id}" for record_id in record_ids}
            if not feedback_ids.issubset(self._released_feedback_ids):
                continue
            expected_release = source_batch + self._delay + 1
            if expected_release > batch_index:
                continue
            targets = {record_id: self._gold[record_id].target for record_id in record_ids}
            metrics = _classification_metrics(targets, predictions)
            released.append(
                PrequentialMetric(
                    source_batch_index=source_batch,
                    available_from_batch=expected_release,
                    model_version_id=version_id,
                    **metrics,
                )
            )
            self._released_metric_keys.add(key)
        return tuple(released)

    def finalize_prequential_metrics(self) -> tuple[PrequentialMetric, ...]:
        """Score due feedback for offline reporting; never return it to the agent."""
        if self._last_started_batch != len(self._batches) - 1:
            raise ValueError("the development stream must be complete before finalization")
        terminal_boundary = len(self._batches)
        due_ids = {
            item.feedback_id
            for release_batch, items in self._schedule.items()
            if release_batch <= terminal_boundary
            for item in items
        }
        self._released_feedback_ids.update(due_ids)
        released: list[PrequentialMetric] = []
        for (source_batch, version_id), predictions in sorted(self._predictions.items()):
            key = (source_batch, version_id)
            if key in self._released_metric_keys:
                continue
            expected_ids = {
                f"feedback-{record_id}"
                for record_id in self._batch_record_ids[source_batch]
            }
            if not expected_ids.issubset(due_ids):
                continue
            expected_release = source_batch + self._delay + 1
            if expected_release > terminal_boundary:
                continue
            targets = {
                record_id: self._gold[record_id].target
                for record_id in self._batch_record_ids[source_batch]
            }
            released.append(
                PrequentialMetric(
                    source_batch_index=source_batch,
                    available_from_batch=expected_release,
                    model_version_id=version_id,
                    **_classification_metrics(targets, predictions),
                )
            )
            self._released_metric_keys.add(key)
        self._finalized = True
        return tuple(released)

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_started_batch": self._last_started_batch,
            "finalized": self._finalized,
            "released_feedback_ids": sorted(self._released_feedback_ids),
            "predictions": [
                {
                    "source_batch_index": batch_index,
                    "model_version_id": version_id,
                    "values": predictions,
                }
                for (batch_index, version_id), predictions in sorted(self._predictions.items())
            ],
            "released_metric_keys": [
                {"source_batch_index": batch_index, "model_version_id": version_id}
                for batch_index, version_id in sorted(self._released_metric_keys)
            ],
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        last_started = int(state.get("last_started_batch", -1))
        finalized = bool(state.get("finalized", False))
        if not -1 <= last_started < len(self._batches):
            raise ValueError("oracle state has an invalid current batch")
        if finalized and last_started != len(self._batches) - 1:
            raise ValueError("oracle state is finalized before the stream ended")
        valid_feedback_ids = {
            item.feedback_id for items in self._schedule.values() for item in items
        }
        released_ids = set(state.get("released_feedback_ids", ()))
        if not released_ids.issubset(valid_feedback_ids):
            raise ValueError("oracle state contains unknown feedback IDs")
        release_limit = len(self._batches) if finalized else last_started
        for release_batch, items in self._schedule.items():
            if release_batch > release_limit and any(
                item.feedback_id in released_ids for item in items
            ):
                raise ValueError("oracle state releases feedback before its scheduled batch")

        predictions: dict[tuple[int, str], dict[str, str]] = {}
        for item in state.get("predictions", ()):
            source_batch = int(item["source_batch_index"])
            version_id = str(item["model_version_id"])
            if not version_id:
                raise ValueError("oracle state contains an empty model version")
            values = dict(item["values"])
            if not 0 <= source_batch <= last_started:
                raise ValueError("oracle state contains predictions for a future batch")
            if set(values) != set(self._batch_record_ids[source_batch]):
                raise ValueError("oracle state predictions do not match their source batch")
            if any(value not in LABELS for value in values.values()):
                raise ValueError("oracle state contains an invalid predicted label")
            predictions[(source_batch, version_id)] = values
        metric_keys = {
            (int(item["source_batch_index"]), str(item["model_version_id"]))
            for item in state.get("released_metric_keys", ())
        }
        if not metric_keys.issubset(predictions):
            raise ValueError("oracle state marks unavailable predictions as scored")
        for source_batch, _version_id in metric_keys:
            if source_batch + self._delay + 1 > release_limit:
                raise ValueError("oracle state contains a pre-sequence metric released early")
            expected_ids = {
                f"feedback-{record_id}"
                for record_id in self._batch_record_ids[source_batch]
            }
            if not expected_ids.issubset(released_ids):
                raise ValueError("oracle state scores feedback before all labels were released")
        self._last_started_batch = last_started
        self._finalized = finalized
        self._released_feedback_ids = released_ids
        self._predictions = predictions
        self._released_metric_keys = metric_keys


@dataclass(frozen=True)
class PreparedLongHorizonTask:
    agent_view: AgentTaskView
    scheduler: HelpSteer2BatchScheduler
    feedback_oracle: FeedbackOracle

    def next_agent_input(self) -> AgentBatchInput:
        batch = self.scheduler.next_batch()
        feedback = self.feedback_oracle.release_for_batch(batch.batch_index)
        metrics = self.feedback_oracle.release_prequential_metrics(batch.batch_index)
        return AgentBatchInput(batch, feedback, metrics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "feedback_oracle": self.feedback_oracle.state_dict(),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        scheduler_state = state["scheduler"]
        oracle_state = state["feedback_oracle"]
        expected_next = int(oracle_state.get("last_started_batch", -1)) + 1
        if int(scheduler_state["next_batch_index"]) != expected_next:
            raise ValueError("scheduler and feedback oracle snapshots are inconsistent")
        self.feedback_oracle.restore_state(oracle_state)
        self.scheduler.restore_state(scheduler_state)


def write_agent_preflight_bundle(
    task: PreparedLongHorizonTask,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write only seed labels and unlabeled stream inputs for data preflight."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"preflight output directory is not empty: {root}")
    agent_dir = root / "agent-visible"
    runner_dir = root / "runner-only"
    agent_dir.mkdir()
    runner_dir.mkdir()

    seed_path = agent_dir / "seed.jsonl"
    with seed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in task.agent_view.seed_examples:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    batches_path = runner_dir / "development-batches.jsonl"
    schedule_path = runner_dir / "feedback-schedule.jsonl"
    with batches_path.open("w", encoding="utf-8", newline="\n") as batch_handle, \
            schedule_path.open("w", encoding="utf-8", newline="\n") as schedule_handle:
        for batch in task.scheduler.batches:
            batch_handle.write(json.dumps(batch.to_dict(), ensure_ascii=False) + "\n")
            release_batch = (
                batch.batch_index + task.feedback_oracle.feedback_delay_batches + 1
            )
            for record in batch.records:
                schedule_handle.write(
                    json.dumps(
                        {
                            "feedback_id": f"feedback-{record.record_id}",
                            "record_id": record.record_id,
                            "source_group_id": record.source_group_id,
                            "source_batch_index": batch.batch_index,
                            "available_from_batch": release_batch,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    artifacts = {
        "seed_jsonl": _sha256(seed_path.read_bytes()),
        "development_batches_jsonl": _sha256(batches_path.read_bytes()),
        "feedback_schedule_jsonl": _sha256(schedule_path.read_bytes()),
    }
    manifest = dict(task.agent_view.manifest)
    manifest["agent_bundle"] = {
        "preflight_only": True,
        "visibility_note": "No development targets, scores, held-out labels, or final-test records are written here.",
        "artifact_sha256": artifacts,
    }
    manifest["manifest_sha256"] = _sha256(_canonical_json(manifest))
    manifest_path = root / "task_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (agent_dir / "task_manifest.json").write_text(
        manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return manifest


def _classification_metrics(
    targets: Mapping[str, str], predictions: Mapping[str, str]
) -> dict[str, Any]:
    if set(targets) != set(predictions):
        raise ValueError("targets and predictions must cover the same records")
    correct = sum(targets[key] == predictions[key] for key in targets)
    f1_values: list[float] = []
    counts: dict[str, int] = {}
    confusion = {target: {predicted: 0 for predicted in LABELS} for target in LABELS}
    for label in LABELS:
        tp = sum(targets[key] == label and predictions[key] == label for key in targets)
        fp = sum(targets[key] != label and predictions[key] == label for key in targets)
        fn = sum(targets[key] == label and predictions[key] != label for key in targets)
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
        counts[label] = sum(target == label for target in targets.values())
    for record_id, target in targets.items():
        confusion[target][predictions[record_id]] += 1
    size = len(targets)
    return {
        "accuracy": correct / size if size else 0.0,
        "macro_f1": sum(f1_values) / len(LABELS),
        "num_records": size,
        "per_class_count": counts,
        "confusion_matrix": confusion,
    }


def prepare_helpsteer2_long_horizon_task(
    train_rows: Iterable[Mapping[str, Any]],
    validation_rows: Iterable[Mapping[str, Any]],
    *,
    config: LongHorizonTaskConfig | None = None,
) -> PreparedLongHorizonTask:
    """Prepare a leak-audited controlled stream without changing the legacy adapter."""
    config = config or LongHorizonTaskConfig()
    train_groups, train_canonical, train_singletons = _make_groups(
        train_rows, source_split="train", config=config
    )
    validation_groups, validation_canonical, validation_singletons = _make_groups(
        validation_rows, source_split="validation", config=config
    )
    train_prompts = {row["prompt"] for row in train_canonical}
    validation_prompts = {row["prompt"] for row in validation_canonical}
    overlap = train_prompts & validation_prompts
    if overlap:
        raise TaskIntegrityError(
            f"prompt groups cross official train/validation splits: {len(overlap)}"
        )

    by_split: dict[str, list[_Group]] = {name: [] for name in ("seed", "development", "held_in", "held_out", "final_test")}
    all_groups = train_groups + validation_groups
    seen_record_ids: set[str] = set()
    for group in all_groups:
        by_split[group.split].append(group)
        for record in group.records:
            if record.record_id in seen_record_ids:
                raise TaskIntegrityError(f"duplicate normalized record ID: {record.record_id}")
            seen_record_ids.add(record.record_id)

    dev_groups = sorted(
        by_split["development"],
        key=lambda group: _sha256(f"{config.stream_seed}:{group.group_id}".encode("utf-8")),
    )
    all_batches: list[StreamBatch] = []
    for start in range(0, len(dev_groups), config.groups_per_batch):
        group_slice = dev_groups[start : start + config.groups_per_batch]
        batch_index = len(all_batches)
        observed = tuple(
            ObservedExample(
                record_id=record.record_id,
                source_group_id=group.group_id,
                input=record.input,
                source_batch_index=batch_index,
            )
            for group in group_slice
            for record in group.records
        )
        all_batches.append(
            StreamBatch(
                batch_index=batch_index,
                batch_id=f"helpsteer2-stream-{batch_index:04d}",
                source_group_ids=tuple(group.group_id for group in group_slice),
                records=observed,
            )
        )
    if not all_batches:
        raise TaskIntegrityError("development split produced no stream batches")
    batches = (
        all_batches[: config.max_batches]
        if config.max_batches is not None
        else all_batches
    )

    gold_records = {
        record.record_id: record
        for group in all_groups
        for record in group.records
    }
    seed_examples = tuple(
        LabeledExample(record.record_id, group.group_id, record.input, record.target)
        for group in by_split["seed"]
        for record in group.records
    )
    if not seed_examples:
        raise TaskIntegrityError("seed split produced no labeled records")
    split_group_ids = {
        split: sorted(group.group_id for group in groups)
        for split, groups in by_split.items()
    }
    split_record_counts = {
        split: sum(len(group.records) for group in groups)
        for split, groups in by_split.items()
    }
    stream_schedule = [
        {
            "batch_index": batch.batch_index,
            "source_group_ids": list(batch.source_group_ids),
            "record_ids": [record.record_id for record in batch.records],
        }
        for batch in batches
    ]
    canonical_source = sorted(
        train_canonical + validation_canonical,
        key=lambda row: _sha256(_canonical_json(row)),
    )
    manifest = {
        "task_definition_version": TASK_DEFINITION_VERSION,
        "task_id": "helpsteer2-preference-long-horizon",
        "dataset": "nvidia/HelpSteer2",
        "dataset_revision": config.dataset_revision,
        "label_policy": {
            "score_fields": list(SCORE_FIELDS),
            "aggregation": "arithmetic_mean",
            "highest": "preferred",
            "lowest": "rejected",
            "strict_intermediate": "tie",
            "all_equal": "all_tie",
        },
        "grouping": "normalized exact prompt; all responses stay together",
        "split_percentages": {
            "seed": config.seed_percent,
            "development": config.development_percent,
            "held_in": config.held_in_percent,
            "held_out": config.held_out_percent,
            "final_test": "official validation split",
        },
        "stream_seed": config.stream_seed,
        "groups_per_batch": config.groups_per_batch,
        "max_batches": config.max_batches,
        "feedback_delay_batches": config.feedback_delay_batches,
        "feedback_timing": (
            "feedback from source batch b becomes available at the start of "
            "batch b + delay + 1; delay 0 is released after source batch b"
        ),
        "split_group_counts": {
            split: len(groups) for split, groups in by_split.items()
        },
        "split_record_counts": split_record_counts,
        "development_batch_count": len(batches),
        "available_development_batch_count": len(all_batches),
        "scheduled_development_batch_count": len(batches),
        "stream_prefix_limited": len(batches) < len(all_batches),
        "excluded_singleton_group_counts": {
            "train": train_singletons,
            "validation": validation_singletons,
        },
        "source_data_sha256": _sha256(_canonical_json(canonical_source)),
        "split_group_sha256": {
            split: _sha256(_canonical_json(ids)) for split, ids in split_group_ids.items()
        },
        "stream_schedule_sha256": _sha256(_canonical_json(stream_schedule)),
        "feedback_visibility": "development labels and raw scores are absent until release",
        "evaluation": "prequential predictions are scored only after their labels release",
    }
    agent_view = AgentTaskView(
        task_id=manifest["task_id"],
        manifest=manifest,
        seed_examples=seed_examples,
    )
    batch_tuple = tuple(batches)
    oracle = FeedbackOracle(
        batches=batch_tuple,
        gold_records=gold_records,
        feedback_delay_batches=config.feedback_delay_batches,
    )
    return PreparedLongHorizonTask(
        agent_view=agent_view,
        scheduler=HelpSteer2BatchScheduler(batch_tuple),
        feedback_oracle=oracle,
    )
