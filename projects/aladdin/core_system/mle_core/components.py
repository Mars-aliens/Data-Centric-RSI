from __future__ import annotations

from collections import Counter
from typing import Any

from .interfaces import DataOperation, Evaluator, TaskModelTrainer
from .models import Budget, TaskRecord, utc_now
from .store import ArtifactStore


class SelectRecords(DataOperation):
    name = "select_records"
    version = "1.0"

    def run(
        self,
        *,
        arguments: dict[str, Any],
        records: list[TaskRecord],
        store: ArtifactStore,
    ) -> tuple[list[TaskRecord], Budget]:
        record_ids = set(arguments.get("record_ids", []))
        if record_ids:
            selected = [record for record in records if record.record_id in record_ids]
        else:
            selected = records
        return selected, Budget()


class PassThroughTransform(DataOperation):
    name = "pass_through_transform"
    version = "1.0"

    def run(
        self,
        *,
        arguments: dict[str, Any],
        records: list[TaskRecord],
        store: ArtifactStore,
    ) -> tuple[list[TaskRecord], Budget]:
        return records, Budget()


class FilterByMetadata(DataOperation):
    name = "filter_by_metadata"
    version = "1.0"

    def run(
        self,
        *,
        arguments: dict[str, Any],
        records: list[TaskRecord],
        store: ArtifactStore,
    ) -> tuple[list[TaskRecord], Budget]:
        key = arguments["key"]
        expected = arguments["equals"]
        return [
            record for record in records if record.metadata.get(key) == expected
        ], Budget()


class GenerateTemplateRecords(DataOperation):
    name = "generate_template_records"
    version = "1.0"

    def run(
        self,
        *,
        arguments: dict[str, Any],
        records: list[TaskRecord],
        store: ArtifactStore,
    ) -> tuple[list[TaskRecord], Budget]:
        task_id = arguments["task_id"]
        split = arguments.get("split", "development")
        templates = arguments.get("templates", [])
        generated: list[TaskRecord] = list(records) if arguments.get("include_input_records") else []
        for index, item in enumerate(templates):
            if not isinstance(item, dict):
                raise ValueError("Each template must be an object")
            generated.append(
                TaskRecord(
                    record_id=item.get(
                        "record_id", f"generated-{utc_now()}-{index:04d}"
                    ),
                    task_id=task_id,
                    split=split,
                    input=item["input"],
                    target=item.get("target"),
                    source_group_id=item.get("source_group_id", "generated"),
                    metadata={
                        "generated": True,
                        "strategy": arguments.get("strategy", "template"),
                        **item.get("metadata", {}),
                    },
                )
            )
        return generated, Budget(
            llm_calls=int(arguments.get("llm_calls", 0)),
            llm_tokens=int(arguments.get("llm_tokens", 0)),
        )


class ApplyFeedbackLabels(DataOperation):
    name = "apply_feedback_labels"
    version = "1.0"

    def run(
        self,
        *,
        arguments: dict[str, Any],
        records: list[TaskRecord],
        store: ArtifactStore,
    ) -> tuple[list[TaskRecord], Budget]:
        labels = arguments.get("labels", {})
        accepted_ids = set(arguments.get("accepted_record_ids", ()))
        if not isinstance(labels, dict):
            raise ValueError("Feedback labels must be an object")
        output = []
        for record in records:
            if record.record_id not in accepted_ids:
                output.append(record)
                continue
            if record.record_id not in labels:
                raise ValueError(f"No feedback label for accepted record {record.record_id}")
            output.append(
                TaskRecord(
                    record_id=record.record_id,
                    task_id=record.task_id,
                    split=record.split,
                    input=record.input,
                    target=labels[record.record_id],
                    source_group_id=record.source_group_id,
                    metadata={**record.metadata, "feedback_labeled": True},
                    parent_record_ids=record.parent_record_ids,
                )
            )
        return output, Budget()


class MajorityLabelTrainer(TaskModelTrainer):
    """Small deterministic trainer used only to verify the core loop."""

    name = "majority_label_trainer"
    version = "1.0"

    def train(
        self,
        *,
        train_records: list[TaskRecord],
        initial_checkpoint: dict[str, Any] | None,
        config: dict[str, Any],
        seed: int,
        store: ArtifactStore,
    ) -> tuple[dict[str, Any], Budget]:
        labels = [record.target for record in train_records if record.target is not None]
        counts = Counter(labels)
        if not counts:
            raise ValueError("Task Model trainer received no labeled records")
        majority = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]
        checkpoint = {
            "trainer": self.name,
            "trainer_version": self.version,
            "majority_label": majority,
            "label_counts": dict(counts),
            "num_records": len(train_records),
            "seed": seed,
            "parent_checkpoint": initial_checkpoint,
        }
        return checkpoint, Budget(training_steps=1)


class ExactMatchEvaluator(Evaluator):
    name = "exact_match"
    version = "1.0"

    def evaluate(
        self,
        *,
        checkpoint: dict[str, Any],
        records: list[TaskRecord],
        config: dict[str, Any],
    ) -> tuple[dict[str, float], Budget]:
        if not records:
            return {"exact_match": 0.0}, Budget(evaluator_calls=1)
        prediction = checkpoint["majority_label"]
        correct = sum(record.target == prediction for record in records)
        return {
            "exact_match": correct / len(records),
            "num_records": float(len(records)),
        }, Budget(evaluator_calls=1)
