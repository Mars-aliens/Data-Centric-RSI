from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .interfaces import Evaluator, TaskModelTrainer
from .models import Budget, TaskRecord
from .store import ArtifactStore


def tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


class TokenPrototypeTrainer(TaskModelTrainer):
    """Dependency-free text classifier for local pipeline experiments."""

    name = "token_prototype_trainer"
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
        label_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        label_counts: Counter[str] = Counter()
        for record in train_records:
            if record.target is None:
                continue
            label = str(record.target)
            label_counts[label] += 1
            label_token_counts[label].update(tokenize(record.input))
        if not label_counts:
            raise ValueError("TokenPrototypeTrainer received no labeled records")
        checkpoint = {
            "trainer": self.name,
            "trainer_version": self.version,
            "label_counts": dict(label_counts),
            "label_token_counts": {
                label: dict(counts) for label, counts in label_token_counts.items()
            },
            "seed": seed,
            "num_records": len(train_records),
            "parent_checkpoint": initial_checkpoint,
        }
        return checkpoint, Budget(training_steps=1)

    @staticmethod
    def predict(checkpoint: dict[str, Any], value: Any) -> str:
        tokens = Counter(tokenize(value))
        labels = checkpoint["label_counts"]
        best_label = None
        best_score = float("-inf")
        total = max(1, sum(labels.values()))
        for label, count in labels.items():
            token_counts = checkpoint["label_token_counts"].get(label, {})
            score = 0.0
            for token, amount in tokens.items():
                score += amount * (1.0 + token_counts.get(token, 0))
            score += count / total
            if score > best_score or (
                score == best_score and str(label) < str(best_label)
            ):
                best_label = label
                best_score = score
        return str(best_label)


class TextExactMatchEvaluator(Evaluator):
    name = "text_exact_match"
    version = "1.0"

    def evaluate(
        self,
        *,
        checkpoint: dict[str, Any],
        records: list[TaskRecord],
        config: dict[str, Any],
    ) -> tuple[dict[str, float], Budget]:
        if not records:
            return {"accuracy": 0.0, "num_records": 0.0}, Budget(
                evaluator_calls=1
            )
        correct = sum(
            TokenPrototypeTrainer.predict(checkpoint, record.input)
            == str(record.target)
            for record in records
            if record.target is not None
        )
        labeled = sum(record.target is not None for record in records)
        return {
            "accuracy": correct / max(1, labeled),
            "num_records": float(labeled),
        }, Budget(evaluator_calls=1)
