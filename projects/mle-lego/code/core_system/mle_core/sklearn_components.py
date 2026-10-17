from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .interfaces import Evaluator, TaskModelTrainer
from .models import Budget, TaskRecord
from .store import ArtifactStore


def _checkpoint_terms(value: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = re.findall(r"(?u)\b\w+\b", value.lower())
    terms = list(tokens) if ngram_range[0] <= 1 <= ngram_range[1] else []
    if ngram_range[0] <= 2 <= ngram_range[1]:
        terms.extend(f"{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False))
    return terms


def _checkpoint_scores(checkpoint: dict[str, Any], value: str) -> dict[str, float]:
    """Score text in the same TF-IDF space used by the sklearn trainer."""
    vocabulary = checkpoint["vocabulary"]
    idf = checkpoint["idf"]
    ngram_range = tuple(checkpoint["ngram_range"])
    counts = Counter(_checkpoint_terms(value, ngram_range))
    vector = {
        index: count * float(idf[index])
        for term, count in counts.items()
        if (index := vocabulary.get(term)) is not None
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        vector = {index: value / norm for index, value in vector.items()}
    scores: dict[str, float] = {}
    for label, weights in checkpoint["class_weights"].items():
        scores[label] = float(checkpoint["class_intercepts"][label]) + sum(
            value * float(weights[index])
            for index, value in vector.items()
            if index < len(weights)
        )
    return scores


class TfidfLogisticTrainer(TaskModelTrainer):
    """Local real-model baseline with JSON-serializable checkpoints."""

    name = "tfidf_logistic_regression"
    version = "1.1"

    def train(
        self,
        *,
        train_records: list[TaskRecord],
        initial_checkpoint: dict[str, Any] | None,
        config: dict[str, Any],
        seed: int,
        store: ArtifactStore,
    ) -> tuple[dict[str, Any], Budget]:
        labeled = [record for record in train_records if record.target is not None]
        if len(labeled) < 2:
            raise ValueError("TfidfLogisticTrainer needs at least two labeled records")
        texts = [str(record.input) for record in labeled]
        labels = [str(record.target) for record in labeled]
        if len(set(labels)) < 2:
            raise ValueError("TfidfLogisticTrainer needs at least two labels")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=tuple(config.get("ngram_range", (1, 2))),
            min_df=int(config.get("min_df", 1)),
            max_features=config.get("max_features"),
            token_pattern=r"(?u)\b\w+\b",
        )
        features = vectorizer.fit_transform(texts)
        model = LogisticRegression(
            max_iter=int(config.get("max_iter", 200)),
            random_state=seed,
        )
        model.fit(features, labels)
        if len(model.classes_) == 2 and model.coef_.shape[0] == 1:
            class_weights = {
                str(model.classes_[0]): (-model.coef_[0]).tolist(),
                str(model.classes_[1]): model.coef_[0].tolist(),
            }
            class_intercepts = {
                str(model.classes_[0]): float(-model.intercept_[0]),
                str(model.classes_[1]): float(model.intercept_[0]),
            }
        else:
            class_weights = {
                str(label): row.tolist()
                for label, row in zip(model.classes_, model.coef_, strict=False)
            }
            class_intercepts = {
                str(label): float(intercept)
                for label, intercept in zip(
                    model.classes_, model.intercept_, strict=False
                )
            }
        checkpoint = {
            "trainer": self.name,
            "trainer_version": self.version,
            "vocabulary": vectorizer.vocabulary_,
            "idf": vectorizer.idf_.tolist(),
            "ngram_range": list(vectorizer.ngram_range),
            "classes": model.classes_.tolist(),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "class_weights": class_weights,
            "class_intercepts": class_intercepts,
            "num_records": len(labeled),
            "seed": seed,
            "parent_checkpoint": initial_checkpoint,
        }
        return checkpoint, Budget(training_steps=int(config.get("steps", 1)))


class TfidfLogisticEvaluator(Evaluator):
    name = "tfidf_logistic_exact_match"
    version = "1.1"

    def evaluate(
        self,
        *,
        checkpoint: dict[str, Any],
        records: list[TaskRecord],
        config: dict[str, Any],
    ) -> tuple[dict[str, float], Budget]:
        labeled = [record for record in records if record.target is not None]
        if not labeled:
            return {"accuracy": 0.0, "num_records": 0.0}, Budget(
                evaluator_calls=1
            )
        correct = 0
        for record in labeled:
            scores = _checkpoint_scores(checkpoint, str(record.input))
            prediction = max(scores, key=lambda label: (scores[label], label))
            correct += prediction == str(record.target)
        return {
            "accuracy": correct / len(labeled),
            "num_records": float(len(labeled)),
        }, Budget(evaluator_calls=1)
