from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .models import (
    ActionProposal,
    ArtifactRef,
    Budget,
    FeedbackRecord,
    Observation,
    TaskRecord,
)
from .store import ArtifactStore


class PolicyAdapter(ABC):
    """Upper-layer decision boundary for Aladdin or LEGO."""

    @abstractmethod
    def propose(self, observation: Observation) -> ActionProposal:
        raise NotImplementedError


class DataOperation(ABC):
    name: str
    version: str

    @abstractmethod
    def run(
        self,
        *,
        arguments: dict[str, Any],
        records: list[TaskRecord],
        store: ArtifactStore,
    ) -> tuple[list[TaskRecord], Budget]:
        raise NotImplementedError


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, DataOperation] = {}

    def register(self, operation: DataOperation) -> None:
        key = f"{operation.name}@{operation.version}"
        if key in self._operations:
            raise ValueError(f"Operation already registered: {key}")
        self._operations[key] = operation

    def get(self, name: str, version: str) -> DataOperation:
        try:
            return self._operations[f"{name}@{version}"]
        except KeyError as exc:
            raise KeyError(f"Unknown data operation: {name}@{version}") from exc


class TaskModelTrainer(ABC):
    name: str
    version: str

    @abstractmethod
    def train(
        self,
        *,
        train_records: list[TaskRecord],
        initial_checkpoint: dict[str, Any] | None,
        config: dict[str, Any],
        seed: int,
        store: ArtifactStore,
    ) -> tuple[dict[str, Any], Budget]:
        raise NotImplementedError


class Evaluator(ABC):
    name: str
    version: str

    @abstractmethod
    def evaluate(
        self,
        *,
        checkpoint: dict[str, Any],
        records: list[TaskRecord],
        config: dict[str, Any],
    ) -> tuple[dict[str, float], Budget]:
        raise NotImplementedError


class FeedbackProvider(ABC):
    @abstractmethod
    def request(self, *, target_artifact_id: str, target_version: str) -> FeedbackRecord:
        raise NotImplementedError


@dataclass
class RegisteredComponents:
    operations: OperationRegistry
    trainer: TaskModelTrainer
    evaluator: Evaluator
