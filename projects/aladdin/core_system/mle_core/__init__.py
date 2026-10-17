"""Shared Action System substrate for the MLE-RSI idea prototypes."""

from .models import (
    ActionProposal,
    ArtifactRef,
    Budget,
    FeedbackRecord,
    RunConfig,
    RunStatus,
    TaskRecord,
    TaskSpec,
)
from .runtime import CoreRun, RunOrchestrator
from .local_data import build_local_text_registry
from .text_components import TextExactMatchEvaluator, TokenPrototypeTrainer
from .feedback import FeedbackCatalog, InMemoryFeedbackProvider


def __getattr__(name: str):
    # Keep lightweight model imports usable without the optional sklearn stack.
    if name in {"TfidfLogisticEvaluator", "TfidfLogisticTrainer"}:
        from .sklearn_components import TfidfLogisticEvaluator, TfidfLogisticTrainer

        return {
            "TfidfLogisticEvaluator": TfidfLogisticEvaluator,
            "TfidfLogisticTrainer": TfidfLogisticTrainer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ActionProposal",
    "ArtifactRef",
    "Budget",
    "CoreRun",
    "FeedbackRecord",
    "RunConfig",
    "RunOrchestrator",
    "RunStatus",
    "TaskRecord",
    "TaskSpec",
    "TextExactMatchEvaluator",
    "TokenPrototypeTrainer",
    "build_local_text_registry",
    "FeedbackCatalog",
    "InMemoryFeedbackProvider",
    "TfidfLogisticEvaluator",
    "TfidfLogisticTrainer",
]
