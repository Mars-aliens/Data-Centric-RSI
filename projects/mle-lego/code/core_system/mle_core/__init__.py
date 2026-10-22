"""Shared Action System substrate for the MLE-RSI idea prototypes."""

from .evolution import CandidateVersion, DataSystemVersion, FeedbackEvent
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
from .sklearn_components import TfidfLogisticEvaluator, TfidfLogisticTrainer
from .feedback import FeedbackCatalog, InMemoryFeedbackProvider

__all__ = [
    "ActionProposal",
    "ArtifactRef",
    "Budget",
    "CandidateVersion",
    "CoreRun",
    "DataSystemVersion",
    "FeedbackRecord",
    "FeedbackEvent",
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
