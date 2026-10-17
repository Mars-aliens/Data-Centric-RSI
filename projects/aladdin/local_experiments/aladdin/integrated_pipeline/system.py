from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core_system.mle_core.models import Budget, TaskRecord, to_jsonable
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)
from core_system.mle_core.store import ArtifactStore

from .calibration import ReliabilityCalibrator


class DataIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class RequirementHypothesis:
    hypothesis_id: str
    text: str


@dataclass
class RequirementState:
    hypotheses: list[RequirementHypothesis]
    weights: dict[str, float]
    confirmed_edits: list[str]
    edit_prior_mass: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.edit_prior_mass < 1:
            raise ValueError("edit_prior_mass must be in (0, 1)")
        ids = {item.hypothesis_id for item in self.hypotheses}
        if not self.hypotheses or len(ids) != len(self.hypotheses):
            raise ValueError("requirement hypotheses must be non-empty and uniquely named")
        if set(self.weights) != ids or any(value < 0 for value in self.weights.values()):
            raise ValueError("requirement weights must match hypotheses and be non-negative")
        total = sum(self.weights.values())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("requirement weights must have positive finite mass")
        self.weights = {key: value / total for key, value in self.weights.items()}

    def add_confirmed_edit(self, text: str) -> None:
        normalized = text.strip()
        if not normalized or normalized in self.confirmed_edits:
            return
        if any(item.text == normalized for item in self.hypotheses):
            self.confirmed_edits.append(normalized)
            return
        self.confirmed_edits.append(normalized)
        hypothesis_id = f"human-edit-{len(self.confirmed_edits):04d}"
        self.hypotheses.append(RequirementHypothesis(hypothesis_id, normalized))
        self.weights = {
            key: value * (1.0 - self.edit_prior_mass)
            for key, value in self.weights.items()
        }
        self.weights[hypothesis_id] = self.edit_prior_mass

    @classmethod
    def uniform(
        cls, hypotheses: Sequence[RequirementHypothesis]
    ) -> "RequirementState":
        items = list(hypotheses)
        if not items or len({item.hypothesis_id for item in items}) != len(items):
            raise ValueError("requirement hypotheses must be non-empty and uniquely named")
        return cls(
            hypotheses=items,
            weights={item.hypothesis_id: 1.0 / len(items) for item in items},
            confirmed_edits=[],
        )


@dataclass(frozen=True)
class QualityAssessment:
    score: float
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class HumanFeedback:
    accept_for_training: bool
    target: str | None = None
    edited_requirement: str | None = None
    review_seconds: float = 0.0
    feedback_type: str = "binary"
    preferred_record_id: str | None = None
    rationale: str | None = None
    demonstration: str | None = None

    def __post_init__(self) -> None:
        if self.feedback_type not in ("binary", "pairwise", "revision", "demonstration"):
            raise ValueError(f"Unknown human feedback type: {self.feedback_type}")
        if self.review_seconds < 0 or not math.isfinite(self.review_seconds):
            raise ValueError("review_seconds must be non-negative and finite")
        if self.feedback_type == "pairwise" and not self.preferred_record_id:
            raise ValueError("pairwise feedback requires preferred_record_id")
        if self.feedback_type == "demonstration" and not self.demonstration:
            raise ValueError("demonstration feedback requires demonstration text")


@dataclass(frozen=True)
class FeedbackQuery:
    record_id: str
    score: float
    disagreement: float
    expected_selection_change: float
    review_cost: float
    random_exploration: bool


@dataclass(frozen=True)
class RoundResult:
    run_id: str
    generator_strategy: str
    generated_count: int
    exact_duplicates_removed: int
    near_duplicate_pairs: tuple[tuple[str, str, float], ...]
    quality_counts: dict[str, int]
    queries: tuple[FeedbackQuery, ...]
    feedback_received: int
    selected_record_ids: tuple[str, ...]
    baseline_metrics: dict[str, float]
    augmented_metrics: dict[str, float]
    validation_gain: float
    synthesis_cost: float
    evidence_scope: str = "local smoke test; not efficacy evidence"
    selected_records: tuple[TaskRecord, ...] = ()
    training_seed_count: int = 0


class RequirementScorer(Protocol):
    def score(
        self, hypothesis: RequirementHypothesis, record: TaskRecord
    ) -> float: ...


class QualityScorer(Protocol):
    def assess(self, record: TaskRecord) -> QualityAssessment: ...


class CandidateGenerator(Protocol):
    def generate(
        self,
        *,
        seed_records: Sequence[TaskRecord],
        requirements: Sequence[RequirementHypothesis],
        confirmed_edits: Sequence[str],
        strategy: str,
        count: int,
        round_index: int,
        cost_budget: float | None,
    ) -> tuple[list[TaskRecord], Budget]: ...


class FeedbackReviewer(Protocol):
    def review(
        self, record: TaskRecord, query: FeedbackQuery
    ) -> HumanFeedback | None: ...


class QualityAuditor(Protocol):
    """Independent audit of an automatic quality action."""

    def audit(self, record: TaskRecord, proposed_decision: str) -> bool | None: ...


class DownstreamTrainer(Protocol):
    def train(
        self,
        *,
        train_records: list[TaskRecord],
        initial_checkpoint: dict[str, object] | None,
        config: dict[str, object],
        seed: int,
        store: ArtifactStore,
    ) -> tuple[dict[str, object], Budget]: ...


class DownstreamEvaluator(Protocol):
    def evaluate(
        self,
        *,
        checkpoint: dict[str, object],
        records: list[TaskRecord],
        config: dict[str, object],
    ) -> tuple[dict[str, float], Budget]: ...


class SynthesisBudgetAllocator:
    """UCB1 over a bounded, cost-normalized development reward."""

    def __init__(
        self,
        strategies: Sequence[str],
        exploration: float = 1.0,
        reward_decay: float = 0.8,
    ):
        if not strategies or len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be a non-empty unique sequence")
        if exploration < 0 or not math.isfinite(exploration):
            raise ValueError("exploration must be finite and non-negative")
        if not 0 < reward_decay <= 1:
            raise ValueError("reward_decay must be in (0, 1]")
        self.strategies = tuple(strategies)
        self.exploration = exploration
        self.reward_decay = reward_decay
        self.counts = {strategy: 0 for strategy in self.strategies}
        self.reward_per_cost = {strategy: 0.0 for strategy in self.strategies}

    def choose(self) -> str:
        for strategy in self.strategies:
            if self.counts[strategy] == 0:
                return strategy
        total = sum(self.counts.values())
        return max(
            self.strategies,
            key=lambda strategy: (
                self.reward_per_cost[strategy]
                + self.exploration
                * math.sqrt(math.log(total) / self.counts[strategy]),
                strategy,
            ),
        )

    def update(self, strategy: str, validation_gain: float, cost: float) -> None:
        if strategy not in self.counts:
            raise KeyError(f"Unknown synthesis strategy: {strategy}")
        if not math.isfinite(validation_gain):
            raise ValueError("validation_gain must be finite")
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("cost must be positive and finite")
        if cost < 1.0:
            raise ValueError("cost must use units of at least one token/call/example")
        if not -1.0 <= validation_gain <= 1.0:
            raise ValueError("accuracy gain must be in [-1, 1]")
        reward = (validation_gain + 1.0) / (2.0 * cost)
        self.reward_per_cost[strategy] = (
            self.reward_decay * self.reward_per_cost[strategy]
            + (1.0 - self.reward_decay) * reward
            if self.counts[strategy]
            else reward
        )
        self.counts[strategy] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "strategies": list(self.strategies),
            "exploration": self.exploration,
            "reward_decay": self.reward_decay,
            "counts": dict(self.counts),
            "reward_per_cost": dict(self.reward_per_cost),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SynthesisBudgetAllocator":
        allocator = cls(
            [str(item) for item in value["strategies"]],
            exploration=float(value.get("exploration", 1.0)),
            reward_decay=float(value.get("reward_decay", 1.0)),
        )
        counts = value.get("counts", {})
        rewards = value.get("reward_per_cost", {})
        for strategy in allocator.strategies:
            allocator.counts[strategy] = int(counts.get(strategy, 0))
            allocator.reward_per_cost[strategy] = float(rewards.get(strategy, 0.0))
        return allocator


def clean_and_deduplicate(
    records: Sequence[TaskRecord],
) -> tuple[list[TaskRecord], int]:
    """Validate IDs/split provenance, remove exact duplicates, reject conflicts."""
    ids: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    exact: dict[tuple[str, str], TaskRecord] = {}
    kept: list[TaskRecord] = []
    duplicates = 0
    for record in records:
        if not record.record_id or record.record_id in ids:
            raise DataIntegrityError(f"Missing or duplicate record_id: {record.record_id!r}")
        ids.add(record.record_id)
        if record.split == "final_test":
            raise DataIntegrityError("final_test records are not allowed in this pipeline")
        if record.source_group_id:
            key = (record.task_id, record.source_group_id)
            previous_split = group_splits.setdefault(key, record.split)
            if previous_split != record.split:
                raise DataIntegrityError(
                    f"Source group {record.source_group_id!r} crosses splits"
                )
        normalized = re.sub(r"\s+", " ", str(record.input)).strip().casefold()
        if not normalized:
            raise DataIntegrityError(f"Empty input in record {record.record_id}")
        exact_key = (record.task_id, normalized)
        old = exact.get(exact_key)
        if old is not None:
            if old.split != record.split:
                raise DataIntegrityError(
                    f"Exact input appears across splits: {old.record_id}, {record.record_id}"
                )
            if old.target != record.target:
                raise DataIntegrityError(
                    f"Conflicting targets for exact duplicate: {old.record_id}, {record.record_id}"
                )
            duplicates += 1
            continue
        exact[exact_key] = record
        kept.append(record)
    return kept, duplicates


def find_near_duplicate_pairs(
    records: Sequence[TaskRecord], *, threshold: float = 0.9
) -> tuple[tuple[str, str, float], ...]:
    """Flag high-TF-IDF-cosine pairs for review; never silently drops them."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if len(records) < 2:
        return ()
    texts = [str(item.input) for item in records]
    try:
        matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return ()
        raise
    similarities = cosine_similarity(matrix)
    pairs = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left].split != records[right].split:
                continue
            similarity = float(similarities[left, right])
            if similarity >= threshold:
                pairs.append(
                    (records[left].record_id, records[right].record_id, similarity)
                )
    return tuple(pairs)


def quality_gate(
    records: Sequence[TaskRecord],
    scorer: QualityScorer,
    *,
    accept_threshold: float,
    min_confidence: float,
    calibrator: ReliabilityCalibrator | None = None,
) -> tuple[dict[str, str], dict[str, QualityAssessment]]:
    if not 0 <= accept_threshold <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("quality thresholds must be in [0, 1]")
    decisions: dict[str, str] = {}
    assessments: dict[str, QualityAssessment] = {}
    for record in records:
        assessment = scorer.assess(record)
        if not (0 <= assessment.score <= 1 and 0 <= assessment.confidence <= 1):
            raise ValueError(f"Quality assessment out of range for {record.record_id}")
        assessments[record.record_id] = assessment
        if assessment.confidence < min_confidence:
            proposed = "defer"
        elif assessment.score >= accept_threshold:
            proposed = "accept"
        else:
            proposed = "reject"
        decisions[record.record_id] = (
            calibrator.route(record, proposed) if calibrator is not None else proposed
        )
    return decisions, assessments


def _profile_predictions(
    record: TaskRecord,
    requirements: Sequence[RequirementHypothesis],
    weights: dict[str, float],
    scorer: RequirementScorer,
) -> dict[str, bool]:
    if not requirements:
        raise ValueError("at least one requirement hypothesis is required")
    predictions = {}
    for item in requirements:
        score = scorer.score(item, record)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                f"Requirement score for {item.hypothesis_id} must be finite in [0, 1]"
            )
        predictions[item.hypothesis_id] = score >= 0.5
    return predictions


def _weighted_top_k(
    records: Sequence[TaskRecord],
    requirements: Sequence[RequirementHypothesis],
    weights: dict[str, float],
    scorer: RequirementScorer,
    k: int,
) -> set[str]:
    ranked = []
    total_weight = sum(weights.get(item.hypothesis_id, 0.0) for item in requirements)
    for record in records:
        predictions = _profile_predictions(record, requirements, weights, scorer)
        expected = sum(
            weights.get(item.hypothesis_id, 0.0)
            * float(predictions[item.hypothesis_id])
            for item in requirements
        ) / max(total_weight, 1e-12)
        ranked.append((expected, record.record_id))
    return {
        record_id
        for _, record_id in sorted(ranked, key=lambda item: (-item[0], item[1]))[:k]
    }


def select_feedback_queries(
    query_pool: Sequence[TaskRecord],
    downstream_pool: Sequence[TaskRecord],
    requirements: Sequence[RequirementHypothesis],
    weights: dict[str, float],
    scorer: RequirementScorer,
    *,
    max_queries: int,
    downstream_selection_limit: int,
    random_fraction: float,
    seed: int,
    cost_budget: float | None = None,
) -> tuple[FeedbackQuery, ...]:
    """Rank queries by requirement disagreement x expected downstream set change / effort."""
    if max_queries < 0:
        raise ValueError("max_queries cannot be negative")
    if downstream_selection_limit < 0:
        raise ValueError("downstream_selection_limit cannot be negative")
    if not 0 <= random_fraction <= 1:
        raise ValueError("random_fraction must be in [0, 1]")
    if cost_budget is not None and (cost_budget < 0 or not math.isfinite(cost_budget)):
        raise ValueError("cost_budget must be finite and non-negative")
    if not query_pool or max_queries == 0:
        return ()
    if any(item.split == "final_test" for item in (*query_pool, *downstream_pool)):
        raise DataIntegrityError("final_test cannot be used for query or selection policy")
    unknown = set(weights) - {item.hypothesis_id for item in requirements}
    if unknown:
        raise ValueError(f"Weights refer to unknown hypotheses: {sorted(unknown)}")
    base = _weighted_top_k(
        downstream_pool,
        requirements,
        weights,
        scorer,
        min(downstream_selection_limit, len(downstream_pool)),
    )
    rows: list[FeedbackQuery] = []
    for record in query_pool:
        predictions = _profile_predictions(record, requirements, weights, scorer)
        total = sum(weights.get(item.hypothesis_id, 0.0) for item in requirements)
        p_accept = sum(
            weights.get(item.hypothesis_id, 0.0)
            for item in requirements
            if predictions[item.hypothesis_id]
        ) / max(total, 1e-12)
        disagreement = 4.0 * p_accept * (1.0 - p_accept)
        expected_change = 0.0
        downstream_k = min(downstream_selection_limit, len(downstream_pool))
        for answer, probability in ((False, 1.0 - p_accept), (True, p_accept)):
            if probability <= 0:
                continue
            posterior = {
                item.hypothesis_id: weights.get(item.hypothesis_id, 0.0)
                for item in requirements
                if predictions[item.hypothesis_id] is answer
            }
            if not any(posterior.values()):
                continue
            changed = _weighted_top_k(
                downstream_pool,
                requirements,
                posterior,
                scorer,
                downstream_k,
            )
            expected_change += probability * len(base.symmetric_difference(changed))
        effort = float(record.metadata.get("review_cost", 1.0))
        if not math.isfinite(effort) or effort <= 0:
            raise ValueError(f"Invalid review_cost for {record.record_id}")
        rows.append(
            FeedbackQuery(
                record_id=record.record_id,
                score=disagreement * expected_change / effort,
                disagreement=disagreement,
                expected_selection_change=expected_change,
                review_cost=effort,
                random_exploration=False,
            )
        )
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: row.record_id)
    reserve_count = min(max_queries, int(round(max_queries * random_fraction)))
    if random_fraction > 0 and max_queries > 0 and reserve_count == 0:
        reserve_count = 1
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    random_rows: list[FeedbackQuery] = []
    spent = 0.0
    random_cost_cap = (
        cost_budget * random_fraction if cost_budget is not None else math.inf
    )
    for item in shuffled:
        if len(random_rows) >= reserve_count:
            break
        if spent + item.review_cost <= random_cost_cap + 1e-9:
            random_rows.append(item)
            spent += item.review_cost
    random_ids = {item.record_id for item in random_rows}
    exploitation = sorted(
        (item for item in rows if item.record_id not in random_ids),
        key=lambda row: (-row.score, row.record_id),
    )
    selected = [replace(item, random_exploration=True) for item in random_rows]
    for item in exploitation:
        if len(selected) >= max_queries:
            break
        if cost_budget is None or spent + item.review_cost <= cost_budget + 1e-9:
            selected.append(item)
            spent += item.review_cost
    return tuple(selected)


def update_requirement_weights(
    requirements: Sequence[RequirementHypothesis],
    weights: dict[str, float],
    record: TaskRecord,
    feedback: HumanFeedback,
    scorer: RequirementScorer,
    *,
    match_likelihood: float = 0.9,
) -> dict[str, float]:
    if not 0.5 < match_likelihood < 1:
        raise ValueError("match_likelihood must be in (0.5, 1)")
    predictions = _profile_predictions(record, requirements, weights, scorer)
    updated = {
        hypothesis.hypothesis_id: weights.get(hypothesis.hypothesis_id, 0.0)
        * (
            match_likelihood
            if predictions[hypothesis.hypothesis_id] == feedback.accept_for_training
            else 1.0 - match_likelihood
        )
        for hypothesis in requirements
    }
    total = sum(updated.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Requirement posterior collapsed")
    return {key: value / total for key, value in updated.items()}


def select_mmr(
    records: Sequence[TaskRecord],
    relevance: dict[str, float],
    *,
    limit: int,
    relevance_weight: float = 0.7,
) -> tuple[TaskRecord, ...]:
    """Maximal Marginal Relevance selection over TF-IDF cosine similarities."""
    if limit < 0 or not 0 <= relevance_weight <= 1:
        raise ValueError("invalid MMR limit or relevance_weight")
    candidates = list(records)
    if limit == 0 or not candidates:
        return ()
    texts = [f"{item.input}\n{item.target or ''}" for item in candidates]
    matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
    similarities = cosine_similarity(matrix)
    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected_indices) < limit:
        def score(index: int) -> tuple[float, str]:
            redundancy = max(
                (float(similarities[index, chosen]) for chosen in selected_indices),
                default=0.0,
            )
            value = (
                relevance_weight * relevance.get(candidates[index].record_id, 0.0)
                - (1.0 - relevance_weight) * redundancy
            )
            return value, candidates[index].record_id

        chosen = min(remaining, key=lambda index: (-score(index)[0], score(index)[1]))
        selected_indices.append(chosen)
        remaining.remove(chosen)
    return tuple(candidates[index] for index in selected_indices)


def _with_target(record: TaskRecord, target: str, feedback: HumanFeedback) -> TaskRecord:
    return replace(
        record,
        target=target,
        metadata={
            **record.metadata,
            "human_feedback_received": True,
            "human_review_seconds": feedback.review_seconds,
        },
    )


def run_round(
    *,
    run_id: str,
    round_index: int,
    seed_records: Sequence[TaskRecord],
    development_records: Sequence[TaskRecord],
    requirement_state: RequirementState,
    requirement_scorer: RequirementScorer,
    quality_scorer: QualityScorer,
    generator: CandidateGenerator,
    reviewer: FeedbackReviewer,
    allocator: SynthesisBudgetAllocator,
    output_dir: str | Path,
    candidate_count: int = 12,
    feedback_budget: int = 4,
    feedback_cost_budget: float | None = None,
    generation_cost_budget: float | None = None,
    selected_data_limit: int = 8,
    random_feedback_fraction: float = 0.25,
    quality_threshold: float = 0.6,
    quality_min_confidence: float = 0.7,
    quality_calibrator: ReliabilityCalibrator | None = None,
    quality_auditor: QualityAuditor | None = None,
    quality_evaluator_state: object | None = None,
    task_model_config: dict[str, object] | None = None,
    task_trainer: DownstreamTrainer | None = None,
    task_evaluator: DownstreamEvaluator | None = None,
    model_seed: int = 17,
) -> RoundResult:
    """Run one auditable local round; only development is used for utility feedback."""
    if not seed_records or not development_records:
        raise ValueError("seed and development records must be non-empty")
    if any(item.split != "seed" for item in seed_records):
        raise DataIntegrityError("seed_records must all belong to split 'seed'")
    if any(item.split != "development" for item in development_records):
        raise DataIntegrityError("development_records must all belong to split 'development'")
    if any(item.target is None for item in seed_records):
        raise DataIntegrityError("seed records must have targets")
    for name, value in (
        ("feedback_cost_budget", feedback_cost_budget),
        ("generation_cost_budget", generation_cost_budget),
    ):
        if value is not None and (value < 0 or not math.isfinite(value)):
            raise ValueError(f"{name} must be finite and non-negative")
    requirements = requirement_state.hypotheses
    if not requirements:
        raise ValueError("requirements cannot be empty")
    profiles = dict(requirement_state.weights)
    if set(profiles) != {item.hypothesis_id for item in requirements}:
        raise ValueError("requirement weights do not match active hypotheses")
    strategy = allocator.choose()
    source_records, source_duplicates = clean_and_deduplicate(
        [*seed_records, *development_records]
    )
    seed_ids_original = {item.record_id for item in seed_records}
    development_ids_original = {item.record_id for item in development_records}
    seed_records = [item for item in source_records if item.record_id in seed_ids_original]
    development_records = [
        item for item in source_records if item.record_id in development_ids_original
    ]
    store = ArtifactStore(output_dir)
    store.append_event(
        "integrated_round_started",
        {"run_id": run_id, "round_index": round_index, "strategy": strategy},
    )
    seed_ref = store.put_json(
        kind="seed-dataset",
        value=to_jsonable(seed_records),
        version=f"seed-round-{round_index}",
    )
    generated, generation_budget = generator.generate(
        seed_records=seed_records,
        requirements=requirements,
        confirmed_edits=tuple(requirement_state.confirmed_edits),
        strategy=strategy,
        count=candidate_count,
        round_index=round_index,
        cost_budget=generation_cost_budget,
    )
    if generation_cost_budget is not None and generation_budget.llm_tokens > generation_cost_budget:
        raise DataIntegrityError("generator exceeded its declared token budget")
    if any(item.split != "seed" for item in generated):
        raise DataIntegrityError("generated training records must inherit the seed split")
    seed_ids = {item.record_id for item in seed_records}
    seed_by_id = {item.record_id: item for item in seed_records}
    if any(
        not item.parent_record_ids
        or not set(item.parent_record_ids).issubset(seed_ids)
        or any(seed_by_id[parent].split != item.split for parent in item.parent_record_ids)
        for item in generated
    ):
        raise DataIntegrityError("generated records must cite seed parents only")
    generated_ref = store.put_json(
        kind="candidate-pool",
        value=to_jsonable(generated),
        version=f"{strategy}-round-{round_index}",
        parents=[seed_ref.artifact_id],
    )
    store.append_event(
        "candidate_generation_completed",
        {
            "strategy": strategy,
            "count": len(generated),
            "cost": generation_budget,
            "cost_budget": generation_cost_budget,
            "lineage_repairs": int(getattr(generator, "last_lineage_repairs", 0)),
            "lineage_drops": int(getattr(generator, "last_lineage_drops", 0)),
            "artifact_id": generated_ref.artifact_id,
        },
    )

    clean_records, generated_duplicates = clean_and_deduplicate(
        [*seed_records, *development_records, *generated]
    )
    duplicates = source_duplicates + generated_duplicates
    near_duplicates = find_near_duplicate_pairs(clean_records)
    seed_ids_clean = {item.record_id for item in seed_records}
    development_ids_clean = {item.record_id for item in development_records}
    candidates = [
        item
        for item in clean_records
        if item.record_id not in seed_ids_clean | development_ids_clean
    ]
    proposed_quality_decisions, assessments = quality_gate(
        candidates,
        quality_scorer,
        accept_threshold=quality_threshold,
        min_confidence=quality_min_confidence,
    )
    calibration_before = (
        quality_calibrator.summary() if quality_calibrator is not None else None
    )
    quality_decisions = {
        record.record_id: (
            quality_calibrator.route(
                record, proposed_quality_decisions[record.record_id]
            )
            if quality_calibrator is not None
            else proposed_quality_decisions[record.record_id]
        )
        for record in candidates
    }
    independent_quality_audits = 0
    if quality_calibrator is not None and quality_auditor is not None:
        for record in candidates:
            proposed = proposed_quality_decisions[record.record_id]
            if proposed not in ("accept", "reject"):
                continue
            human_accept = quality_auditor.audit(record, proposed)
            if human_accept is None:
                continue
            quality_calibrator.add_audit(
                quality_calibrator.slice_for(record), proposed, human_accept
            )
            if quality_evaluator_state is not None and hasattr(
                quality_evaluator_state, "record_audit"
            ):
                quality_evaluator_state.record_audit(
                    record, proposed, human_accept
                )
            if (
                hasattr(quality_scorer, "set_prompt_context")
                and quality_evaluator_state is not None
            ):
                quality_scorer.set_prompt_context(
                    quality_evaluator_state.prompt_context()
                )
            independent_quality_audits += 1
    store.append_event(
        "quality_gate_completed",
        {
            "proposed_decisions": proposed_quality_decisions,
            "effective_decisions": quality_decisions,
            "calibration": calibration_before,
            "near_duplicate_pairs": near_duplicates,
        },
    )
    query_candidates = [
        item for item in candidates if quality_decisions[item.record_id] != "reject"
    ]
    queries = select_feedback_queries(
        query_candidates,
        query_candidates,
        requirements,
        profiles,
        requirement_scorer,
        max_queries=feedback_budget,
        downstream_selection_limit=selected_data_limit,
        random_fraction=random_feedback_fraction,
        seed=model_seed + round_index,
        cost_budget=feedback_cost_budget,
    )
    candidate_by_id = {item.record_id: item for item in candidates}
    feedback_by_id: dict[str, HumanFeedback] = {}
    for query in queries:
        store.append_event("feedback_query_selected", asdict(query))
        feedback = reviewer.review(candidate_by_id[query.record_id], query)
        if feedback is None:
            continue
        feedback_by_id[query.record_id] = feedback
        store.append_event(
            "feedback_received",
            {"record_id": query.record_id, "feedback": feedback},
        )
        requirement_state.weights = update_requirement_weights(
            requirements,
            requirement_state.weights,
            candidate_by_id[query.record_id],
            feedback,
            requirement_scorer,
        )
        if feedback.edited_requirement:
            requirement_state.add_confirmed_edit(feedback.edited_requirement)
            requirements = requirement_state.hypotheses

    trainable: list[TaskRecord] = []
    relevance: dict[str, float] = {}
    for record in candidates:
        assessment = assessments[record.record_id]
        decision = quality_decisions[record.record_id]
        feedback = feedback_by_id.get(record.record_id)
        if decision == "reject" and feedback is None:
            continue
        if decision == "defer" and (
            feedback is None or not feedback.accept_for_training
        ):
            continue
        if feedback is not None and not feedback.accept_for_training:
            continue
        selected_record = record
        if feedback is not None and feedback.target is not None:
            selected_record = _with_target(record, feedback.target, feedback)
        if selected_record.target is None:
            continue
        trainable.append(selected_record)
        predictions = _profile_predictions(
            record, requirements, requirement_state.weights, requirement_scorer
        )
        expected_relevance = sum(
            requirement_state.weights.get(item.hypothesis_id, 0.0)
            * float(predictions[item.hypothesis_id])
            for item in requirements
        )
        relevance[record.record_id] = 0.5 * assessment.score + 0.5 * expected_relevance

    chosen = select_mmr(
        trainable,
        relevance,
        limit=selected_data_limit,
    )
    augmented_records = [*seed_records, *chosen]
    baseline_dataset_ref = store.put_json(
        kind="training-dataset",
        value=to_jsonable(seed_records),
        version=f"seed-only-round-{round_index}",
        parents=[seed_ref.artifact_id],
    )
    feedback_ref = store.put_json(
        kind="human-feedback",
        value={key: asdict(value) for key, value in feedback_by_id.items()},
        version=f"feedback-round-{round_index}",
        parents=[generated_ref.artifact_id],
    )
    augmented_dataset_ref = store.put_json(
        kind="training-dataset",
        value=to_jsonable(augmented_records),
        version=f"augmented-round-{round_index}",
        parents=[seed_ref.artifact_id, generated_ref.artifact_id, feedback_ref.artifact_id],
    )
    store.append_event(
        "training_dataset_selected",
        {
            "record_ids": [item.record_id for item in chosen],
            "baseline_dataset_id": baseline_dataset_ref.artifact_id,
            "augmented_dataset_id": augmented_dataset_ref.artifact_id,
        },
    )
    model_config = dict(task_model_config or {})
    trainer = task_trainer or TfidfLogisticTrainer()
    evaluator = task_evaluator or TfidfLogisticEvaluator()
    baseline_checkpoint, baseline_cost = trainer.train(
        train_records=list(seed_records),
        initial_checkpoint=None,
        config=model_config,
        seed=model_seed,
        store=store,
    )
    baseline_ref = store.put_json(
        kind="checkpoint",
        value=baseline_checkpoint,
        version=trainer.version,
        parents=[baseline_dataset_ref.artifact_id],
    )
    augmented_checkpoint, augmented_cost = trainer.train(
        train_records=augmented_records,
        initial_checkpoint=None,
        config=model_config,
        seed=model_seed,
        store=store,
    )
    augmented_ref = store.put_json(
        kind="checkpoint",
        value=augmented_checkpoint,
        version=trainer.version,
        parents=[augmented_dataset_ref.artifact_id],
    )
    store.append_event(
        "training_completed",
        {
            "baseline_checkpoint_id": baseline_ref.artifact_id,
            "augmented_checkpoint_id": augmented_ref.artifact_id,
            "baseline_cost": baseline_cost,
            "augmented_cost": augmented_cost,
        },
    )
    baseline_metrics, baseline_eval_cost = evaluator.evaluate(
        checkpoint=baseline_checkpoint,
        records=list(development_records),
        config={},
    )
    augmented_metrics, augmented_eval_cost = evaluator.evaluate(
        checkpoint=augmented_checkpoint,
        records=list(development_records),
        config={},
    )
    gain = augmented_metrics["accuracy"] - baseline_metrics["accuracy"]
    generation_cost = float(
        generation_budget.llm_tokens
        or generation_budget.llm_calls
        or max(1, len(generated))
    )
    allocator.update(strategy, gain, generation_cost)
    store.append_event(
        "development_evaluation_completed",
        {
            "baseline_metrics": baseline_metrics,
            "augmented_metrics": augmented_metrics,
            "gain": gain,
            "split": "development",
        },
    )
    report = {
        "run_id": run_id,
        "round_index": round_index,
        "strategy": strategy,
        "requirements": [asdict(item) for item in requirement_state.hypotheses],
        "requirement_weights": requirement_state.weights,
        "confirmed_requirement_edits": requirement_state.confirmed_edits,
        "quality_decisions": quality_decisions,
        "proposed_quality_decisions": proposed_quality_decisions,
        "quality_calibration_before": calibration_before,
        "quality_calibration_after": (
            quality_calibrator.summary() if quality_calibrator is not None else None
        ),
        "quality_evaluator_state": (
            quality_evaluator_state.to_dict()
            if quality_evaluator_state is not None
            and hasattr(quality_evaluator_state, "to_dict")
            else None
        ),
        "independent_quality_audits_added": independent_quality_audits,
        "automatic_quality_decisions": sum(
            decision in ("accept", "reject") for decision in quality_decisions.values()
        ),
        "automatic_quality_coverage": (
            sum(decision in ("accept", "reject") for decision in quality_decisions.values())
            / len(quality_decisions)
            if quality_decisions
            else 0.0
        ),
        "near_duplicate_pairs_for_review": near_duplicates,
        "queries": [asdict(item) for item in queries],
        "feedback_received": len(feedback_by_id),
        "selected_record_ids": [item.record_id for item in chosen],
        "baseline_metrics": baseline_metrics,
        "augmented_metrics": augmented_metrics,
        "validation_gain": gain,
        "generation_cost_proxy": generation_cost,
        "generation_cost_budget": generation_cost_budget,
        "lineage_repairs": int(getattr(generator, "last_lineage_repairs", 0)),
        "lineage_drops": int(getattr(generator, "last_lineage_drops", 0)),
        "feedback_cost_budget": feedback_cost_budget,
        "estimated_feedback_cost": sum(item.review_cost for item in queries),
        "human_review_budget_overrun": (
            max(
                0.0,
                sum(item.review_seconds for item in feedback_by_id.values())
                - feedback_cost_budget,
            )
            if feedback_cost_budget is not None
            else None
        ),
        "budget": {
            "generation": to_jsonable(generation_budget),
            "requirement_scoring": to_jsonable(
                getattr(requirement_scorer, "budget", Budget())
            ),
            "quality_scoring": to_jsonable(
                getattr(quality_scorer, "budget", Budget())
            ),
            "baseline_training": to_jsonable(baseline_cost),
            "augmented_training": to_jsonable(augmented_cost),
            "baseline_evaluation": to_jsonable(baseline_eval_cost),
            "augmented_evaluation": to_jsonable(augmented_eval_cost),
            "human_review_seconds": sum(
                item.review_seconds for item in feedback_by_id.values()
            ),
        },
        "artifacts": [
            seed_ref.artifact_id,
            generated_ref.artifact_id,
            feedback_ref.artifact_id,
            baseline_dataset_ref.artifact_id,
            augmented_dataset_ref.artifact_id,
            baseline_ref.artifact_id,
            augmented_ref.artifact_id,
        ],
        "final_test_accessed": False,
    }
    store.append_event("integrated_round_completed", report)
    store.write_run_snapshot(run_id, report)
    counts = {key: 0 for key in ("accept", "defer", "reject")}
    for decision in quality_decisions.values():
        counts[decision] += 1
    return RoundResult(
        run_id=run_id,
        generator_strategy=strategy,
        generated_count=len(generated),
        exact_duplicates_removed=duplicates,
        near_duplicate_pairs=near_duplicates,
        quality_counts=counts,
        queries=queries,
        feedback_received=len(feedback_by_id),
        selected_record_ids=tuple(item.record_id for item in chosen),
        baseline_metrics=baseline_metrics,
        augmented_metrics=augmented_metrics,
        validation_gain=gain,
        synthesis_cost=generation_cost,
        selected_records=tuple(chosen),
        training_seed_count=len(seed_records),
    )
