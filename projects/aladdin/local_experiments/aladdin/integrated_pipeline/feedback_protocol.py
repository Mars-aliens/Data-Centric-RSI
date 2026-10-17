from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from core_system.mle_core.models import TaskRecord

from .system import (
    FeedbackQuery,
    HumanFeedback,
    RequirementHypothesis,
    RequirementScorer,
    _profile_predictions,
)


@dataclass(frozen=True)
class PairwiseFeedbackQuery:
    left_record_id: str
    right_record_id: str
    score: float
    disagreement: float
    review_cost: float
    random_exploration: bool


def select_pairwise_feedback_queries(
    records: Sequence[TaskRecord],
    requirements: Sequence[RequirementHypothesis],
    weights: dict[str, float],
    scorer: RequirementScorer,
    *,
    max_queries: int,
    random_fraction: float,
    seed: int,
) -> tuple[PairwiseFeedbackQuery, ...]:
    if max_queries < 0 or not 0 <= random_fraction <= 1:
        raise ValueError("invalid pairwise query budget")
    if len(records) < 2 or max_queries == 0:
        return ()
    rows: list[PairwiseFeedbackQuery] = []
    for left_index, left in enumerate(records):
        left_predictions = _profile_predictions(left, requirements, weights, scorer)
        for right in records[left_index + 1 :]:
            right_predictions = _profile_predictions(right, requirements, weights, scorer)
            disagreement = sum(
                weights.get(item.hypothesis_id, 0.0)
                * (left_predictions[item.hypothesis_id] != right_predictions[item.hypothesis_id])
                for item in requirements
            )
            if disagreement <= 0:
                continue
            left_cost = float(left.metadata.get("review_cost", 1.0))
            right_cost = float(right.metadata.get("review_cost", 1.0))
            cost = left_cost + right_cost
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("pairwise review costs must be positive and finite")
            rows.append(
                PairwiseFeedbackQuery(
                    left.record_id,
                    right.record_id,
                    disagreement / cost,
                    disagreement,
                    cost,
                    False,
                )
            )
    ordered = sorted(rows, key=lambda row: (-row.score, row.left_record_id, row.right_record_id))
    reserve = min(max_queries, int(round(max_queries * random_fraction)))
    rng = random.Random(seed)
    random_rows = rng.sample(ordered, min(reserve, len(ordered))) if reserve else []
    selected = [PairwiseFeedbackQuery(**{**row.__dict__, "random_exploration": True}) for row in random_rows]
    selected_ids = {(row.left_record_id, row.right_record_id) for row in selected}
    selected.extend(
        row
        for row in ordered
        if (row.left_record_id, row.right_record_id) not in selected_ids
        and len(selected) < max_queries
    )
    return tuple(selected)


def pairwise_feedback_to_records(
    query: PairwiseFeedbackQuery,
    feedback: HumanFeedback,
) -> dict[str, HumanFeedback]:
    """Expand a human preference into two record-level training decisions."""
    if feedback.feedback_type != "pairwise":
        raise ValueError("pairwise query requires feedback_type='pairwise'")
    if feedback.preferred_record_id not in (
        query.left_record_id,
        query.right_record_id,
    ):
        raise ValueError("preferred_record_id must identify one compared record")
    chosen = feedback.preferred_record_id
    return {
        query.left_record_id: HumanFeedback(
            chosen == query.left_record_id,
            review_seconds=feedback.review_seconds / 2,
            feedback_type="pairwise",
            preferred_record_id=chosen,
            rationale=feedback.rationale,
        ),
        query.right_record_id: HumanFeedback(
            chosen == query.right_record_id,
            review_seconds=feedback.review_seconds / 2,
            feedback_type="pairwise",
            preferred_record_id=chosen,
            rationale=feedback.rationale,
        ),
    }


def update_requirement_weights_from_pairwise(
    requirements: Sequence[RequirementHypothesis],
    weights: dict[str, float],
    left: TaskRecord,
    right: TaskRecord,
    preferred_record_id: str,
    scorer: RequirementScorer,
    *,
    match_likelihood: float = 0.9,
) -> dict[str, float]:
    if preferred_record_id not in (left.record_id, right.record_id):
        raise ValueError("preferred_record_id must identify one compared record")
    if not 0.5 < match_likelihood < 1:
        raise ValueError("match_likelihood must be in (0.5, 1)")
    left_prediction = _profile_predictions(left, requirements, weights, scorer)
    right_prediction = _profile_predictions(right, requirements, weights, scorer)
    preferred_is_left = preferred_record_id == left.record_id
    posterior = {}
    for requirement in requirements:
        hid = requirement.hypothesis_id
        predicts_left = left_prediction[hid] and not right_prediction[hid]
        predicts_right = right_prediction[hid] and not left_prediction[hid]
        matches = predicts_left if preferred_is_left else predicts_right
        posterior[hid] = weights.get(hid, 0.0) * (
            match_likelihood if matches else 1.0 - match_likelihood
        )
    total = sum(posterior.values())
    if total <= 0:
        raise ValueError("pairwise feedback leaves no requirement hypothesis mass")
    return {key: value / total for key, value in posterior.items()}
