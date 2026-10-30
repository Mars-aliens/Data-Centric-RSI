"""Leakage-aware calibration and query selection policies."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any


def split_development_groups(
    records: list[dict[str, Any]], *, calibration_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole source groups into selector calibration and outcome evaluation."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    groups = sorted(
        {str(row.get("source_group_id")) if row.get("source_group_id") is not None else "" for row in records}
    )
    if len(groups) < 2 or "" in groups:
        raise ValueError("development data needs at least two non-empty source groups")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    calibration_count = min(len(groups) - 1, max(1, round(len(groups) * calibration_fraction)))
    calibration_groups = set(shuffled[:calibration_count])
    calibration = [row for row in records if str(row.get("source_group_id")) in calibration_groups]
    evaluation = [row for row in records if str(row.get("source_group_id")) not in calibration_groups]
    if {row["source_group_id"] for row in calibration} & {
        row["source_group_id"] for row in evaluation
    }:
        raise RuntimeError("development group split leaked source groups")
    return calibration, evaluation


def _feature_text(record: dict[str, Any]) -> str:
    context = record.get("context") or []
    context_text = []
    for turn in context:
        if isinstance(turn, dict):
            content = str(turn.get("content") or "").strip()
        else:
            content = str(turn).strip()
        if content:
            context_text.append(content)
    return "\n".join(
        [*context_text, str(record.get("input") or "").strip(), str(record.get("output") or "").strip()]
    )


class PreferenceUncertaintyModel:
    """A frozen development-calibrated preference proxy for query uncertainty."""

    def __init__(self) -> None:
        self.vectorizer: Any | None = None
        self.classifier: Any | None = None

    def fit(
        self, records: list[dict[str, Any]], *, seed: int = 17
    ) -> "PreferenceUncertaintyModel":
        labels = [row.get("if_chosen") for row in records]
        if not records or any(not isinstance(label, bool) for label in labels):
            raise ValueError("selector calibration rows must have boolean if_chosen labels")
        if len(set(labels)) != 2:
            raise ValueError("selector calibration requires both chosen and rejected examples")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=150_000,
            sublinear_tf=True,
        )
        features = self.vectorizer.fit_transform([_feature_text(row) for row in records])
        self.classifier = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=seed
        )
        self.classifier.fit(features, labels)
        return self

    def uncertainty(self, record: dict[str, Any]) -> float:
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("selector uncertainty model has not been fitted")
        features = self.vectorizer.transform([_feature_text(record)])
        probabilities = self.classifier.predict_proba(features)[0]
        if len(probabilities) != 2:
            raise RuntimeError("preference proxy did not produce a binary probability")
        p = min(1.0, max(0.0, float(probabilities[1])))
        if p in (0.0, 1.0):
            return 0.0
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def _stable_query_id(run_id: str, arm_id: str, round_index: int, record_id: str) -> str:
    value = f"{run_id}:{arm_id}:{round_index}:{record_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def select_queries(
    records: list[dict[str, Any]],
    *,
    method: str,
    count: int,
    seed: int,
    run_id: str,
    arm_id: str,
    round_index: int,
    uncertainty_model: PreferenceUncertaintyModel,
    estimated_review_seconds: float,
    impact_weight: float = 0.5,
    quarantine_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    if method not in {"random", "uncertainty", "feedback-value"}:
        raise ValueError(f"unknown feedback selection method: {method}")
    if (
        count <= 0
        or estimated_review_seconds <= 0
        or impact_weight < 0
        or not 0.0 <= quarantine_fraction <= 1.0
    ):
        raise ValueError("count, estimated_review_seconds, and impact_weight must be valid")
    if len({str(row.get("record_id") or "") for row in records}) != len(records):
        raise ValueError("query candidates must have unique record_id values")
    if not records:
        return []

    quarantined = [
        row for row in records
        if (row.get("cleaning") or {}).get("status") == "quarantined"
    ]
    retained = [
        row for row in records
        if (row.get("cleaning") or {}).get("status") != "quarantined"
    ]
    target_count = min(count, len(records))
    target_quarantine = math.ceil(target_count * quarantine_fraction)
    quarantine_count = min(len(quarantined), target_quarantine)
    retained_count = min(len(retained), target_count - quarantine_count)
    unfilled = target_count - quarantine_count - retained_count
    add_retained = min(unfilled, len(retained) - retained_count)
    retained_count += add_retained
    unfilled -= add_retained
    quarantine_count += min(unfilled, len(quarantined) - quarantine_count)

    group_counts = Counter(str(row.get("source_group_id") or row.get("record_id")) for row in records)
    ranked = []
    for stratum, pool, quota in (
        ("retained", retained, retained_count),
        ("quarantined", quarantined, quarantine_count),
    ):
        stratum_rows = []
        for record in pool:
            record_id = str(record.get("record_id") or "")
            uncertainty = uncertainty_model.uncertainty(record)
            affected_count = group_counts[str(record.get("source_group_id") or record_id)]
            score = uncertainty
            if method == "feedback-value":
                score = uncertainty * (1.0 + impact_weight * math.log1p(affected_count)) / estimated_review_seconds
            stratum_rows.append((score, record_id, uncertainty, affected_count, record))
        if method == "random":
            random.Random(seed + (1 if stratum == "quarantined" else 0)).shuffle(stratum_rows)
        else:
            stratum_rows.sort(key=lambda row: (-row[0], row[1]))
        ranked.extend(stratum_rows[:quota])
    queries = []
    for score, record_id, uncertainty, affected_count, record in ranked:
        cleaning = record.get("cleaning") or {}
        queries.append(
            {
                "query_id": _stable_query_id(run_id, arm_id, round_index, record_id),
                "arm_id": arm_id,
                "round": round_index,
                "selection_method": method,
                "record_id": record_id,
                "source_group_id": str(record.get("source_group_id") or ""),
                "candidate": record,
                "candidate_was_quarantined": cleaning.get("status") == "quarantined",
                "cleaning_reason_codes": list(cleaning.get("reason_codes") or []),
                "uncertainty": uncertainty,
                "affected_candidate_count": affected_count,
                "estimated_review_seconds": estimated_review_seconds,
                "selection_score": score,
            }
        )
    return queries
