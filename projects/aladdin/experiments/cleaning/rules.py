"""Deterministic schema, repetition, and near-duplicate filtering."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CleaningPolicy:
    version: int = 0
    near_duplicate_threshold: float = 0.92
    repeated_ngram_threshold: float = 0.65
    min_input_chars: int = 1
    min_output_chars: int = 1

    def validate(self) -> None:
        if not 0.5 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0.5 and 1.0")
        if not 0.0 <= self.repeated_ngram_threshold <= 1.0:
            raise ValueError("repeated_ngram_threshold must be between 0 and 1")
        if self.min_input_chars < 1 or self.min_output_chars < 1:
            raise ValueError("minimum text lengths must be positive")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    normalized = normalize_text(text)
    if len(normalized) < n * 2:
        return 0.0
    ngrams = [normalized[index : index + n] for index in range(len(normalized) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _candidate_id(record: dict[str, Any], index: int) -> str:
    value = str(record.get("record_id") or "").strip()
    if value:
        return value
    canonical = repr(sorted((str(key), repr(value)) for key, value in record.items()))
    return "missing-id-" + hashlib.sha256(f"{index}:{canonical}".encode()).hexdigest()[:16]


def _near_duplicate_pairs(texts: list[str], threshold: float) -> dict[int, int]:
    if len(texts) < 2:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=False,
            min_df=1,
            norm="l2",
        ).fit_transform(texts)
    except ValueError:
        return {}

    duplicates: dict[int, int] = {}
    kept: list[int] = []
    for index in range(len(texts)):
        if not texts[index]:
            continue
        for earlier in kept:
            similarity = float(cosine_similarity(matrix[index], matrix[earlier])[0, 0])
            if similarity >= threshold:
                duplicates[index] = earlier
                break
        else:
            kept.append(index)
    return duplicates


def clean_candidates(
    records: list[dict[str, Any]], policy: CleaningPolicy
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return retained records, quarantine records, and one decision per input row."""
    policy.validate()
    normalized: list[tuple[str, str]] = []
    reasons: list[list[str]] = []
    ids: list[str] = []
    for index, record in enumerate(records):
        record_id = _candidate_id(record, index)
        ids.append(record_id)
        input_text = normalize_text(record.get("input"))
        output_text = normalize_text(record.get("output"))
        normalized.append((input_text, output_text))
        row_reasons: list[str] = []
        if len(input_text) < policy.min_input_chars:
            row_reasons.append("empty_or_short_input")
        if len(output_text) < policy.min_output_chars:
            row_reasons.append("empty_or_short_output")
        if input_text and output_text and input_text == output_text:
            row_reasons.append("output_duplicates_input")
        if output_text and repeated_ngram_ratio(output_text) > policy.repeated_ngram_threshold:
            row_reasons.append("excessive_repetition")
        reasons.append(row_reasons)

    exact_seen: dict[tuple[str, str], int] = {}
    duplicate_of_index: dict[int, int] = {}
    for index, pair in enumerate(normalized):
        if reasons[index] or not all(pair):
            continue
        if pair in exact_seen:
            reasons[index].append("exact_duplicate")
            duplicate_of_index[index] = exact_seen[pair]
        else:
            exact_seen[pair] = index

    eligible = [index for index, row_reasons in enumerate(reasons) if not row_reasons]
    duplicate_positions = _near_duplicate_pairs(
        [normalized[index][0] + " [SEP] " + normalized[index][1] for index in eligible],
        policy.near_duplicate_threshold,
    )
    for index, previous in duplicate_positions.items():
        duplicate_of_index[eligible[index]] = eligible[previous]
    near_duplicate_indices = {eligible[index] for index in duplicate_positions}
    for index in near_duplicate_indices:
        reasons[index].append("near_duplicate")

    retained: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        kept = not reasons[index]
        row = {
            **record,
            "record_id": ids[index],
            "cleaning": {
                "policy_version": policy.version,
                "status": "retained" if kept else "quarantined",
                "reason_codes": reasons[index],
                "duplicate_of": ids[duplicate_of_index[index]] if index in duplicate_of_index else None,
                "repeated_ngram_ratio": repeated_ngram_ratio(str(record.get("output") or "")),
                "policy": asdict(policy),
            },
        }
        decision = {
            "record_id": ids[index],
            "policy_version": policy.version,
            "keep": kept,
            "reason_codes": reasons[index],
            "duplicate_of": row["cleaning"]["duplicate_of"],
            "repeated_ngram_ratio": row["cleaning"]["repeated_ngram_ratio"],
        }
        decisions.append(decision)
        (retained if kept else quarantine).append(row)
    return retained, quarantine, decisions
