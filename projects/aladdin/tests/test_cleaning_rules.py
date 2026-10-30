from experiments.cleaning.rules import CleaningPolicy, clean_candidates, repeated_ngram_ratio


def test_cleaner_keeps_first_exact_record_and_quarantines_duplicate():
    rows = [
        {"record_id": "a", "input": "A question", "output": "A clear answer", "source_group_id": "u1"},
        {"record_id": "b", "input": "  A QUESTION ", "output": "a clear answer", "source_group_id": "u2"},
    ]
    retained, quarantine, decisions = clean_candidates(rows, CleaningPolicy())

    assert [row["record_id"] for row in retained] == ["a"]
    assert quarantine[0]["cleaning"]["reason_codes"] == ["exact_duplicate"]
    assert decisions[1]["duplicate_of"] == "a"


def test_cleaner_quarantines_empty_and_identical_input_output():
    rows = [
        {"record_id": "empty", "input": "", "output": "answer"},
        {"record_id": "same", "input": "repeat", "output": "repeat"},
    ]
    retained, quarantine, _ = clean_candidates(rows, CleaningPolicy())

    assert not retained
    assert quarantine[0]["cleaning"]["reason_codes"] == ["empty_or_short_input"]
    assert "output_duplicates_input" in quarantine[1]["cleaning"]["reason_codes"]


def test_repetition_ratio_is_deterministic_and_bounded():
    value = repeated_ngram_ratio("abcdefghijabcdefghij")
    assert 0.0 < value <= 1.0
    assert repeated_ngram_ratio("short") == 0.0


def test_cleaning_policy_rejects_unusable_thresholds():
    import pytest

    with pytest.raises(ValueError, match="near_duplicate_threshold"):
        clean_candidates([], CleaningPolicy(near_duplicate_threshold=0.2))
