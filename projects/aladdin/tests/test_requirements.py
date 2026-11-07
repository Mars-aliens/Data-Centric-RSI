import pytest

from experiments.rsi.requirements import evaluate_requirement_compliance, validate_requirement_schedule


def test_requirement_schedule_validates_and_scores_objective_rules():
    requirements = validate_requirement_schedule([
        {"requirement_id": "short", "release_round": 1, "text": "Be concise.",
         "criterion": {"type": "max_chars", "field": "output", "value": 5}},
        {"requirement_id": "prefix", "release_round": 0, "text": "Start with OK.",
         "criterion": {"type": "starts_with", "field": "output", "value": "OK"}},
    ], rounds=2)
    assert [row["requirement_id"] for row in requirements] == ["prefix", "short"]
    report = evaluate_requirement_compliance([{"output": "OK yes"}, {"output": "No"}], requirements)
    assert report["per_requirement"]["prefix"]["rate"] == 0.5
    assert report["per_requirement"]["short"]["rate"] == 0.5


def test_requirement_schedule_rejects_unknown_or_unreachable_rules():
    with pytest.raises(ValueError, match="supported objective"):
        validate_requirement_schedule([{"requirement_id": "x", "release_round": 0,
            "text": "x", "criterion": {"type": "subjective"}}], rounds=1)
    with pytest.raises(ValueError, match="within configured rounds"):
        validate_requirement_schedule([{"requirement_id": "x", "release_round": 1,
            "text": "x", "criterion": {"type": "contains", "value": "x"}}], rounds=1)
