import json
from types import SimpleNamespace

import pytest

from experiments.evaluation.model_pairwise import collect_model_pairwise
from experiments.evaluation.blind_pairwise import analyze_blind_pairwise_feedback


class Judge:
    def __init__(self, values):
        self.values = iter(values)
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, complete_usage_requests=0)

    def generate(self, prompt, **kwargs):
        assert 'system_a' not in prompt
        self.usage.prompt_tokens += 20
        self.usage.completion_tokens += 10
        self.usage.complete_usage_requests += 1
        return json.dumps({"preference": next(self.values), "reason": "comparison"})


def test_swapped_model_judging_normalizes_and_resumes(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({"query_id": "q", "source_group_id": "u", "input": "question",
                                 "candidate_a": "answer one", "candidate_b": "answer two"}) + "\n")
    output = tmp_path / "ratings.jsonl"
    events = collect_model_pairwise(queue, output_path=output, generator=Judge(["A", "B"]))
    assert events[0]["preference"] == "A"
    assert events[0]["order_consistent"] is True
    assert collect_model_pairwise(queue, output_path=output, generator=Judge([])) == events
    key = [{"query_id": "q", "source_group_id": "u", "system_a": "fixed", "system_b": "recursive"}]
    report = analyze_blind_pairwise_feedback(events, key, expected_source="final_test_blind_model")
    assert report["systems"]["fixed"]["wins"] == 1
    with pytest.raises(ValueError, match="non-final-test"):
        analyze_blind_pairwise_feedback(events, key)


def test_order_bias_is_not_reported_as_a_win(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({"query_id": "q", "source_group_id": "u", "input": "question",
                                 "candidate_a": "answer one", "candidate_b": "answer two"}) + "\n")
    output = tmp_path / "ratings.jsonl"
    result = collect_model_pairwise(queue, output_path=output, generator=Judge(["A", "A"]))
    assert result[0]["preference"] == "cannot_judge"
    assert result[0]["order_consistent"] is False
    with pytest.raises(ValueError, match="configuration changed"):
        collect_model_pairwise(queue, output_path=output, generator=Judge([]), max_tokens=256)
