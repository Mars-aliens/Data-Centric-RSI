import json
from types import SimpleNamespace

import pytest

from experiments.feedback.annotation import read_feedback
from experiments.feedback.model_feedback import MODEL_REVISION, collect_model_feedback, parse_feedback


class Judge:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.usage = SimpleNamespace(requests=0, complete_usage_requests=0, prompt_tokens=0, completion_tokens=0)

    def generate(self, prompt, *, max_tokens, temperature):
        self.usage.requests += 1
        self.usage.complete_usage_requests += 1
        self.usage.prompt_tokens += 30
        self.usage.completion_tokens += 10
        return next(self.outputs)


def _query(index):
    return {"query_id": f"q{index}", "record_id": f"r{index}", "source_group_id": "g",
            "arm_id": "fixed_policy", "round": 0, "selection_method": "feedback-value",
            "candidate": {"input": "Question", "output": "Answer"}}


def _response():
    return json.dumps({"judgment": "needs_revision", "issue_code": "incomplete",
                       "requirement_clarity": "clear", "corrected_output": "Better answer",
                       "new_requirement": "Address the full question."})


def test_model_feedback_saves_source_usage_and_resumes(tmp_path):
    path = tmp_path / "feedback.jsonl"
    judge = Judge([_response(), _response()])
    first = collect_model_feedback([_query(1)], output_path=path, generator=judge,
                                   model_revision=MODEL_REVISION, max_tokens=128, temperature=0)
    assert first[0]["feedback_source"] == "model_simulated"
    assert first[0]["prompt_tokens"] == 30
    with pytest.raises(ValueError, match="unexpected feedback source"):
        read_feedback(path)
    second = collect_model_feedback([_query(1), _query(2)], output_path=path, generator=judge,
                                    model_revision=MODEL_REVISION, max_tokens=128, temperature=0)
    assert len(second) == 2
    assert judge.usage.requests == 2
    assert len(path.with_name("feedback.model-attempts.jsonl").read_text().splitlines()) == 2


def test_invalid_model_output_is_logged_and_does_not_become_feedback(tmp_path):
    path = tmp_path / "feedback.jsonl"
    judge = Judge(["not json"])
    with pytest.raises(ValueError, match="JSON"):
        collect_model_feedback([_query(1)], output_path=path, generator=judge,
                               model_revision=MODEL_REVISION, max_tokens=128, temperature=0)
    assert not path.exists()
    attempt = json.loads(path.with_name("feedback.model-attempts.jsonl").read_text())
    assert attempt["status"] == "error"
    assert attempt["prompt_tokens"] == 30


def test_model_feedback_schema_rejects_unexpected_fields():
    response = json.loads(_response())
    response["extra"] = "unexpected"
    with pytest.raises(ValueError, match="exactly"):
        parse_feedback(json.dumps(response))


def test_failed_review_resumes_without_repeating_completed_queries(tmp_path):
    path = tmp_path / "feedback.jsonl"
    with pytest.raises(ValueError):
        collect_model_feedback([_query(1), _query(2)], output_path=path,
                               generator=Judge([_response(), "invalid"]),
                               model_revision=MODEL_REVISION, max_tokens=128, temperature=0)
    judge = Judge([_response()])
    events = collect_model_feedback([_query(1), _query(2)], output_path=path, generator=judge,
                                    model_revision=MODEL_REVISION, max_tokens=128, temperature=0)
    assert len(events) == 2
    assert judge.usage.requests == 1
    attempts = [json.loads(line) for line in path.with_name("feedback.model-attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == ["success", "error", "success"]
    assert sum(row["requests"] for row in attempts) == 3
