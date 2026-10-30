from experiments.feedback.annotation import (
    collect_interactive_feedback,
    collect_one_feedback,
    read_feedback,
)


def test_interactive_feedback_records_correction_and_elapsed_time():
    answers = iter(["2", "6", "2", "Use the shorter answer.", "Keep the response concise."])
    ticks = iter([10.0, 14.25])
    query = {
        "query_id": "q1",
        "record_id": "c1",
        "arm_id": "recursive_policy",
        "round": 0,
        "selection_method": "feedback-value",
        "candidate": {"input": "Question", "output": "Long repetitive response."},
    }

    event = collect_one_feedback(
        query,
        annotator_id="reviewer-01",
        input_fn=lambda _prompt: next(answers),
        clock=lambda: next(ticks),
    )

    assert event["feedback_source"] == "live_human"
    assert event["judgment"] == "needs_revision"
    assert event["issue_code"] == "repetitive"
    assert event["corrected_output"] == "Use the shorter answer."
    assert event["new_requirement"] == "Keep the response concise."
    assert event["duration_seconds"] == 4.25


def test_feedback_reader_rejects_replay_or_duplicate_query_ids(tmp_path):
    import json
    import pytest

    path = tmp_path / "feedback.jsonl"
    row = {
        "query_id": "q1", "feedback_source": "development_replay", "judgment": "usable",
        "duration_seconds": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-live"):
        read_feedback(path)


def test_interactive_queue_resumes_and_reserves_estimated_review_time(tmp_path):
    queue = [
        {
            "query_id": f"q{i}", "record_id": f"c{i}", "arm_id": "recursive_policy",
            "estimated_review_seconds": 6,
            "candidate": {"input": f"Question {i}", "output": f"Answer {i}"},
        }
        for i in range(2)
    ]
    answers = iter(["1", "0", "", ""])
    ticks = iter([0.0, 8.0])
    path = tmp_path / "feedback.jsonl"
    events = collect_interactive_feedback(
        queue,
        annotator_id="reviewer-01",
        budget_seconds_per_arm=10,
        output_path=path,
        input_fn=lambda _prompt: next(answers),
        clock=lambda: next(ticks),
    )

    assert [event["query_id"] for event in events] == ["q0"]
    assert events[0]["duration_seconds"] == 8.0
    resumed = collect_interactive_feedback(
        queue,
        annotator_id="reviewer-01",
        budget_seconds_per_arm=10,
        output_path=path,
        input_fn=lambda _prompt: (_ for _ in ()).throw(AssertionError("completed query was repeated")),
        clock=lambda: (_ for _ in ()).throw(AssertionError("completed query was repeated")),
    )
    assert [event["query_id"] for event in resumed] == ["q0"]


def test_feedback_reader_rejects_non_finite_durations(tmp_path):
    import json
    import pytest

    path = tmp_path / "feedback.jsonl"
    path.write_text(
        json.dumps({
            "query_id": "q1", "feedback_source": "live_human", "judgment": "usable",
            "duration_seconds": float("nan"),
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duration"):
        read_feedback(path)
