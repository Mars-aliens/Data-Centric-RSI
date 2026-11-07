import pytest

from experiments.feedback.agreement import compare_feedback


def _event(source, query_id, judgment, issue, actionable):
    return {
        "feedback_source": source,
        "query_id": query_id,
        "judgment": judgment,
        "issue_code": issue,
        "actionable": actionable,
    }


def test_feedback_comparison_reports_agreement_and_actionable_recall():
    model = [
        _event("model_simulated", "q1", "usable", "none", False),
        _event("model_simulated", "q2", "needs_revision", "incomplete", True),
        _event("model_simulated", "q3", "unusable", "irrelevant", True),
    ]
    human = [
        _event("live_human", "q1", "usable", "none", False),
        _event("live_human", "q2", "needs_revision", "incomplete", True),
        _event("live_human", "q3", "needs_revision", "incomplete", False),
    ]

    report = compare_feedback(model, human)

    assert report["paired_queries"] == 3
    assert report["judgment_exact_agreement"] == pytest.approx(2 / 3)
    assert report["issue_code_exact_agreement"] == pytest.approx(2 / 3)
    assert report["actionable_precision"] == pytest.approx(1 / 2)
    assert report["actionable_recall"] == pytest.approx(1.0)


def test_feedback_comparison_rejects_unpaired_queries_and_sources():
    with pytest.raises(ValueError, match="exactly the same"):
        compare_feedback([_event("model_simulated", "q1", "usable", "none", False)], [])
    with pytest.raises(ValueError, match="model_simulated"):
        compare_feedback([_event("live_human", "q1", "usable", "none", False)],
                         [_event("live_human", "q1", "usable", "none", False)])
