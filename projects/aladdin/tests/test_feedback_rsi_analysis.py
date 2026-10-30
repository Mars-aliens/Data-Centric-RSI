import pytest

from experiments.rsi.analysis import analyze_result


def test_analysis_reports_paired_development_deltas_by_seed():
    result = {
        "status": "completed",
        "protocol": "aladdin-feedback-driven-rsi-v1",
        "run_id": "r1",
        "comparison": "overall",
        "data": {"final_test_accessed": False},
        "summary": {},
        "arms": {
            "fixed_policy": {
                "rounds": [
                    {"seed": 17, "training": {"pairwise_accuracy": 0.4}},
                    {"seed": 17, "training": {"pairwise_accuracy": 0.6}},
                    {"seed": 23, "training": {"pairwise_accuracy": 0.5}},
                ]
            },
            "recursive_policy": {
                "rounds": [
                    {"seed": 17, "training": {"pairwise_accuracy": 0.5}},
                    {"seed": 17, "training": {"pairwise_accuracy": 0.7}},
                    {"seed": 23, "training": {"pairwise_accuracy": 0.4}},
                ]
            },
        },
    }
    report = analyze_result(result)
    delta = report["paired_development_comparisons"]["overall_recursive_minus_fixed"]

    assert delta["paired_seeds"] == 2
    assert delta["per_seed_delta"]["17"] == pytest.approx(0.1)
    assert delta["per_seed_delta"]["23"] == pytest.approx(-0.1)
    assert delta["wins_ties_losses"] == [1, 0, 1]
    assert report["final_test_accessed"] is False


def test_analysis_rejects_incomplete_or_final_test_accessed_runs():
    with pytest.raises(ValueError, match="completed"):
        analyze_result({"status": "running"})
    with pytest.raises(ValueError, match="final_test_accessed"):
        analyze_result({"status": "completed", "data": {"final_test_accessed": True}})
