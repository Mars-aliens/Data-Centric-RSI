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


def test_analysis_reports_full_factorial_effects():
    names = {
        (0, 0, 0): "fixed_policy", (1, 0, 0): "synthesis_only",
        (0, 1, 0): "cleaning_only", (0, 0, 1): "selection_only",
        (1, 1, 0): "synthesis_cleaning", (1, 0, 1): "synthesis_selection",
        (0, 1, 1): "cleaning_selection", (1, 1, 1): "recursive_policy",
    }
    arms = {
        name: {"rounds": [{"seed": 17, "training": {"pairwise_accuracy": sum(bits)}}]}
        for bits, name in names.items()
    }
    result = {
        "status": "completed", "data": {"final_test_accessed": False},
        "arms": arms, "summary": {}, "comparison": "factorial",
    }
    effects = analyze_result(result)["factorial_update_effects"]["mean_effects"]
    assert effects["main_synthesis"] == pytest.approx(1.0)
    assert effects["main_cleaning"] == pytest.approx(1.0)
    assert effects["main_selection"] == pytest.approx(1.0)


def test_factorial_analysis_uses_last_round_not_round_mean():
    names = {
        (0, 0, 0): "fixed_policy", (1, 0, 0): "synthesis_only",
        (0, 1, 0): "cleaning_only", (0, 0, 1): "selection_only",
        (1, 1, 0): "synthesis_cleaning", (1, 0, 1): "synthesis_selection",
        (0, 1, 1): "cleaning_selection", (1, 1, 1): "recursive_policy",
    }
    arms = {
        name: {"rounds": [
            {"seed": 17, "round": 0, "training": {"pairwise_accuracy": 0.0}},
            {"seed": 17, "round": 2, "training": {"pairwise_accuracy": float(sum(bits))}},
        ]}
        for bits, name in names.items()
    }
    report = analyze_result({
        "status": "completed", "data": {"final_test_accessed": False},
        "arms": arms, "summary": {}, "comparison": "factorial",
    })
    assert report["factorial_update_effects"]["mean_effects"]["main_synthesis"] == pytest.approx(1.0)
    assert report["factorial_update_effects"]["outcome"] == "last completed round per seed"


def test_factorial_interactions_match_known_response_surface():
    from experiments.rsi.runner import ARMS, COMPARISONS

    arms = {}
    for name in COMPARISONS["factorial"]:
        arm = ARMS[name]
        s, c, q = (int(arm.update_synthesis), int(arm.update_cleaning), int(arm.update_selection))
        score = 0.1 + 0.02*s + 0.03*c + 0.04*q + 0.05*s*c + 0.06*s*q + 0.07*c*q + 0.08*s*c*q
        arms[name] = {"rounds": [{"seed": 17, "round": 0, "training": {"pairwise_accuracy": score}}]}
    report = analyze_result({"status": "completed", "data": {"final_test_accessed": False}, "arms": arms})
    effects = report["factorial_update_effects"]["mean_effects"]
    assert effects["main_synthesis"] == pytest.approx(0.095)
    assert effects["main_cleaning"] == pytest.approx(0.11)
    assert effects["main_selection"] == pytest.approx(0.125)
    assert effects["interaction_synthesis_x_cleaning"] == pytest.approx(0.09)
    assert effects["interaction_synthesis_x_selection"] == pytest.approx(0.10)
    assert effects["interaction_cleaning_x_selection"] == pytest.approx(0.11)
    assert effects["interaction_synthesis_x_cleaning_x_selection"] == pytest.approx(0.08)
