import json
from pathlib import Path

from experiments.rsi.runner import _validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_every_example_rsi_config_validates():
    files = list((ROOT / "configs").glob("*.example.json"))
    checked = []
    for path in files:
        config = json.loads(path.read_text(encoding="utf-8"))
        if "comparison" not in config:
            continue
        validated = _validate_config(config)
        assert validated["arms"]
        assert validated["seeds"]
        checked.append(path.name)
    assert set(checked) == {
        "experiment-one-single-seed.example.json",
        "experiment-two-ablation.example.json",
        "experiment-two-factorial.example.json",
        "experiment-three-selection.example.json",
        "experiment-three-selection-fixed-pool.example.json",
        "feedback-rsi-pilot.example.json",
        "experiment-one-smoke.example.json",
        "experiment-two-factorial-smoke.example.json",
        "experiment-three-selection-smoke.example.json",
        "experiment-four-data-budget-smoke.example.json",
        "experiment-five-requirements-smoke.example.json",
    }


def test_experiment_five_requirements_smoke_config_has_release_schedule():
    config = json.loads((ROOT / "configs" / "experiment-five-requirements-smoke.example.json").read_text(encoding="utf-8"))
    validated = _validate_config(config)
    assert validated["comparison"] == "requirements"
    assert validated["arms"] == ["fixed_policy", "recursive_policy", "full_requirement_reference"]
    assert validated["requirement_schedule"][0]["release_round"] == 1


def test_experiment_four_data_budget_smoke_config_has_nested_limits():
    config = json.loads((ROOT / "configs" / "experiment-four-data-budget-smoke.example.json").read_text(encoding="utf-8"))
    assert config["seed_group_limits"] == [2, 4]
    assert config["comparison"] == "data_budget"


def test_experiment_two_ablation_config_has_four_update_arms():
    config = json.loads(
        (ROOT / "configs" / "experiment-two-ablation.example.json").read_text(
            encoding="utf-8"
        )
    )

    validated = _validate_config(config)

    assert validated["comparison"] == "ablation"
    assert validated["output"] == "runs/experiment-two-ablation"
    assert validated["arms"] == [
        "recursive_policy",
        "freeze_synthesis",
        "freeze_cleaning",
        "freeze_selection",
    ]
    assert validated["rounds"] == 3
    assert validated["feedback_mode"] == "model"


def test_experiment_three_selection_config_has_three_selection_arms():
    config = json.loads(
        (ROOT / "configs" / "experiment-three-selection.example.json").read_text(
            encoding="utf-8"
        )
    )

    validated = _validate_config(config)

    assert validated["comparison"] == "selection"
    assert validated["output"] == "runs/experiment-three-selection"
    assert validated["arms"] == [
        "query_random",
        "query_uncertainty",
        "query_feedback_value",
    ]
    assert validated["rounds"] == 3
    assert validated["feedback_mode"] == "model"


def test_factorial_config_contains_all_eight_update_combinations():
    config = json.loads(
        (ROOT / "configs" / "experiment-two-factorial.example.json").read_text(encoding="utf-8")
    )

    validated = _validate_config(config)

    assert validated["comparison"] == "factorial"
    assert len(validated["arms"]) == 8
    assert validated["arms"][0] == "fixed_policy"
    assert validated["arms"][-1] == "recursive_policy"
