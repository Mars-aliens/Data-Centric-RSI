from experiments.rsi.runner import _select_seed_groups
from scripts.run_data_budget_sweep import run_data_budget_sweep
import pytest


def test_seed_group_budgets_are_nested_for_a_fixed_seed():
    rows = [{"record_id": f"r{i}", "source_group_id": f"g{i}"} for i in range(8)]
    low = {row["source_group_id"] for row in _select_seed_groups(rows, 2, 17)}
    high = {row["source_group_id"] for row in _select_seed_groups(rows, 5, 17)}
    assert low < high


def test_budget_sweep_runs_each_level_and_reports_paired_deltas(tmp_path):
    calls = []

    def fake_run(config):
        calls.append(config)
        limit = config["seed_group_limit"]
        return {
            "status": "completed",
            "data": {"seed_groups": limit},
            "summary": {},
            "arms": {
                "fixed_policy": {"rounds": [{"seed": 17, "training": {"pairwise_accuracy": 0.4}}]},
                "recursive_policy": {"rounds": [{"seed": 17, "training": {"pairwise_accuracy": 0.5}}]},
            },
        }

    report = run_data_budget_sweep({
        "comparison": "data_budget", "seed_group_limits": [2, 4], "output": str(tmp_path / "sweep")
    }, run_callable=fake_run)

    assert [call["seed_group_limit"] for call in calls] == [2, 4]
    assert report["nested_budgets"] is True
    assert [row["per_seed_recursive_minus_fixed"]["17"] for row in report["paired_development_deltas"]] == pytest.approx([0.1, 0.1])
    assert (tmp_path / "sweep" / "sweep-summary.json").is_file()
