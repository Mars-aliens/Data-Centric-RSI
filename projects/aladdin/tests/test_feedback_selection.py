from experiments.feedback.selection import select_queries, split_development_groups


class FixedUncertainty:
    def uncertainty(self, record):
        return float(record["test_uncertainty"])


def _rows():
    return [
        {"record_id": "a", "source_group_id": "g1", "test_uncertainty": 0.1},
        {"record_id": "b", "source_group_id": "g2", "test_uncertainty": 0.95},
        {"record_id": "c", "source_group_id": "g2", "test_uncertainty": 0.4},
    ]


def test_uncertainty_selector_picks_highest_entropy():
    selected = select_queries(
        _rows(), method="uncertainty", count=1, seed=17, run_id="r", arm_id="a",
        round_index=0, uncertainty_model=FixedUncertainty(), estimated_review_seconds=30,
    )
    assert selected[0]["record_id"] == "b"


def test_feedback_value_uses_uncertainty_impact_and_time():
    selected = select_queries(
        _rows(), method="feedback-value", count=1, seed=17, run_id="r", arm_id="a",
        round_index=0, uncertainty_model=FixedUncertainty(), estimated_review_seconds=30,
    )
    assert selected[0]["record_id"] == "b"
    assert selected[0]["affected_candidate_count"] == 2
    assert selected[0]["selection_score"] > selected[0]["uncertainty"] / 30


def test_random_selection_is_reproducible():
    kwargs = dict(
        method="random", count=2, seed=23, run_id="r", arm_id="a", round_index=1,
        uncertainty_model=FixedUncertainty(), estimated_review_seconds=30,
    )
    left = select_queries(_rows(), **kwargs)
    right = select_queries(_rows(), **kwargs)
    assert [row["record_id"] for row in left] == [row["record_id"] for row in right]


def test_selection_reserves_quarantine_audit_quota():
    rows = [
        {"record_id": f"r{i}", "source_group_id": f"g{i}", "test_uncertainty": 0.5,
         "cleaning": {"status": "retained"}}
        for i in range(3)
    ] + [
        {"record_id": f"q{i}", "source_group_id": f"qg{i}", "test_uncertainty": 0.5,
         "cleaning": {"status": "quarantined"}}
        for i in range(2)
    ]
    selected = select_queries(
        rows, method="random", count=4, seed=17, run_id="r", arm_id="a",
        round_index=0, uncertainty_model=FixedUncertainty(), estimated_review_seconds=30,
        quarantine_fraction=0.25,
    )
    assert len(selected) == 4
    assert sum(row["candidate_was_quarantined"] for row in selected) == 1


def test_development_calibration_split_keeps_source_groups_intact():
    rows = [
        {"record_id": f"{group}-{i}", "source_group_id": group}
        for group in range(20)
        for i in range(2)
    ]
    calibration, evaluation = split_development_groups(
        rows, calibration_fraction=0.4, seed=17
    )
    calibration_groups = {row["source_group_id"] for row in calibration}
    evaluation_groups = {row["source_group_id"] for row in evaluation}
    assert calibration_groups.isdisjoint(evaluation_groups)
    assert len(calibration) + len(evaluation) == len(rows)
