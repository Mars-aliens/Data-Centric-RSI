from experiments.rsi.policies import initial_policy_state, update_policy_state


def test_updates_are_versioned_and_only_change_enabled_components():
    state = initial_policy_state()
    events = [
        {"query_id": "q1", "new_requirement": "Keep answers concise.", "actionable": True},
        {"query_id": "q2", "issue_code": "repetitive", "actionable": False},
        {"query_id": "q3", "issue_code": "repetitive", "actionable": True},
    ]
    updated, event = update_policy_state(
        state,
        events,
        update_synthesis=True,
        update_cleaning=True,
        update_selection=False,
    )

    assert updated["synthesis"]["guidance"] == ["Keep answers concise."]
    assert updated["synthesis"]["version"] == 1
    assert updated["cleaning"]["repeated_ngram_threshold"] < state["cleaning"]["repeated_ngram_threshold"]
    assert updated["selection"] == state["selection"]
    assert event["status"] == "updated"
    assert event["updates_enabled"]["selection"] is False


def test_empty_feedback_is_recorded_as_noop():
    updated, event = update_policy_state(initial_policy_state(), [])
    assert event["status"] == "no_op"
    assert updated == initial_policy_state()
