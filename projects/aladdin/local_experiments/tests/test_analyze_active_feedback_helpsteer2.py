from __future__ import annotations

import unittest

from local_experiments.aladdin.analyze_active_feedback_helpsteer2 import analyze_report


def _row(strategy: str, selected: list[str], queries: list[str]) -> dict[str, object]:
    return {
        "strategy": strategy,
        "split_seed": 17,
        "policy_seed": 17,
        "target_profile_simulator": "helpfulness-only",
        "selected_data_ids": selected,
        "query_trace": [
            {
                "group_id": group_id,
                "disagreement": 0.5,
                "expected_selection_change": 1.0 if group_id == "q1" else 0.0,
                "impact_value": 0.5 if group_id == "q1" else 0.0,
                "selection_score": 0.25,
            }
            for group_id in queries
        ],
    }


class AnalyzeActiveFeedbackHelpSteer2Tests(unittest.TestCase):
    def test_separates_query_changes_from_downstream_changes(self) -> None:
        report = {
            "status": "completed",
            "final_test_accessed": False,
            "results": [
                _row("impact-aware", ["d1"], ["q1"]),
                _row("disagreement-only", ["d1"], ["q2"]),
            ],
        }

        analysis = analyze_report(report)
        comparison = analysis["paired_set_diagnostics"]["impact-aware_vs_disagreement-only"]
        self.assertEqual(comparison["different_query_set_conditions"], 1)
        self.assertEqual(comparison["different_selected_data_set_conditions"], 0)
        self.assertTrue(analysis["interpretation"]["impact_changes_query_order"])
        self.assertFalse(analysis["interpretation"]["impact_changes_downstream_selection"])

    def test_reports_nonzero_expected_selection_change(self) -> None:
        report = {
            "status": "completed",
            "final_test_accessed": False,
            "results": [_row("impact-aware", ["d1"], ["q1", "q2"])],
        }

        summary = analyze_report(report)["strategy_trace_summary"]["impact-aware"]
        self.assertEqual(summary["scored_queries"], 2)
        self.assertEqual(summary["nonzero_expected_change_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
