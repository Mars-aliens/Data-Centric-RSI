from __future__ import annotations

import unittest

from technologies.technology_three.trace_evaluation import (
    evaluate_future_window,
    evaluate_trace,
    trace_from_t1_summary,
)


def trace(gains: list[float], *, failed_at: int | None = None) -> dict:
    comparisons = []
    events = []
    for index, gain in enumerate(gains):
        stable_version = "stable-v1"
        candidate_version = "candidate-v2"
        comparisons.append(
            {
                "source_batch_index": index,
                "stable_version_id": stable_version,
                "candidate_version_id": candidate_version,
                "stable_score": 0.5,
                "candidate_score": 0.5 + gain,
                "gain": gain,
                "available_from_batch": index + 3,
                "candidate_status": "failed" if index == failed_at else "eligible_for_observation",
            }
        )
        events.extend(
            [
                {
                    "feedback_id": f"s-{index}",
                    "source_batch_index": index,
                    "available_from_batch": index + 3,
                    "target_version_id": stable_version,
                    "score": 0.5,
                },
                {
                    "feedback_id": f"c-{index}",
                    "source_batch_index": index,
                    "available_from_batch": index + 3,
                    "target_version_id": candidate_version,
                    "score": 0.5 + gain,
                },
            ]
        )
    return {
        "schema": "mle-lego-t3-feedback-trace-v1",
        "source_run_id": "t1-test",
        "held_out_used": False,
        "final_test_used": False,
        "comparisons": comparisons,
        "feedback_events": events,
    }


class T3TraceEvaluationTests(unittest.TestCase):
    def test_future_window_uses_batches_after_decision_window(self) -> None:
        result = evaluate_future_window(
            trace([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]),
            decision_window_batches=3,
            outcome_window_batches=3,
        )
        self.assertEqual(result["decision_source_batches"], [0, 1, 2])
        self.assertEqual(result["outcome_source_batches"], [3, 4, 5])
        self.assertFalse(result["outcome_success"])
        self.assertTrue(result["recommendations"]["source_attributed_window"]["false_accept"])

    def test_future_window_marks_short_trace_unscorable(self) -> None:
        result = evaluate_future_window(
            trace([0.1, 0.1, 0.1]),
            decision_window_batches=3,
            outcome_window_batches=3,
        )
        self.assertFalse(result["outcome_complete"])
        self.assertIsNone(
            result["recommendations"]["source_attributed_window"]["false_accept"]
        )
    def test_trace_builder_keeps_failed_candidate_missing_not_zero(self) -> None:
        summary = {
            "status": "complete",
            "run_id": "t1-test",
            "held_out_used": False,
            "final_test_used": False,
            "batch_history": [
                {
                    "batch_index": 0,
                    "model_version_id": "stable-v1",
                    "released_metrics": [
                        {
                            "source_batch_index": 0,
                            "model_version_id": "stable-v1",
                            "macro_f1": 0.5,
                            "available_from_batch": 3,
                        },
                        {
                            "source_batch_index": 0,
                            "model_version_id": "candidate-v2",
                            "macro_f1": 0.6,
                            "available_from_batch": 3,
                        },
                    ],
                },
                {"batch_index": 1, "model_version_id": "stable-v1", "released_metrics": []},
            ],
            "candidate_history": [
                {"batch_index": 0, "candidate_version_id": "candidate-v2", "status": "eligible_for_observation"},
                {"batch_index": 1, "candidate_version_id": "candidate-v2", "status": "failed"},
            ],
            "terminal_metrics": [
                {
                    "source_batch_index": 1,
                    "model_version_id": "stable-v1",
                    "macro_f1": 0.5,
                    "available_from_batch": 4,
                }
            ],
        }
        result = trace_from_t1_summary(summary)
        self.assertEqual(result["comparisons"][0]["candidate_score"], 0.6)
        self.assertEqual(result["comparisons"][1]["candidate_status"], "failed")
        self.assertIsNone(result["comparisons"][1]["candidate_score"])

    def test_window_rejects_candidate_with_cross_batch_regression(self) -> None:
        result = evaluate_trace(
            trace([0.12, 0.10, -0.08]),
            window_batches=3,
            min_mean_gain=0.0,
            max_regression=0.05,
        )
        policy = result["policies"]["source_attributed_window"]
        self.assertEqual(policy["action"], "keep_stable")
        self.assertEqual(policy["reason"], "paired_source_window_did_not_pass")

    def test_arrival_baseline_counts_feedback_for_non_current_version(self) -> None:
        result = evaluate_trace(trace([0.1, 0.1, 0.1]))
        self.assertGreater(
            result["policies"]["arrival_as_current"]["misattributed_feedback_count"], 0
        )

    def test_failed_candidate_is_rejected_not_scored_as_zero(self) -> None:
        result = evaluate_trace(trace([0.1, 0.1, 0.1], failed_at=1))
        policy = result["policies"]["source_attributed_window"]
        self.assertEqual(policy["action"], "reject_failed_candidate")
        self.assertIsNone(policy.get("mean_gain"))

    def test_held_out_or_final_test_trace_is_rejected(self) -> None:
        invalid = trace([0.1, 0.1, 0.1])
        invalid["final_test_used"] = True
        with self.assertRaisesRegex(ValueError, "must not use held-out"):
            evaluate_trace(invalid)


if __name__ == "__main__":
    unittest.main()
