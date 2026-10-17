from __future__ import annotations

import unittest

from local_experiments.aladdin.analyze_hh_rlhf_report import analyze_report


def result(strategy: str, seed: int, pairwise: float, accuracy: float, ids: list[str]) -> dict:
    return {
        "strategy": strategy,
        "seed": seed,
        "metrics": {"pairwise_accuracy": pairwise, "accuracy": accuracy},
        "diagnostics": {"selected_group_ids": ids},
        "budget": {
            "feedback_requests": 20,
            "feedback_records_released": 40,
            "training_steps": 1,
            "evaluator_calls": 1,
            "llm_calls": 0,
        },
    }


class AnalyzeHHRLHFReportTests(unittest.TestCase):
    def test_reports_paired_deltas_overlap_and_budget(self) -> None:
        report = {
            "experiment": "test",
            "dataset": "Anthropic/hh-rlhf",
            "dataset_revision": "revision",
            "adapter_version": "v3",
            "results": [
                result("aladdin", 17, 0.52, 0.51, ["a", "b"]),
                result("random", 17, 0.50, 0.50, ["b", "c"]),
                result("aladdin", 18, 0.51, 0.50, ["a", "b"]),
                result("random", 18, 0.52, 0.51, ["x", "y"]),
            ],
        }
        analysis = analyze_report(report)
        self.assertEqual(analysis["reference_strategy"], "aladdin")
        self.assertAlmostEqual(analysis["deltas_vs_reference"]["random"]["pairwise_delta_by_seed"][0], 0.02)
        self.assertAlmostEqual(analysis["deltas_vs_reference"]["random"]["pairwise_delta_by_seed"][1], -0.01)
        self.assertEqual(analysis["deltas_vs_reference"]["random"]["pairwise_delta_positive_seeds"], 1)
        self.assertAlmostEqual(analysis["selection_jaccard_vs_reference"]["random"]["17"], 1 / 3)
        self.assertTrue(analysis["budget_signatures_equal"])
        self.assertTrue(any("not present for every seed" in warning for warning in analysis["warnings"]))


if __name__ == "__main__":
    unittest.main()
