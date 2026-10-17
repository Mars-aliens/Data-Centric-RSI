import unittest

from local_experiments.aladdin.analyze_helpsteer2_qwen_lora_matrix import analyze


class AnalyzeHelpSteer2QwenTests(unittest.TestCase):
    def test_reports_paired_strategy_differences(self):
        rows = []
        for split_seed in (17, 23):
            for strategy, value in (
                ("seed-only", 0.50),
                ("random", 0.52),
                ("disagreement-only", 0.53),
                ("impact-aware-random", 0.54),
            ):
                rows.append({
                    "split_seed": split_seed,
                    "policy_seed": 17,
                    "strategy": strategy,
                    "metrics": {"pairwise_accuracy": value},
                    "cost": {"training_seconds": 1.0},
                })
        result = analyze({
            "status": "completed",
            "protocol_version": "test",
            "configuration": {
                "strategies": [
                    "seed-only", "random", "disagreement-only", "impact-aware-random"
                ],
                "final_test_accessed": False,
            },
            "results": rows,
        })
        self.assertEqual(result["results"], 8)
        self.assertAlmostEqual(
            result["paired_primary_vs_baseline"]["random"]["mean_delta"], 0.02
        )
        self.assertFalse(result["final_test_accessed"])


if __name__ == "__main__":
    unittest.main()
