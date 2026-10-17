from __future__ import annotations

import unittest

from local_experiments.aladdin.analyze_hh_rlhf_llm_training_pilot import analyze_training_pilot


def result(strategy: str, seed: int, llm: float, oracle: float, feedback: float | None) -> dict:
    return {
        "strategy": strategy,
        "seed": seed,
        "metrics": {
            "feedback_accuracy": feedback,
            "llm_feedback_task_model": {"pairwise_accuracy": llm},
            "oracle_feedback_task_model": {"pairwise_accuracy": oracle},
        },
        "budget": {
            "llm_calls": 2 if strategy != "seed-only" else 0,
            "llm_tokens": 20 if strategy != "seed-only" else 0,
            "feedback_requests": 2 if strategy != "seed-only" else 0,
            "training_steps": 2,
            "evaluator_calls": 2,
        },
    }


class AnalyzeTrainingPilotTests(unittest.TestCase):
    def test_reports_paired_delta_against_seed_only(self) -> None:
        report = {
            "experiment": "pilot",
            "dataset": "Anthropic/hh-rlhf",
            "dataset_revision": "revision",
            "results": [
                result("seed-only", 17, 0.50, 0.50, None),
                result("aladdin", 17, 0.52, 0.51, 0.75),
                result("seed-only", 18, 0.51, 0.51, None),
                result("aladdin", 18, 0.50, 0.52, 0.50),
            ],
        }
        analysis = analyze_training_pilot(report)
        self.assertEqual(analysis["reference_strategy"], "seed-only")
        delta = analysis["paired_deltas_vs_reference"]["aladdin"]["llm_pairwise_delta"]
        self.assertAlmostEqual(delta["mean"], 0.005)
        self.assertEqual(delta["positive_count"], 1)
        self.assertTrue(any("crosses zero" in warning for warning in analysis["warnings"]))

    def test_warns_when_llm_token_cost_is_unbalanced(self) -> None:
        report = {
            "results": [
                result("seed-only", 17, 0.50, 0.50, None),
                result("random", 17, 0.50, 0.50, 0.50),
                result("aladdin", 17, 0.50, 0.50, 0.50),
            ],
        }
        report["results"][1]["budget"]["llm_tokens"] = 100
        report["results"][2]["budget"]["llm_tokens"] = 200
        analysis = analyze_training_pilot(report)
        self.assertTrue(any("token consumption differs" in warning for warning in analysis["warnings"]))


if __name__ == "__main__":
    unittest.main()
