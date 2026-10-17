from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_experiments.aladdin.active_feedback_helpsteer2 import (
    PROFILES,
    run_experiment,
)


def _row(group_index: int, candidate_index: int, split: str) -> dict[str, object]:
    group_id = f"prompt-{group_index:03d}"
    score_fields = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
    scores = {
        field: ((group_index * (field_index + 2) + candidate_index * (field_index + 1)) % 4) + 1
        for field_index, field in enumerate(score_fields)
    }
    return {
        "record_id": f"{group_id}-{candidate_index}",
        "task_id": "helpsteer2-preference",
        "split": split,
        "input": f"Prompt: question {group_index}\nResponse: answer {candidate_index} feature {group_index * candidate_index}",
        "target": "preferred" if candidate_index == 0 else "rejected",
        "source_group_id": group_id,
        "metadata": {
            "dataset": "nvidia/HelpSteer2",
            "pair_index": candidate_index,
            "pair_size": 2,
            "batch_id": f"batch-{group_index % 4}",
            "score_mean": sum(scores.values()) / len(scores),
            "scores": scores,
            "feedback": {"decision": "preferred" if candidate_index == 0 else "rejected", "scores": scores},
        },
    }


class ActiveFeedbackHelpSteer2Tests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        path = root / "helpsteer2.jsonl"
        rows = []
        for group_index in range(48):
            split = "seed" if group_index < 12 else "held_in" if group_index < 36 else "held_out"
            rows.extend(_row(group_index, candidate, split) for candidate in (0, 1))
        for group_index in range(48, 50):
            rows.extend(_row(group_index, candidate, "final_test") for candidate in (0, 1))
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_end_to_end_factorial_replay_is_budgeted_and_split_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_experiment(
                self._fixture(Path(directory)),
                seeds=(17,),
                target_profiles=tuple(PROFILES),
                seed_group_limit=8,
                query_group_limit=6,
                selection_group_limit=8,
                heldout_group_limit=6,
                feedback_time_budget_seconds=90,
                max_queries=4,
                selection_budget=3,
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["protocol_version"], "active-feedback-helpsteer2-local-v2")
        self.assertFalse(report["configuration"]["final_test_loaded"])
        self.assertFalse(report["final_test_accessed"])
        self.assertTrue(report["label_leakage_audit"]["pool_overlap_counts_all_zero"])
        self.assertEqual(len(report["results"]), len(PROFILES) * 5)
        self.assertEqual(
            set(report["paired_comparisons_primary_vs_baseline"]),
            {"random", "disagreement-only", "seed-only"},
        )
        for result in report["results"]:
            self.assertLessEqual(result["feedback_cost_seconds"], 90)
            self.assertLessEqual(result["feedback_requests"], 4)
            self.assertTrue(all(value == 0 for value in result["pool_overlap_counts"].values()))
            ids = result["pool_ids"]
            self.assertTrue(set(ids["query"]).isdisjoint(ids["selection"]))
            self.assertTrue(set(ids["selection"]).isdisjoint(ids["heldout"]))
            self.assertTrue(set(result["selected_data_ids"]).issubset(ids["selection"]))
            self.assertTrue(all(query["group_id"] in ids["query"] for query in result["query_trace"]))
            self.assertEqual(result["selected_data_count"], 0 if result["strategy"] == "seed-only" else 3)
            self.assertGreater(result["heldout_pairwise"]["pairwise_groups"], 0)
            self.assertEqual(result["feedback_records_released"], result["feedback_requests"] * 2)
            for query in result["query_trace"]:
                if query["impact_value"] is not None:
                    self.assertAlmostEqual(
                        query["impact_value"],
                        query["disagreement"] * query["expected_selection_change"],
                    )

    def test_split_and_policy_seeds_are_paired_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_experiment(
                self._fixture(Path(directory)),
                split_seeds=(17, 23),
                policy_seeds=(31, 32),
                target_profiles=("helpfulness-only",),
                seed_group_limit=8,
                query_group_limit=6,
                selection_group_limit=8,
                heldout_group_limit=6,
                feedback_time_budget_seconds=90,
                max_queries=3,
                selection_budget=3,
            )

        self.assertEqual(len(report["results"]), 2 * 2 * 5)
        for result in report["results"]:
            self.assertIn(result["split_seed"], (17, 23))
            self.assertIn(result["policy_seed"], (31, 32))
        comparison = report["paired_comparisons_primary_vs_baseline"]["random"]
        self.assertEqual(comparison["n"], 2)
        self.assertEqual(comparison["unit"], "split_seed (profile and policy-seed deltas averaged within split)")


if __name__ == "__main__":
    unittest.main()
