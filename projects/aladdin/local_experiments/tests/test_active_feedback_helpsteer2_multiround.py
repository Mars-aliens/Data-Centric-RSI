from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_experiments.aladdin.active_feedback_helpsteer2_multiround import run_multiround
from local_experiments.tests.test_active_feedback_helpsteer2 import _row


class MultiRoundHelpSteer2Tests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        path = root / "helpsteer2.jsonl"
        rows = []
        for group_index in range(48):
            split = "seed" if group_index < 12 else "held_in" if group_index < 36 else "held_out"
            rows.extend(_row(group_index, candidate, split) for candidate in (0, 1))
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_recomputes_selection_and_evaluates_each_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_multiround(
                self._fixture(Path(directory)),
                split_seeds=(17,),
                policy_seeds=(17,),
                rounds=2,
                round_feedback_budget_seconds=90,
                round_max_queries=2,
                seed_group_limit=8,
                query_group_limit=6,
                selection_group_limit=8,
                heldout_group_limit=6,
                selection_budget=3,
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["results"]), 3 * 5)
        self.assertTrue(report["configuration"]["selection_recomputed_after_each_round"])
        self.assertFalse(report["final_test_accessed"])
        for row in report["results"]:
            self.assertEqual(len(row["rounds"]), 2)
            self.assertEqual(len(row["rounds"][0]["heldout_pairwise"]), 4)
            self.assertLessEqual(row["feedback_cost_seconds"], 180 + 1e-9)
            pools = row["pool_ids"]
            self.assertTrue(set(pools["query"]).isdisjoint(pools["selection"]))
            self.assertTrue(set(pools["selection"]).isdisjoint(pools["heldout"]))


if __name__ == "__main__":
    unittest.main()
