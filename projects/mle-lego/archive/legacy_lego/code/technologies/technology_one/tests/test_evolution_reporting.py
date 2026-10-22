from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .evolution_reporting import audit_evolution, summarize_evolution


def _summary(seed: int, *, final_test_used: bool = False) -> dict:
    batches = []
    for index, score in enumerate((0.20, 0.25, 0.30)):
        batches.append(
            {
                "active_result": {
                    "held_in": {"macro_f1": score},
                    "held_out": {"macro_f1": score + 0.01},
                }
            }
        )
    transitions = [
        {
            "status": "accepted",
            "held_out_used_for_acceptance": False,
            "rollback_performed": False,
        },
        {
            "status": "rejected",
            "held_out_used_for_acceptance": False,
            "rollback_performed": True,
        },
    ]
    return {
        "experiment": "technology-one-server-harness-evolution",
        "seed": seed,
        "decision_llm_used": True,
        "final_test_used": final_test_used,
        "held_out_used_for_acceptance": False,
        "batch_count": 3,
        "batches": batches,
        "transitions": transitions,
        "accepted_version_count": 1,
        "rejected_candidate_count": 1,
        "rollback_count": 1,
        "budget": {"llm_calls": 2, "training_steps": 500, "evaluator_calls": 10},
        "dataset": {"source_revision": "test"},
        "task_model": {"revision": "test"},
        "evaluator": {"revision": "test"},
        "timing": {"wall_clock_seconds": 10.0},
    }


class EvolutionReportingTests(unittest.TestCase):
    def test_audit_and_summary_accept_complete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (17, 23):
                path = root / f"run-seed-{seed}"
                path.mkdir()
                (path / "summary.json").write_text(json.dumps(_summary(seed)), encoding="utf-8")
            audit = audit_evolution(root, (17, 23))
            self.assertEqual(audit["status"], "PASS")
            report = summarize_evolution(root, (17, 23))
            self.assertEqual(report["completed_rows"], 2)
            self.assertAlmostEqual(report["aggregate"]["delta_held_out"]["mean"], 0.10)

    def test_audit_rejects_final_test_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "run-seed-17"
            path.mkdir()
            (path / "summary.json").write_text(
                json.dumps(_summary(17, final_test_used=True)), encoding="utf-8"
            )
            report = audit_evolution(root, (17,))
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("final_test_used" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
