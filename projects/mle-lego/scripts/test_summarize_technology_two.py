from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_technology_two import ARMS, summarize


class SummarizeTechnologyTwoTests(unittest.TestCase):
    def _write_run(self, root: Path, arm: str, seed: int, *, manifest: str | None = None) -> None:
        run_dir = root / arm / f"seed-{seed}"
        run_dir.mkdir(parents=True)
        value = {
            "status": "complete",
            "seed": seed,
            "completed_batches": 4,
            "held_out_used": False,
            "final_test_used": False,
            "task_manifest_sha256": manifest or f"stream-{seed}",
            "model_config": {"base_model_id": "test", "base_model_revision": "rev"},
            "prequential_metrics": {"macro_f1": 0.4 + ARMS.index(arm) / 100},
            "training_data_policy": {"arm": arm},
            "batch_history": [
                {"training_data_policy": {"records_selected": 10, "records_dropped": 2}}
            ],
        }
        (run_dir / "summary.json").write_text(json.dumps(value), encoding="utf-8")

    def test_summarizes_paired_runs_and_checks_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (17, 23):
                for arm in ARMS:
                    self._write_run(root, arm, seed)
            result = summarize(root, [17, 23], expected_batches=4)
        self.assertEqual(result["arms"]["lifecycle_governed"]["runs"], 2)
        self.assertFalse(result["final_test_used"])

    def test_rejects_unpaired_task_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ARMS:
                self._write_run(root, arm, 17, manifest="different" if arm == ARMS[-1] else "same")
            with self.assertRaisesRegex(ValueError, "task manifest mismatch"):
                summarize(root, [17], expected_batches=4)


if __name__ == "__main__":
    unittest.main()
