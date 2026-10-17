import json
import tempfile
import unittest
from pathlib import Path

from local_experiments.aladdin.audit_dataset_protocol import audit


class DatasetAuditTests(unittest.TestCase):
    def test_detects_cross_split_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [
                {"source_group_id": "g1", "split": "seed", "target": "preferred"},
                {"source_group_id": "g1", "split": "held_out", "target": "rejected"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = audit(path)
        self.assertEqual(report["status"], "error")
        self.assertIn("g1", report["cross_split_groups"])
        self.assertEqual(report["final_test_rows_loaded"], 0)


if __name__ == "__main__":
    unittest.main()
