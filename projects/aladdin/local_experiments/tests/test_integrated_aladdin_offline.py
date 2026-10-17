from __future__ import annotations

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from local_experiments.aladdin.integrated_pipeline.prepare_offline_replay import (
    prepare,
)


class IntegratedOfflineReplayTests(unittest.TestCase):
    def test_prepare_splits_complete_source_groups_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            output = root / "offline.jsonl"
            manifest = root / "offline-manifest.json"
            rows = [
                {
                    "record_id": "seed-1",
                    "task_id": "task",
                    "split": "seed",
                    "input": "seed",
                    "target": "preferred",
                    "source_group_id": "seed-group",
                },
                {
                    "record_id": "dev-a",
                    "task_id": "task",
                    "split": "development",
                    "input": "a",
                    "target": "preferred",
                    "source_group_id": "group-a",
                },
                {
                    "record_id": "dev-b",
                    "task_id": "task",
                    "split": "development",
                    "input": "b",
                    "target": "rejected",
                    "source_group_id": "group-a",
                },
                {
                    "record_id": "dev-c",
                    "task_id": "task",
                    "split": "development",
                    "input": "c",
                    "target": "tie",
                    "source_group_id": "group-b",
                },
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = prepare(source, output, manifest, seed=17, development_fraction=0.5)
            output_rows = [json.loads(line) for line in output.read_text().splitlines()]
            splits = defaultdict(set)
            for row in output_rows:
                splits[row["source_group_id"]].add(row["split"])
            self.assertEqual(set(splits["group-a"]).__len__(), 1)
            self.assertEqual(set(splits["group-b"]).__len__(), 1)
            self.assertEqual(result["cross_split_source_groups"], 0)
            self.assertFalse(result["final_test_accessed"])
            self.assertEqual(json.loads(manifest.read_text())["status"], "ready")


if __name__ == "__main__":
    unittest.main()
