import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_experiments.aladdin.helpsteer2_qwen_lora_matrix import run_matrix
from local_experiments.tests.test_active_feedback_helpsteer2 import _row


class HelpSteer2QwenMatrixTests(unittest.TestCase):
    def test_frozen_matrix_reuses_selected_groups_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            rows = []
            for group_index in range(12):
                split = "seed" if group_index < 4 else "held_in" if group_index < 8 else "held_out"
                rows.extend(_row(group_index, candidate, split) for candidate in (0, 1))
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            ids = [f"prompt-{i:03d}" for i in range(12)]
            frozen_rows = []
            for strategy in ("seed-only", "random", "disagreement-only", "impact-aware-random"):
                selected = [] if strategy == "seed-only" else ["prompt-006"]
                frozen_rows.append({
                    "strategy": strategy,
                    "split_seed": 17,
                    "policy_seed": 17,
                    "target_profile_simulator": "helpfulness-only",
                    "selected_data_ids": selected,
                    "rounds": [{
                        "posterior": {
                            "helpfulness-only": 1.0,
                            "correctness-only": 0.0,
                            "coherence-only": 0.0,
                        }
                    }],
                    "pool_ids": {
                        "seed": ids[:4],
                        "heldout": ids[8:],
                        "query": ids[4:5],
                        "selection": ids[5:8],
                    },
                })
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({
                "results": frozen_rows,
                "configuration": {},
                "final_test_accessed": False,
            }), encoding="utf-8")
            output = root / "out.json"
            fake_checkpoint = {
                "base_model_path": "base",
                "adapter_path": str(root / "adapter"),
                "label_ids": {"A": 1, "B": 2, "C": 3},
            }
            with patch(
                "local_experiments.aladdin.helpsteer2_qwen_lora_matrix.train_checkpoint",
                return_value=(fake_checkpoint, {"training_steps": 150}),
            ), patch(
                "local_experiments.aladdin.helpsteer2_qwen_lora_matrix.predict_scores",
                return_value=[{"A": 2.0, "B": 0.0, "C": 0.0}, {"A": 0.0, "B": 2.0, "C": 0.0}] * 4,
            ):
                report = run_matrix(
                    data, matrix, output, base_model="base",
                    split_seeds=(17,), policy_seeds=(17,),
                )
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["results"]), 4)
            self.assertFalse(report["configuration"]["final_test_accessed"])
            self.assertTrue(all(
                0.0 <= row["metrics"]["pairwise_accuracy"] <= 1.0
                and row["metrics"]["pairwise_groups"] > 0
                for row in report["results"]
            ))


if __name__ == "__main__":
    unittest.main()
