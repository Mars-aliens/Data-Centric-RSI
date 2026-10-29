from __future__ import annotations

import unittest
from unittest.mock import patch

from technologies.technology_one import long_horizon_candidate_qwen_command as candidate_command


class CandidateQwenCommandTests(unittest.TestCase):
    def test_applies_strategy_to_released_data_before_candidate_training(self) -> None:
        request = {
            "protocol": "lego-t1-candidate-step-v1",
            "candidate_id": "candidate-1",
            "parent_version_id": "stable-v1",
            "config": {
                "base_model_path": "/models/task",
                "candidate_strategy": {
                    "schema": "lego-t1-data-strategy-v1",
                    "data_policy": {
                        "min_input_chars": 2,
                        "deduplicate_inputs": True,
                        "max_feedback_examples": 1,
                    },
                },
            },
            "checkpoint": {"adapter_path": "/parent-checkpoint"},
            "records": [
                {"record_id": "current-1", "input": "Current prompt"},
            ],
            "seed_examples": [
                {"record_id": "seed-1", "input": "Shared", "target": "preferred"},
            ],
            "released_feedback": [
                {"record_id": "feedback-duplicate", "input": " shared ", "target": "rejected"},
                {"record_id": "feedback-2", "input": "New example", "target": "preferred"},
                {"record_id": "feedback-3", "input": "Other example", "target": "tie"},
            ],
        }

        with (
            patch.object(candidate_command, "predict", return_value=["A"]),
            patch.object(
                candidate_command,
                "train_checkpoint",
                return_value=({"adapter_path": "/candidate-output/checkpoint"}, {"training_steps": 1}),
            ) as train,
        ):
            response = candidate_command.run_step(request)

        selected = train.call_args.args[0]
        self.assertEqual([item["record_id"] for item in selected], ["seed-1", "feedback-3"])
        self.assertEqual(response["data_processing"]["training_records_selected"], 2)
        self.assertEqual(response["data_processing"]["training_records_dropped"], 2)
        self.assertEqual(response["data_processing"]["feedback_records_dropped_by_cap"], 1)
        self.assertTrue(response["data_processing"]["strategy_sha256"])


if __name__ == "__main__":
    unittest.main()
