from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from . import long_horizon_qwen_command as command


class LongHorizonQwenCommandTests(unittest.TestCase):
    def test_train_command_passes_parent_checkpoint_to_incremental_trainer(self) -> None:
        parent = {"backend": "qwen-lora-preference-v1", "task_model_version": "parent-v1"}
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "train",
            "records": [{"record_id": "r1", "input": "x", "target": "preferred"}],
            "config": {
                "max_steps": 3,
                "base_model_path": "local-model",
                "base_model_revision": "model-rev-1",
            },
            "initial_checkpoint": parent,
            "artifact_store_root": "run/batch-1",
            "seed": 17,
        }
        expected_checkpoint = {"task_model_version": "child-v1"}
        with patch.object(
            command,
            "train_checkpoint",
            return_value=(expected_checkpoint, {"training_steps": 3}),
        ) as trainer:
            result = command.train_once(request)

        self.assertEqual(result["checkpoint"], expected_checkpoint)
        self.assertEqual(trainer.call_args.kwargs["initial_checkpoint"], parent)
        self.assertEqual(
            trainer.call_args.kwargs["output_dir"],
            Path("run/batch-1/long_horizon_qwen_adapter"),
        )

    def test_predict_command_rejects_labels_before_calling_model(self) -> None:
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "predict",
            "checkpoint": {"task_model_version": "model-v1"},
            "records": [{"record_id": "r1", "input": "x", "target": "preferred"}],
        }
        with patch.object(command, "predict") as predictor:
            with self.assertRaisesRegex(ValueError, "must not contain targets"):
                command.predict_once(request)
        predictor.assert_not_called()

    def test_predict_command_preserves_record_to_prediction_alignment(self) -> None:
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "predict",
            "checkpoint": {"task_model_version": "model-v1"},
            "records": [
                {"record_id": "r1", "input": "one"},
                {"record_id": "r2", "input": "two"},
            ],
        }
        with patch.object(command, "predict", return_value=["A", "C"]) as predictor:
            result = command.predict_once(request)
        self.assertEqual(
            result["predictions"],
            [
                {"record_id": "r1", "prediction": "preferred"},
                {"record_id": "r2", "prediction": "tie"},
            ],
        )
        self.assertNotIn("target", predictor.call_args.args[1][0])

    def test_predict_rejects_unknown_internal_label(self) -> None:
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "predict",
            "checkpoint": {"task_model_version": "model-v1"},
            "records": [{"record_id": "r1", "input": "one"}],
        }
        with patch.object(command, "predict", return_value=["unknown"]):
            with self.assertRaisesRegex(ValueError, "unsupported label token"):
                command.predict_once(request)

    def test_train_rejects_any_unlabeled_or_malformed_row_without_filtering(self) -> None:
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "train",
            "records": [
                {"record_id": "r1", "input": "x", "target": "preferred"},
                {"record_id": "r2", "input": "y"},
            ],
            "config": {
                "base_model_path": "local-model",
                "base_model_revision": "model-rev-1",
            },
            "artifact_store_root": "run",
        }
        with patch.object(command, "train_checkpoint") as trainer:
            with self.assertRaisesRegex(ValueError, "valid labeled record target"):
                command.train_once(request)
        trainer.assert_not_called()

    def test_invalid_protocol_and_unlabeled_training_data_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "long-horizon-qwen-v1"):
            command.train_once({"protocol": "lego-trainer-v1"})
        request = {
            "protocol": "long-horizon-qwen-v1",
            "operation": "train",
            "records": [{"record_id": "r1", "input": "x"}],
            "config": {
                "base_model_path": "local-model",
                "base_model_revision": "model-rev-1",
            },
            "artifact_store_root": "run",
        }
        with self.assertRaisesRegex(ValueError, "labeled record"):
            command.train_once(request)


if __name__ == "__main__":
    unittest.main()
