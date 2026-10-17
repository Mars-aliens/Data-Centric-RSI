import unittest

from local_experiments.aladdin.qwen_training_protocol import validate_matrix


def _config(**changes):
    value = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "max_steps": 150,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_length": 1024,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "eval_batch_size": 16,
    }
    value.update(changes)
    return value


class QwenTrainingProtocolTests(unittest.TestCase):
    def test_equalizes_strategy_budgets(self):
        report = validate_matrix({"random": _config(), "active": _config()})
        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["checks"]["random"]["training_budget_equalized"])

    def test_rejects_different_steps(self):
        with self.assertRaises(ValueError):
            validate_matrix({"random": _config(), "active": _config(max_steps=100)})

    def test_allows_zero_lora_dropout(self):
        report = validate_matrix({"random": _config(lora_dropout=0.0)})
        self.assertEqual(report["status"], "ready")


if __name__ == "__main__":
    unittest.main()
