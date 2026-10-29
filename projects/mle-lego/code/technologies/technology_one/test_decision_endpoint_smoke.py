from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import smoke_decision_endpoint  # noqa: E402


class DecisionEndpointSmokeTests(unittest.TestCase):
    def test_reports_valid_model_and_cost_without_echoing_credentials(self) -> None:
        profile = {
            "decision_model": {
                "provider": "bailian_openai_compatible",
                "model_id": "kimi-k3-api-id",
                "api_key_env": "LEGO_POLICY_API_KEY",
            }
        }
        response = {
            "proposal": {
                "rationale": "Connectivity smoke.",
                "edits": [
                    {
                        "path": "candidate_strategy.json",
                        "operation": "write",
                        "content": (
                            '{"schema":"lego-t1-data-strategy-v1","data_policy":'
                            '{"min_input_chars":1,"deduplicate_inputs":true,'
                            '"max_feedback_examples":null}}'
                        ),
                    }
                ],
            },
            "cost": {"llm_calls": 1, "llm_tokens": 20},
        }
        with patch.object(smoke_decision_endpoint, "call_endpoint", return_value=response):
            result = smoke_decision_endpoint.smoke_decision_endpoint(profile)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model_id"], "kimi-k3-api-id")
        self.assertTrue(result["schema_valid"])
        self.assertNotIn("api_key", result)
        self.assertEqual(result["cost"]["llm_tokens"], 20)

    def test_rejects_missing_api_model_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "API model ID"):
            smoke_decision_endpoint.smoke_decision_endpoint(
                {"decision_model": {"provider": "bailian_openai_compatible"}}
            )

    def test_rejects_valid_strategy_that_does_not_follow_smoke_instruction(self) -> None:
        response = {
            "proposal": {
                "rationale": "Different edit.",
                "edits": [
                    {
                        "path": "candidate_strategy.json",
                        "operation": "write",
                        "content": (
                            '{"schema":"lego-t1-data-strategy-v1","data_policy":'
                            '{"min_input_chars":5,"deduplicate_inputs":true,'
                            '"max_feedback_examples":null}}'
                        ),
                    }
                ],
            },
            "cost": {"llm_calls": 1, "llm_tokens": 20},
        }
        with patch.object(smoke_decision_endpoint, "call_endpoint", return_value=response):
            with self.assertRaisesRegex(ValueError, "requested smoke policy"):
                smoke_decision_endpoint.smoke_decision_endpoint(
                    {"decision_model": {"model_id": "test-model"}}
                )


if __name__ == "__main__":
    unittest.main()
