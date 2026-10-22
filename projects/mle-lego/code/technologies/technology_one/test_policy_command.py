from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from . import policy_command


class _Response:
    def __init__(self, value: dict):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class DecisionModelEndpointTests(unittest.TestCase):
    def test_uses_configured_openai_compatible_model_and_secret_environment_name(self) -> None:
        response = _Response(
            {
                "choices": [
                    {"message": {"content": '{"proposal":{"action_type":"finish"}}'}}
                ],
                "usage": {"total_tokens": 42},
            }
        )
        request = {
            "protocol": "lego-policy-v1",
            "model": {
                "id": "kimi-k3-from-console",
                "revision": "provider-pinned-release-2026-10",
                "provider": "bailian_openai_compatible",
                "api_url_env": "TEST_POLICY_URL",
                "api_key_env": "TEST_POLICY_SECRET",
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 256,
            },
            "prompt": {"system": "JSON only", "user": "Return a proposal"},
        }
        with patch.dict(
            "os.environ",
            {
                "TEST_POLICY_URL": "https://bailian.example/compatible-mode/v1/chat/completions",
                "TEST_POLICY_SECRET": "test-secret",
            },
            clear=True,
        ), patch(
            "technologies.technology_one.policy_command.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = policy_command.call_endpoint(request)

        sent = urlopen.call_args.args[0]
        body = json.loads(sent.data)
        self.assertEqual(sent.full_url, "https://bailian.example/compatible-mode/v1/chat/completions")
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(body["model"], "kimi-k3-from-console")
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(result["cost"], {"llm_calls": 1, "llm_tokens": 42})

    def test_can_disable_provider_json_mode_when_endpoint_lacks_json_schema_support(self) -> None:
        response = _Response(
            {
                "choices": [
                    {"message": {"content": '{"proposal":{"action_type":"finish"}}'}}
                ],
                "usage": {},
            }
        )
        with patch(
            "technologies.technology_one.policy_command.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            policy_command.call_endpoint(
                {
                    "protocol": "lego-policy-v1",
                    "model": {
                        "id": "qwen-model-from-provider",
                        "revision": "immutable-release",
                        "api_url": "http://model.example/v1/chat/completions",
                        "json_mode": False,
                    },
                    "prompt": {"system": "JSON only", "user": "Return a proposal"},
                }
            )
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("response_format", body)

    def test_rejects_non_openai_compatible_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "OpenAI-compatible"):
            policy_command._model_settings(
                {
                    "model": {"id": "x", "provider": "native_custom_api"},
                    "prompt": {"system": "", "user": ""},
                }
            )

    def test_candidate_protocol_accepts_api_model_id_and_schema_correction(self) -> None:
        responses = [
            _Response({"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 4}}),
            _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "proposal": {
                                            "rationale": "A bounded change.",
                                            "edits": [
                                                {
                                                    "path": "runner.py",
                                                    "operation": "write",
                                                    "content": "pass\n",
                                                }
                                            ],
                                        }
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"total_tokens": 8},
                }
            ),
        ]
        with patch(
            "technologies.technology_one.policy_command.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen:
            result = policy_command.call_endpoint(
                {
                    "protocol": "lego-data-system-candidate-v1",
                    "model": {
                        "model_id": "kimi-k3-model-id",
                        "api_url": "https://bailian.example/v1/chat/completions",
                    },
                    "prompt": {"system": "JSON only", "user": "Propose a source change"},
                }
            )

        first_body = json.loads(urlopen.call_args_list[0].args[0].data)
        second_body = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(first_body["model"], "kimi-k3-model-id")
        self.assertIn("Schema correction", second_body["messages"][-1]["content"])
        self.assertEqual(result["cost"], {"llm_calls": 2, "llm_tokens": 12})


if __name__ == "__main__":
    unittest.main()
