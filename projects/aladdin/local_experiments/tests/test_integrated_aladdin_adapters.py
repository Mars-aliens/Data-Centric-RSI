from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core_system.mle_core.models import TaskRecord
from local_experiments.aladdin.integrated_pipeline.adapters import (
    LLMCandidateGenerator,
    OfflineReplayCandidateGenerator,
    OfflineReplayFeedbackReviewer,
    OpenAICompatibleJsonClient,
    load_calibrator,
    load_quality_audits,
    load_requirement_state,
    load_task_records,
)
from local_experiments.aladdin.integrated_pipeline.system import RequirementHypothesis
from local_experiments.aladdin.integrated_pipeline.system import FeedbackQuery


def seed_record() -> TaskRecord:
    return TaskRecord(
        record_id="seed-1",
        task_id="intent",
        split="seed",
        input="I cannot sign in",
        target="account",
        source_group_id="source-1",
    )


def held_in_record() -> TaskRecord:
    return TaskRecord(
        record_id="held-in-1",
        task_id="intent",
        split="held_in",
        input="I need to reset my password",
        target="preferred",
        source_group_id="source-held-in-1",
        metadata={"dataset": "HelpSteer2", "scores": {"correctness": 4}},
    )


class IntegratedAdapterTests(unittest.TestCase):
    def test_offline_replay_hides_label_until_feedback_is_requested(self):
        source = held_in_record()
        generator = OfflineReplayCandidateGenerator([source])
        generated, budget = generator.generate(
            seed_records=[seed_record()],
            requirements=[RequirementHypothesis("account", "Account access")],
            confirmed_edits=[],
            strategy="coverage",
            count=1,
            round_index=0,
            cost_budget=1536,
        )
        self.assertEqual(budget.llm_calls, 0)
        self.assertEqual(len(generated), 1)
        candidate = generated[0]
        self.assertIsNone(candidate.target)
        self.assertNotIn("scores", candidate.metadata)
        feedback = OfflineReplayFeedbackReviewer([source]).review(
            candidate,
            FeedbackQuery(candidate.record_id, 1.0, 1.0, 1.0, 30.0, False),
        )
        self.assertTrue(feedback.accept_for_training)
        self.assertEqual(feedback.target, "preferred")

    def test_openai_client_parses_fenced_json_and_tracks_usage(self):
        def transport(payload):
            self.assertEqual(payload["temperature"], 0.0)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            return {
                "choices": [{"message": {"content": "```json\n{\"score\": 0.75}\n```"}}],
                "usage": {"total_tokens": 19},
            }

        client = OpenAICompatibleJsonClient(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="test-model",
            transport=transport,
        )
        result = client.complete_json(
            system_prompt="score", user_payload={"text": "x"}, max_tokens=32
        )
        self.assertEqual(result, {"score": 0.75})
        self.assertEqual(client.budget.llm_calls, 1)
        self.assertEqual(client.budget.llm_tokens, 19)

    def test_openai_client_separates_prompt_and_completion_usage(self):
        client = OpenAICompatibleJsonClient(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="test-model",
            transport=lambda _: {
                "choices": [{"message": {"content": '{"score": 1}'}}],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 7,
                    "total_tokens": 907,
                },
            },
        )
        client.complete_json(
            system_prompt="score", user_payload={"text": "x"}, max_tokens=32
        )
        self.assertEqual(client.budget.llm_tokens, 907)
        self.assertEqual(client.prompt_tokens, 900)
        self.assertEqual(client.completion_tokens, 7)

    def test_llm_generator_validates_lineage_and_reports_incremental_budget(self):
        def transport(_payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "input": "The member portal rejects my login",
                                            "target": "account",
                                            "parent_record_id": "seed-1",
                                            "quality_slice": "account-access",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 41,
                    "total_tokens": 941,
                },
            }

        client = OpenAICompatibleJsonClient(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="test-model",
            transport=transport,
        )
        generated, budget = LLMCandidateGenerator(client).generate(
            seed_records=[seed_record()],
            requirements=[RequirementHypothesis("account", "Account access")],
            confirmed_edits=[],
            strategy="coverage",
            count=1,
            round_index=0,
            cost_budget=512,
        )
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].parent_record_ids, ("seed-1",))
        self.assertEqual(generated[0].source_group_id, "source-1")
        self.assertEqual(generated[0].metadata["quality_slice"], "account-access")
        self.assertEqual(budget.llm_calls, 1)
        self.assertEqual(budget.llm_tokens, 41)

    def test_llm_generator_bounds_seed_context_deterministically(self):
        seen = {}

        def transport(payload):
            seen.update(json.loads(payload["messages"][1]["content"]))
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"candidates": []})
                        }
                    }
                ]
            }

        records = [
            TaskRecord(
                record_id=f"seed-{index}",
                task_id="intent",
                split="seed",
                input=f"example-{index}-" + ("x" * 100),
                target="account",
            )
            for index in range(20)
        ]
        client = OpenAICompatibleJsonClient(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="test-model",
            transport=transport,
        )
        generator = LLMCandidateGenerator(
            client, seed_example_limit=3, seed_example_chars=12
        )
        generator.generate(
            seed_records=records,
            requirements=[RequirementHypothesis("account", "Account access")],
            confirmed_edits=[],
            strategy="coverage",
            count=1,
            round_index=0,
            cost_budget=128,
        )
        self.assertEqual(seen["seed_context_policy"]["source_seed_count"], 20)
        self.assertEqual(seen["seed_context_policy"]["provided_seed_count"], 3)
        self.assertEqual(len(seen["seed_examples"]), 3)
        self.assertTrue(all(len(item["input"]) <= 12 for item in seen["seed_examples"]))

    def test_llm_generator_repairs_missing_parent_without_fabricating_lineage(self):
        calls = []

        def transport(_payload):
            calls.append(True)
            if len(calls) == 1:
                content = {"candidates": [{
                    "input": "A repaired candidate",
                    "target": "account",
                    "quality_slice": "access",
                }]}
            else:
                content = {"candidates": [{
                    "input": "A repaired candidate",
                    "target": "account",
                    "parent_record_id": "seed-1",
                    "quality_slice": "access",
                }]}
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

        client = OpenAICompatibleJsonClient(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="test-model",
            transport=transport,
        )
        generated, _ = LLMCandidateGenerator(client).generate(
            seed_records=[seed_record()],
            requirements=[RequirementHypothesis("account", "Account access")],
            confirmed_edits=[],
            strategy="coverage",
            count=1,
            round_index=0,
            cost_budget=512,
        )
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].parent_record_ids, ("seed-1",))
        self.assertEqual(len(calls), 2)

    def test_llm_generator_drops_unknown_parent_without_fabricating_lineage(self):
        client = OpenAICompatibleJsonClient(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="test-model",
            transport=lambda _: {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "input": "new",
                                            "target": "account",
                                            "parent_record_id": "not-a-seed",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )
        generator = LLMCandidateGenerator(client)
        generated, _ = generator.generate(
            seed_records=[seed_record()],
            requirements=[RequirementHypothesis("account", "Account access")],
            confirmed_edits=[],
            strategy="coverage",
            count=1,
            round_index=0,
            cost_budget=512,
        )
        self.assertEqual(generated, [])
        self.assertEqual(generator.last_lineage_drops, 1)

    def test_config_loaders_preserve_split_and_calibration_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "record_id": "seed-1",
                        "task_id": "intent",
                        "split": "seed",
                        "input": "login failed",
                        "target": "account",
                        "metadata": {"quality_slice": "access"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "hypotheses": [
                            {"id": "account", "text": "Prefer access coverage"},
                            {"id": "billing", "text": "Prefer billing coverage"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "minimum_audits": 1,
                        "minimum_agreement_lower_bound": 0.1,
                        "audits": [
                            {
                                "slice": "access",
                                "proposed_decision": "accept",
                                "human_accept": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit_file = root / "quality-audits.json"
            audit_file.write_text(
                json.dumps({"audits": {"seed-1": True}}), encoding="utf-8"
            )
            records = load_task_records(data)
            state = load_requirement_state(requirements)
            calibrator = load_calibrator(calibration)
            loaded_audits = load_quality_audits(audit_file)
        self.assertEqual(records[0].metadata["quality_slice"], "access")
        self.assertEqual(len(state.hypotheses), 2)
        self.assertEqual(calibrator.evidence_for("access", "accept").audits, 1)
        self.assertEqual(loaded_audits, {"seed-1": True})

    def test_jsonl_loader_preserves_unicode_paragraph_separators_in_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "record_id": "seed-1",
                        "task_id": "intent",
                        "split": "seed",
                        "input": "first\u2028second",
                        "target": "account",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_task_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].input, "first\u2028second")


if __name__ == "__main__":
    unittest.main()
