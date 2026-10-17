from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_experiments.aladdin.hh_rlhf_llm_cost_pilot import run_cost_pilot


def row(index: int) -> dict[str, str]:
    history = f"\n\nHuman: question {index}\n\nAssistant:"
    return {"chosen": history + f" good answer {index}", "rejected": history + f" bad answer {index}"}


class HHRLHFLlmCostPilotTests(unittest.TestCase):
    def test_cost_pilot_records_actual_cost_and_seed_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.jsonl"
            data_path.write_text(
                "\n".join(json.dumps(row(i)) for i in range(300)) + "\n",
                encoding="utf-8",
            )

            def fake_call(**_: object) -> tuple[str, dict[str, int | str]]:
                return "A", {
                    "content": '{"choice":"A"}',
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }

            with patch(
                "local_experiments.aladdin.hh_rlhf_llm_cost_pilot._call_llm",
                side_effect=fake_call,
            ):
                report = run_cost_pilot(
                    data_path,
                    output_path=root / "report.json",
                    endpoint="http://test/v1/chat/completions",
                    model="test-model",
                    model_revision="test-revision",
                    strategies=("random", "seed-only"),
                    seeds=(17,),
                    candidate_budget=2,
                    token_budget=10000,
                    seed_group_limit=8,
                    pool_group_limit=8,
                    llm_workers=2,
                )

        self.assertEqual(len(report["results"]), 2)
        random_result = report["results"][0]
        seed_result = report["results"][1]
        self.assertEqual(random_result["budget"]["llm_calls"], 2)
        self.assertEqual(random_result["budget"]["llm_tokens"], 24)
        self.assertEqual(random_result["budget"]["llm_workers"], 2)
        self.assertFalse(random_result["budget"]["token_budget_exceeded"])
        self.assertEqual(seed_result["budget"]["llm_calls"], 0)
        self.assertEqual(report["final_test"], "not loaded")

    def test_resume_skips_completed_strategy_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.jsonl"
            output_path = root / "report.json"
            data_path.write_text(
                "\n".join(json.dumps(row(i)) for i in range(300)) + "\n",
                encoding="utf-8",
            )

            def fake_call(**_: object) -> tuple[str, dict[str, int | str]]:
                return "A", {
                    "content": '{"choice":"A"}',
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }

            with patch(
                "local_experiments.aladdin.hh_rlhf_llm_cost_pilot._call_llm",
                side_effect=fake_call,
            ) as call:
                first = run_cost_pilot(
                    data_path,
                    output_path=output_path,
                    endpoint="http://test/v1/chat/completions",
                    model="test-model",
                    model_revision="test-revision",
                    strategies=("random", "seed-only"),
                    seeds=(17,),
                    candidate_budget=2,
                    token_budget=10000,
                    seed_group_limit=8,
                    pool_group_limit=8,
                    llm_workers=2,
                )
                first_calls = call.call_count
                resumed = run_cost_pilot(
                    data_path,
                    output_path=output_path,
                    endpoint="http://test/v1/chat/completions",
                    model="test-model",
                    model_revision="test-revision",
                    strategies=("random", "seed-only"),
                    seeds=(17,),
                    candidate_budget=2,
                    token_budget=10000,
                    seed_group_limit=8,
                    pool_group_limit=8,
                    llm_workers=2,
                    resume=True,
                )

        self.assertEqual(first_calls, call.call_count)
        self.assertEqual(len(first["results"]), 2)
        self.assertEqual(len(resumed["results"]), 2)


if __name__ == "__main__":
    unittest.main()
