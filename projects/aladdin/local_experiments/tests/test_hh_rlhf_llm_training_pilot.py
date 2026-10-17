from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_experiments.aladdin.hh_rlhf_llm_training_pilot import (
    _ordered_candidates,
    run_training_pilot,
)


def row(index: int) -> dict[str, str]:
    history = f"\n\nHuman: question {index}\n\nAssistant:"
    return {"chosen": history + f" good answer {index}", "rejected": history + f" bad answer {index}"}


class HHRLHFLlmTrainingPilotTests(unittest.TestCase):
    def test_candidate_order_is_independent_of_selection_position(self) -> None:
        candidates = [("chosen", "good"), ("rejected", "bad")]
        first = _ordered_candidates(17, "group-a", candidates)
        second = _ordered_candidates(17, "group-a", candidates)
        self.assertEqual(first, second)
        self.assertIn(first[0][0], {"chosen", "rejected"})

    def test_pilot_trains_llm_and_oracle_controls_with_equal_budget(self) -> None:
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
                "local_experiments.aladdin.hh_rlhf_llm_training_pilot._call_llm",
                side_effect=fake_call,
            ):
                report = run_training_pilot(
                    data_path,
                    output_path=root / "report.json",
                    endpoint="http://test/v1/chat/completions",
                    model="test-model",
                    model_revision="test-revision",
                    strategies=("random", "aladdin"),
                    seeds=(17,),
                    feedback_budget=2,
                    seed_group_limit=8,
                    pool_group_limit=8,
                )

        self.assertEqual(report["configuration"]["total_llm_calls"], 4)
        self.assertEqual(len(report["results"]), 2)
        for result in report["results"]:
            self.assertEqual(result["budget"]["llm_calls"], 2)
            self.assertIn("llm_feedback_task_model", result["metrics"])
            self.assertIn("oracle_feedback_task_model", result["metrics"])
        self.assertEqual(report["final_test"], "not loaded")

    def test_seed_only_control_does_not_spend_llm_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.jsonl"
            data_path.write_text(
                "\n".join(json.dumps(row(i)) for i in range(300)) + "\n",
                encoding="utf-8",
            )
            report = run_training_pilot(
                data_path,
                output_path=root / "report.json",
                endpoint="http://test/v1/chat/completions",
                model="test-model",
                model_revision="test-revision",
                strategies=("seed-only",),
                seeds=(17,),
                feedback_budget=2,
                seed_group_limit=8,
                pool_group_limit=8,
            )

        result = report["results"][0]
        self.assertEqual(report["configuration"]["total_llm_calls"], 0)
        self.assertEqual(result["budget"]["llm_calls"], 0)
        self.assertIsNone(result["metrics"]["feedback_accuracy"])


if __name__ == "__main__":
    unittest.main()
