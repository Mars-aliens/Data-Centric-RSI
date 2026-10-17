from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_experiments.aladdin.hh_rlhf_llm_matrix import run_llm_feedback_matrix


def row(index: int) -> dict[str, str]:
    history = f"\n\nHuman: question {index}\n\nAssistant:"
    return {"chosen": history + f" good answer {index}", "rejected": history + f" bad answer {index}"}


class HHRLHFLlmMatrixTests(unittest.TestCase):
    def test_small_matrix_uses_equal_llm_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text("\n".join(json.dumps(row(i)) for i in range(300)) + "\n", encoding="utf-8")

            def fake_call(**_: object) -> tuple[str, dict[str, int | str]]:
                return "A", {"content": '{"choice":"A"}', "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

            with patch("local_experiments.aladdin.hh_rlhf_llm_matrix._call_llm", side_effect=fake_call):
                report = run_llm_feedback_matrix(
                    path,
                    output_path=Path(directory) / "report.json",
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
        self.assertTrue(all(item["budget"]["llm_calls"] == 2 for item in report["results"]))
        self.assertEqual(report["final_test"], "not loaded")


if __name__ == "__main__":
    unittest.main()
