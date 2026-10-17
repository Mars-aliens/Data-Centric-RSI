from __future__ import annotations

import unittest

from local_experiments.aladdin.hh_rlhf_llm_smoke import (
    LLMSmokeError,
    _build_prompt,
    _parse_choice,
)


class HHRLHFLlmSmokeTests(unittest.TestCase):
    def test_parse_structured_choice(self) -> None:
        self.assertEqual(_parse_choice('{"choice":"A"}'), "A")
        self.assertEqual(_parse_choice("```json\n{\"choice\": \"B\"}\n```"), "B")

    def test_reject_invalid_choice(self) -> None:
        with self.assertRaises(LLMSmokeError):
            _parse_choice('{"choice":"C"}')

    def test_prompt_does_not_expose_dataset_label(self) -> None:
        system, user = _build_prompt("Human: question", "answer A", "answer B")
        self.assertIn("Candidate A", user)
        self.assertIn("Candidate B", user)
        self.assertNotIn("chosen", user.lower())
        self.assertNotIn("rejected", user.lower())
        self.assertIn("JSON", system)


if __name__ == "__main__":
    unittest.main()
