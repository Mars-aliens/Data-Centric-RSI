from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "create_t1_smoke_candidate.py"
STRATEGY_FILE = "candidate_strategy.json"


class CreateT1SmokeCandidateTests(unittest.TestCase):
    def test_creates_a_manifested_candidate_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "code"
            source.mkdir()
            (source / "candidate_strategy.json").write_text(
                '{"schema":"lego-t1-data-strategy-v1","data_policy":'
                '{"min_input_chars":1,"deduplicate_inputs":true,"max_feedback_examples":null}}',
                encoding="utf-8",
            )
            source_strategy = (source / "candidate_strategy.json").read_text(encoding="utf-8")
            output = root / "candidate"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-dir",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )

            manifest = json.loads(
                (output / "candidate_proposal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["decision_model"]["provider"], "local_smoke_fixture")
            self.assertEqual(manifest["proposal_cost"], {"llm_calls": 0, "llm_tokens": 0})
            self.assertIn("default policy", manifest["rationale"])
            self.assertEqual((source / STRATEGY_FILE).read_text(encoding="utf-8"), source_strategy)
            self.assertTrue((output / STRATEGY_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
