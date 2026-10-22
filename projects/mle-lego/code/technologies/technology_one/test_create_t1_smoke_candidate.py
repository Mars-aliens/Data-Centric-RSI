from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "create_t1_smoke_candidate.py"
SOURCE_FILE = "technologies/technology_one/long_horizon_candidate_qwen_command.py"


class CreateT1SmokeCandidateTests(unittest.TestCase):
    def test_creates_a_manifested_candidate_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "code"
            (source / "technologies/technology_one").mkdir(parents=True)
            original = "def run():\n    return 1\n"
            (source / SOURCE_FILE).write_text(original, encoding="utf-8")
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
            self.assertIn("without changing behavior", manifest["rationale"])
            self.assertEqual((source / SOURCE_FILE).read_text(encoding="utf-8"), original)
            self.assertIn("Deterministic T1 infrastructure smoke fixture", (output / SOURCE_FILE).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
