from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_long_horizon_task.py"


def _write_split(path: Path, prefix: str, group_count: int) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for group_index in range(group_count):
            for response, score in (("high", 4), ("low", 0)):
                row = {
                    "prompt": f"{prefix} prompt {group_index}",
                    "response": f"{response} response",
                    "helpfulness": score,
                    "correctness": score,
                    "coherence": score,
                    "complexity": score,
                    "verbosity": score,
                }
                handle.write(json.dumps(row) + "\n")


class PrepareLongHorizonCliTests(unittest.TestCase):
    def test_cli_rejects_non_positive_max_batches_without_traceback(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--train-jsonl",
                "unused-train.jsonl",
                "--validation-jsonl",
                "unused-validation.jsonl",
                "--output-dir",
                "unused-output",
                "--max-batches",
                "0",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a positive integer", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_writes_agent_and_runner_preflight_artifacts_to_separate_trees(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            output = root / "bundle"
            _write_split(train, "train", 600)
            _write_split(validation, "validation", 25)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--train-jsonl",
                    str(train),
                    "--validation-jsonl",
                    str(validation),
                    "--output-dir",
                    str(output),
                    "--groups-per-batch",
                    "30",
                    "--feedback-delay",
                    "2",
                    "--max-batches",
                    "3",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )

            manifest = json.loads(
                (output / "agent-visible" / "task_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreater(manifest["development_batch_count"], 1)
            self.assertEqual(manifest["development_batch_count"], 3)
            self.assertTrue(manifest["stream_prefix_limited"])
            self.assertEqual(
                len(
                    (output / "runner-only" / "development-batches.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                3,
            )
            self.assertTrue((output / "agent-visible" / "seed.jsonl").is_file())
            self.assertTrue(
                (output / "runner-only" / "development-batches.jsonl").is_file()
            )
            self.assertTrue(
                (output / "runner-only" / "feedback-schedule.jsonl").is_file()
            )
            self.assertFalse(
                (output / "agent-visible" / "development-batches.jsonl").exists()
            )
            self.assertIn("agent_bundle_contains_gold_feedback: false", result.stdout)


if __name__ == "__main__":
    unittest.main()
