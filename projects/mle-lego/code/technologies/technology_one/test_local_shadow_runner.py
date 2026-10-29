from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .candidate_sandbox import LocalShadowCandidateRunner, SandboxLimits, SandboxMount, hash_directory


MODULE = "technologies.technology_one.long_horizon_candidate_qwen_command"


class LocalShadowRunnerTests(unittest.TestCase):
    def _inputs(self, root: Path):
        candidate = root / "candidate"
        parent = root / "parent"
        model = root / "model"
        for path in (candidate, parent, model):
            path.mkdir()
        (candidate / "candidate_strategy.json").write_text("{}", encoding="utf-8")
        (parent / "adapter.bin").write_bytes(b"stable")
        (model / "config.json").write_text("{}", encoding="utf-8")
        mounts = (
            SandboxMount(candidate, "/candidate"),
            SandboxMount(parent, "/parent-checkpoint"),
            SandboxMount(model, "/models/task"),
        )
        payload = {
            "protocol": "lego-t1-candidate-step-v1",
            "candidate_id": "candidate-1",
            "parent_version_id": "stable-1",
            "batch_index": 0,
            "checkpoint": {
                "base_model_path": "/models/task",
                "adapter_path": "/parent-checkpoint",
                "task_model_version": "stable-1",
                "adapter_hash": hash_directory(parent),
            },
            "config": {},
            "records": [],
        }
        return candidate, parent, model, mounts, payload

    @staticmethod
    def _fake_process(*, mutate_parent: bool = False):
        class Process:
            returncode = 0

            def __init__(self, args, *, stdin, stdout, stderr, **_kwargs):
                self.stdout_file = stdout
                self.stderr_file = stderr

            def communicate(self, *, input, timeout):
                request = json.loads(input)
                parent_copy = Path(request["checkpoint"]["adapter_path"])
                if mutate_parent:
                    (parent_copy / "adapter.bin").write_bytes(b"changed")
                output = Path(request["config"]["candidate_output_path"])
                output.mkdir(parents=True)
                (output / "adapter.bin").write_bytes(b"candidate")
                response = {
                    "checkpoint": {
                        "base_model_path": request["config"]["base_model_path"],
                        "adapter_path": str(output),
                        "base_model_revision": "revision-1",
                        "backend": "qwen-lora-preference-v1",
                        "task_model_version": "candidate-1",
                        "adapter_hash": hash_directory(output),
                    },
                    "predictions": [],
                }
                self.stdout_file.write(json.dumps(response).encode())
                self.stderr_file.write(b"")
                return None, None

            def wait(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

        return Process

    def test_runs_fixed_adapter_with_private_parent_copy_and_separate_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, parent, _, mounts, payload = self._inputs(root)
            before = hash_directory(parent)
            runner = LocalShadowCandidateRunner(limits=SandboxLimits(tmpfs_mib=16))
            with patch(
                "technologies.technology_one.candidate_sandbox.subprocess.Popen",
                self._fake_process(),
            ):
                result = runner.execute(
                    command=("python", "-m", MODULE), payload=payload, mounts=mounts,
                    output_dir=root / "attempts" / "attempt-1",
                )

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.backend, "local_shadow")
            self.assertEqual(result.response["checkpoint"]["adapter_path"], "/candidate-output/checkpoint")
            self.assertTrue((root / "attempts/attempt-1/parent-checkpoint/adapter.bin").is_file())
            self.assertEqual(hash_directory(parent), before)
            self.assertFalse(runner.isolation_evidence["container_isolation"])
            self.assertFalse(runner.isolation_evidence["arbitrary_candidate_code_isolation"])

    def test_detects_candidate_mutation_of_private_parent_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, parent, _, mounts, payload = self._inputs(root)
            before = hash_directory(parent)
            runner = LocalShadowCandidateRunner()
            with patch(
                "technologies.technology_one.candidate_sandbox.subprocess.Popen",
                self._fake_process(mutate_parent=True),
            ):
                result = runner.execute(
                    command=("python", "-m", MODULE), payload=payload, mounts=mounts,
                    output_dir=root / "attempts" / "attempt-1",
                )
            self.assertEqual(result.failure_code, "candidate_parent_copy_mutated")
            self.assertEqual(hash_directory(parent), before)

    def test_refuses_non_trusted_candidate_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, mounts, payload = self._inputs(root)
            result = LocalShadowCandidateRunner().execute(
                command=("python", "candidate.py"), payload=payload, mounts=mounts,
                output_dir=root / "attempts" / "attempt-1",
            )
        self.assertEqual(result.failure_code, "unsupported_command")


if __name__ == "__main__":
    unittest.main()
