from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .candidate_sandbox import (
    DockerCandidateSandbox,
    SandboxLimits,
    SandboxMount,
    validate_immutable_image_reference,
)


class _FakeProcess:
    def __init__(self, *, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class DockerCandidateSandboxTests(unittest.TestCase):
    def test_immutable_image_reference_accepts_image_id_or_repository_digest(self) -> None:
        local_id = "sha256:" + "a" * 64
        repository_digest = "registry.example/lego/candidate@sha256:" + "b" * 64
        self.assertEqual(validate_immutable_image_reference(local_id), local_id)
        self.assertEqual(
            validate_immutable_image_reference(repository_digest), repository_digest
        )

    def test_immutable_image_reference_rejects_mutable_tags(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable sha256"):
            validate_immutable_image_reference("lego-candidate:latest")

    def test_builds_restricted_container_and_parses_one_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            stable = root / "stable"
            output = root / "output"
            candidate.mkdir()
            stable.mkdir()
            response = {"protocol": "test-v1", "ok": True}
            process = _FakeProcess(stdout=json.dumps(response).encode(), stderr=b"candidate log")

            def fake_popen(argv, **_kwargs):
                cidfile = Path(argv[argv.index("--cidfile") + 1])
                cidfile.write_text("container-id", encoding="ascii")
                return process

            with patch(
                "technologies.technology_one.candidate_sandbox.subprocess.Popen",
                side_effect=fake_popen,
            ) as popen, patch(
                "technologies.technology_one.candidate_sandbox.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["docker", "cp"], returncode=0, stdout=b"", stderr=b""
                ),
            ) as run:
                result = DockerCandidateSandbox(
                    image="test-image@sha256:" + "a" * 64,
                    limits=SandboxLimits(gpu_devices="0"),
                ).execute(
                    command=("python", "/candidate/run.py"),
                    payload={"protocol": "test-v1"},
                    mounts=(
                        SandboxMount(candidate, "/candidate"),
                        SandboxMount(stable, "/stable"),
                    ),
                    output_dir=output,
                )

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.response, response)
            argv = popen.call_args.args[0]
            self.assertIn("--network=none", argv)
            self.assertIn("--read-only", argv)
            self.assertIn("--cap-drop=ALL", argv)
            self.assertIn("--security-opt=no-new-privileges", argv)
            self.assertIn("--gpus", argv)
            mount_args = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--mount"]
            self.assertTrue(any("dst=/stable,readonly" in value for value in mount_args))
            self.assertTrue(any("/candidate-output:rw" in argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--tmpfs"))
            self.assertTrue(any(call.args[0][:2] == ["docker", "cp"] for call in run.call_args_list))
            self.assertTrue(any(call.args[0][:3] == ["docker", "rm", "-f"] for call in run.call_args_list))

    def test_rejects_unrestricted_mounts_and_secret_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            writable = root / "writable"
            output = root / "output"
            candidate.mkdir()
            writable.mkdir()
            sandbox = DockerCandidateSandbox(image="test-image")
            with self.assertRaisesRegex(ValueError, "only /candidate-output"):
                sandbox.execute(
                    command=("python", "run.py"),
                    payload={},
                    mounts=(
                        SandboxMount(candidate, "/candidate"),
                        SandboxMount(writable, "/state", read_only=False),
                    ),
                    output_dir=output,
                )
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                sandbox.execute(
                    command=("python", "run.py"),
                    payload={},
                    mounts=(SandboxMount(candidate, "/candidate"),),
                    output_dir=output,
                    environment={"LEGO_POLICY_API_KEY": "secret"},
                )

    def test_fault_schedule_can_probe_read_only_parent_and_short_circuit_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            stable = root / "stable"
            candidate.mkdir()
            stable.mkdir()
            completed = subprocess.CompletedProcess(
                args=["docker", "run"], returncode=1, stdout="",
                stderr="read-only file system",
            )
            sandbox = DockerCandidateSandbox(
                image="test-image",
                fault_injection={0: "write_protected_parent", 1: "timeout"},
            )
            process = _FakeProcess(stdout=b"", stderr=b"read-only file system", returncode=1)

            def fake_popen(argv, **_kwargs):
                cidfile = Path(argv[argv.index("--cidfile") + 1])
                cidfile.write_text("container-id", encoding="ascii")
                return process

            with patch(
                "technologies.technology_one.candidate_sandbox.subprocess.Popen",
                side_effect=fake_popen,
            ) as popen, patch(
                "technologies.technology_one.candidate_sandbox.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["docker", "rm"], returncode=0, stdout=b"", stderr=b""
                ),
            ):
                blocked = sandbox.execute(
                    command=("python", "/candidate/run.py"),
                    payload={"batch_index": 0},
                    mounts=(
                        SandboxMount(candidate, "/candidate"),
                        SandboxMount(stable, "/parent-checkpoint"),
                    ),
                    output_dir=root / "out-0",
                )
                timed_out = sandbox.execute(
                    command=("python", "/candidate/run.py"),
                    payload={"batch_index": 1},
                    mounts=(SandboxMount(candidate, "/candidate"),),
                    output_dir=root / "out-1",
                )

        self.assertEqual(blocked.failure_code, "nonzero_exit")
        self.assertEqual(blocked.fault_injection, "write_protected_parent")
        self.assertEqual(popen.call_args_list[0].args[0][-3:], [
            "python3",
            "-c",
            "from pathlib import Path; Path('/parent-checkpoint/.candidate-write-probe').write_text('blocked')",
        ])
        self.assertEqual(timed_out.status, "timed_out")
        self.assertEqual(timed_out.fault_injection, "timeout")

    def test_stdout_overflow_is_bounded_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            cidfile_path = None

            def fake_popen(argv, **_kwargs):
                nonlocal cidfile_path
                cidfile_path = Path(argv[argv.index("--cidfile") + 1])
                cidfile_path.write_text("container-id", encoding="ascii")
                return _FakeProcess(stdout=b"x" * (4 * 1024 * 1024 + 1))

            with patch(
                "technologies.technology_one.candidate_sandbox.subprocess.Popen",
                side_effect=fake_popen,
            ), patch(
                "technologies.technology_one.candidate_sandbox.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["docker", "rm"], returncode=0, stdout=b"", stderr=b""
                ),
            ) as run:
                result = DockerCandidateSandbox(image="test-image").execute(
                    command=("python", "run.py"),
                    payload={"batch_index": 0},
                    mounts=(SandboxMount(candidate, "/candidate"),),
                    output_dir=root / "output",
                )

        self.assertEqual(result.failure_code, "stdout_limit_exceeded")
        self.assertTrue(any(call.args[0][:3] == ["docker", "rm", "-f"] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
