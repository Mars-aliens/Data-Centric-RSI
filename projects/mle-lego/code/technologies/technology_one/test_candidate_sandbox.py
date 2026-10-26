from __future__ import annotations

import base64
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from .candidate_sandbox import (
    DockerCandidateSandbox,
    KubernetesJobCandidateSandbox,
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


class KubernetesJobCandidateSandboxTests(unittest.TestCase):
    def test_requires_digest_and_shared_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            sandbox = KubernetesJobCandidateSandbox(
                image="registry.example/candidate@sha256:" + "a" * 64,
                shared_root=root,
                pvc_name="lego-shared",
                network_policy_name="lego-t1-deny-egress",
            )
            self.assertFalse(sandbox.isolation_verified)
            with self.assertRaisesRegex(ValueError, "inside shared_root"):
                sandbox._pvc_subpath(Path(tempfile.gettempdir()).resolve())
            with self.assertRaisesRegex(ValueError, "immutable sha256"):
                KubernetesJobCandidateSandbox(
                    image="candidate:latest", shared_root=root, pvc_name="lego-shared",
                    network_policy_name="lego-t1-deny-egress",
                )

    def test_preflight_fails_closed_without_network_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = KubernetesJobCandidateSandbox(
                image="registry.example/candidate@sha256:" + "a" * 64,
                shared_root=root, pvc_name="lego-shared",
                network_policy_name="lego-t1-deny-egress",
                network_policy_enforcement_verified=True,
            )

            def fake_run(args, **_kwargs):
                if "can-i" in args:
                    return subprocess.CompletedProcess(args, 0, stdout=b"yes", stderr=b"")
                if "pvc" in args:
                    return subprocess.CompletedProcess(
                        args, 0, stdout=b'{"status":{"phase":"Bound"}}', stderr=b""
                    )
                policy = {"metadata": {"name": "lego-t1-deny-egress"},
                          "spec": {"podSelector": {"matchLabels": {
                    "app.kubernetes.io/name": "lego-t1-candidate"
                }}, "policyTypes": ["Egress"], "egress": [{"to": []}]}}
                return subprocess.CompletedProcess(
                    args, 0, stdout=json.dumps({"items": [policy]}).encode(), stderr=b""
                )

            with patch.object(sandbox, "_run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "deny all egress"):
                    sandbox.preflight()

    def test_kubernetes_request_size_is_rejected_before_job_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            sandbox = KubernetesJobCandidateSandbox(
                image="registry.example/candidate@sha256:" + "a" * 64,
                shared_root=root, pvc_name="lego-shared",
                network_policy_name="lego-t1-deny-egress",
            )
            result = sandbox.execute(
                command=("python", "run.py"), payload={"data": "x" * (910 * 1024)},
                mounts=(SandboxMount(source, "/candidate"),), output_dir=root / "output",
            )
            self.assertEqual(result.failure_code, "request_too_large")
            self.assertFalse(sandbox.isolation_verified)

    @staticmethod
    def _regression_sandbox(root, **kwargs):
        return KubernetesJobCandidateSandbox(
            image="registry.example/candidate@sha256:" + "a" * 64,
            shared_root=root, pvc_name="lego-shared",
            network_policy_name="lego-t1-deny-egress",
            network_policy_enforcement_verified=True, **kwargs,
        )

    @staticmethod
    def _regression_cluster(sandbox, manifests, *, exit_code=0, delete_fail=False,
                            create_fail=False, stdout=b'{"ok":true}', extra_policies=()):
        def run(args, **kwargs):
            data = {}
            code = 0
            if args[0] == "auth":
                return subprocess.CompletedProcess(args, 0, stdout=b"yes", stderr=b"")
            if args[:2] == ("get", "pvc"):
                data = {"status": {"phase": "Bound"}}
            elif args[:2] == ("get", "networkpolicies"):
                data = {"items": [{
                    "metadata": {"name": "lego-t1-deny-egress", "uid": "policy-uid"},
                    "spec": {"podSelector": {"matchLabels": {
                        "app.kubernetes.io/name": "lego-t1-candidate",
                    }}, "policyTypes": ["Egress"], "egress": []},
                }, *extra_policies]}
            elif args[0] == "create":
                manifests.append(json.loads(kwargs["input_data"]))
                code = int(create_fail)
            elif args[:2] == ("get", "pods"):
                data = {"items": [{
                    "metadata": {"name": "candidate-pod", "uid": "pod-uid"},
                    "status": {
                        "initContainerStatuses": [{
                            "name": "candidate", "imageID": sandbox.image,
                            "state": {"terminated": {"exitCode": exit_code}},
                        }],
                        "containerStatuses": [{
                            "name": "collector", "state": {"running": {}},
                        }],
                    },
                }]}
            elif args[0] == "logs":
                data = {
                    "returncode": exit_code,
                    "stdout_b64": base64.b64encode(stdout).decode(),
                    "stderr_tail": "candidate stderr",
                }
            elif args[0] == "delete":
                code = int(delete_fail)
            else:
                raise AssertionError(f"unexpected kubectl command: {args}")
            return subprocess.CompletedProcess(
                args, code, stdout=json.dumps(data).encode(), stderr=b"test diagnostic",
            )
        return run

    def test_k8s_regression_collects_after_candidate_exit_and_before_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            sandbox = self._regression_sandbox(root, gpu_count=1)
            manifests = []
            cluster = self._regression_cluster(sandbox, manifests)

            def export(argv, **kwargs):
                self.assertIn("collector", argv)
                self.assertIn("exec", argv)
                with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
                    entry = tarfile.TarInfo("checkpoint/adapter.bin")
                    entry.size = 7
                    archive.addfile(entry, io.BytesIO(b"adapter"))
                return subprocess.CompletedProcess(argv, 0, stderr=b"")

            with patch.object(sandbox, "_run", side_effect=cluster) as run, patch(
                "technologies.technology_one.candidate_sandbox.subprocess.run",
                side_effect=export,
            ):
                result = sandbox.execute(
                    command=("python3", "run.py"), payload={"batch_index": 0},
                    mounts=(SandboxMount(source, "/candidate"),),
                    output_dir=root / "output",
                )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.response, {"ok": True})
            self.assertEqual((root / "output/checkpoint/adapter.bin").read_bytes(), b"adapter")
            self.assertTrue(sandbox.isolation_verified)
            evidence = sandbox.isolation_evidence
            self.assertEqual(evidence["pod_uid"], "pod-uid")
            self.assertTrue(evidence["output_retrieved"])
            self.assertEqual(evidence["cleanup_errors"], [])
            pod = manifests[1]["spec"]["template"]["spec"]
            candidate, collector = pod["initContainers"][0], pod["containers"][0]
            self.assertEqual(candidate["resources"]["limits"]["nvidia.com/gpu"], 1)
            self.assertNotIn("nvidia.com/gpu", collector["resources"]["limits"])
            self.assertFalse(pod["automountServiceAccountToken"])
            self.assertTrue(candidate["securityContext"]["readOnlyRootFilesystem"])
            self.assertTrue(collector["volumeMounts"][0]["readOnly"])
            self.assertTrue(pod["volumes"][0]["persistentVolumeClaim"]["readOnly"])
            self.assertGreater(manifests[1]["spec"]["ttlSecondsAfterFinished"], 120)
            self.assertEqual([call.args[0][0] for call in run.call_args_list][-2:], ["delete", "delete"])

    def test_k8s_regression_rejects_unsupported_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "fault_injection is not supported"):
                self._regression_sandbox(root, fault_injection={1: "timeout"})
            with self.assertRaisesRegex(ValueError, "gpu_count"):
                self._regression_sandbox(root, gpu_count=True)
            with self.assertRaisesRegex(ValueError, "gpu_count"):
                self._regression_sandbox(root, limits=SandboxLimits(gpu_devices="0"))
            with self.assertRaisesRegex(ValueError, "repository digest"):
                KubernetesJobCandidateSandbox(
                    image="sha256:" + "a" * 64, shared_root=root,
                    pvc_name="lego-shared", network_policy_name="deny",
                )

    def test_k8s_regression_requires_network_enforcement_and_rejects_additive_allow(self):
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = self._regression_sandbox(Path(temporary))
            sandbox.network_policy_enforcement_verified = False
            with self.assertRaisesRegex(RuntimeError, "CNI NetworkPolicy enforcement"):
                sandbox.preflight()
            sandbox.network_policy_enforcement_verified = True
            other = {"metadata": {"name": "allow-egress"},
                     "spec": {"podSelector": {}, "egress": [{}]}}
            cluster = self._regression_cluster(sandbox, [], extra_policies=(other,))
            with patch.object(sandbox, "_run", side_effect=cluster):
                with self.assertRaisesRegex(RuntimeError, "another NetworkPolicy allows egress"):
                    sandbox.preflight()

    def test_k8s_regression_failures_leave_no_verified_isolation_and_cleanup_is_attempted(self):
        for case in ("create", "exit", "timeout", "invalid_json", "copy", "cleanup"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "candidate"
                source.mkdir()
                sandbox = self._regression_sandbox(root)
                cluster = self._regression_cluster(
                    sandbox, [], create_fail=case == "create",
                    exit_code=17 if case == "exit" else 0,
                    delete_fail=case == "cleanup",
                    stdout=b"not-json" if case == "invalid_json" else b'{"ok":true}',
                )
                wait_context = (
                    patch.object(sandbox, "_wait_candidate", side_effect=TimeoutError("deadline"))
                    if case == "timeout" else nullcontext()
                )
                with patch.object(sandbox, "_run", side_effect=cluster) as run, patch.object(
                    sandbox, "_copy_output",
                    side_effect=RuntimeError("output copy failed") if case == "copy" else None,
                ), wait_context:
                    result = sandbox.execute(
                        command=("python3", "run.py"), payload={},
                        mounts=(SandboxMount(source, "/candidate"),),
                        output_dir=root / "output",
                    )
                self.assertFalse(sandbox.isolation_verified)
                self.assertTrue(any(call.args[0][0] == "delete" for call in run.call_args_list))
                if case == "cleanup":
                    self.assertTrue(sandbox.cleanup_errors)
                else:
                    self.assertNotEqual(result.status, "completed")

    def test_k8s_regression_rejects_large_requests_before_cluster_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "candidate"
            source.mkdir()
            sandbox = self._regression_sandbox(root)
            with patch.object(sandbox, "_run") as run:
                result = sandbox.execute(
                    command=("python3", "run.py"),
                    payload={"data": "x" * (701 * 1024)},
                    mounts=(SandboxMount(source, "/candidate"),),
                    output_dir=root / "output",
                )
            self.assertEqual(result.failure_code, "request_too_large")
            run.assert_not_called()

    def test_k8s_regression_shared_path_and_output_archive_are_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sandbox = self._regression_sandbox(root)
            with self.assertRaisesRegex(ValueError, "inside shared_root"):
                sandbox.validate_shared_path(root.parent / "outside")
            self.assertEqual(sandbox.validate_shared_path(root / "future/output"), "future/output")
            for name, kind, size in (
                ("../escape", tarfile.REGTYPE, 0),
                ("link", tarfile.SYMTYPE, 0),
                ("large", tarfile.REGTYPE, 5),
            ):
                with self.subTest(name=name):
                    archive = io.BytesIO()
                    with tarfile.open(fileobj=archive, mode="w") as contents:
                        member = tarfile.TarInfo(name)
                        member.type, member.size = kind, size
                        contents.addfile(member, io.BytesIO(b"xxxxx") if size else None)
                    archive.seek(0)
                    with self.assertRaises(ValueError):
                        sandbox._extract_output(archive, root, 4)


if __name__ == "__main__":
    unittest.main()
