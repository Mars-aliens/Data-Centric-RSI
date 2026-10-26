from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_CONTAINER_PATH = re.compile(r"^/(?:[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)?$")
_IMMUTABLE_IMAGE_REFERENCE = re.compile(
    r"^(?:sha256:[0-9a-f]{64}|[^@\s]+@sha256:[0-9a-f]{64})$"
)
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SAFE_ENVIRONMENT = {
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "CUDA_VISIBLE_DEVICES",
}
_FAULT_INJECTION_KINDS = {
    "timeout",
    "nonzero_exit",
    "invalid_json",
    "invalid_protocol",
    "wrong_parent",
    "missing_prediction",
    "invalid_label",
    "budget_overrun",
    "write_protected_parent",
}
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_KUBERNETES_REQUEST_BYTES = 700 * 1024
_MAX_STDOUT_BYTES = 4 * 1024 * 1024
_STDERR_TAIL_BYTES = 4000

# The candidate exits before the collector starts. Only the collector remains
# alive while the trusted host retrieves the bounded output volume.
_KUBERNETES_CAPTURE = """
import base64, json, subprocess, sys, tempfile
with open('/candidate-input/request.json', 'rb') as request:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run(sys.argv[1:], stdin=request, stdout=stdout, stderr=stderr)
        stdout.seek(0)
        data = stdout.read(4 * 1024 * 1024 + 1)
        stderr.seek(0, 2)
        stderr.seek(max(0, stderr.tell() - 4000))
        print(json.dumps({
            'returncode': result.returncode,
            'stdout_b64': base64.b64encode(data).decode('ascii'),
            'stderr_tail': stderr.read().decode('utf-8', errors='replace'),
        }), flush=True)
sys.exit(0 if result.returncode == 0 else 1)
"""
_KUBERNETES_ARCHIVE = """
import pathlib, stat, sys, tarfile
root = pathlib.Path('/candidate-output')
paths = list(root.rglob('*'))
if len(paths) > 10000:
    raise ValueError('too many output files')
total = 0
for path in paths:
    info = path.lstat()
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError('output may contain only regular files and directories')
    if stat.S_ISREG(info.st_mode):
        total += info.st_size
if total > int(sys.argv[1]):
    raise ValueError('output exceeds storage limit')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    for path in paths:
        archive.add(str(path), arcname=path.relative_to(root).as_posix(), recursive=False)
"""


def validate_immutable_image_reference(image: object) -> str:
    if not isinstance(image, str) or not _IMMUTABLE_IMAGE_REFERENCE.fullmatch(image):
        raise ValueError(
            "candidate image must be an immutable sha256 image ID or repository digest"
        )
    return image


@dataclass(frozen=True)
class SandboxMount:
    source: Path
    target: str
    read_only: bool = True

    def __post_init__(self) -> None:
        source = self.source.resolve(strict=True)
        if not source.is_dir():
            raise ValueError("sandbox mount sources must be directories")
        if not _CONTAINER_PATH.fullmatch(self.target) or self.target == "/":
            raise ValueError(f"invalid container mount target: {self.target!r}")
        if "," in str(source):
            raise ValueError("Docker bind mount paths cannot contain commas")
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 600.0
    memory_gib: float = 8.0
    cpu_cores: float = 4.0
    pids_limit: int = 256
    tmpfs_mib: int = 1024
    gpu_devices: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            not math.isfinite(self.memory_gib)
            or not math.isfinite(self.cpu_cores)
            or self.memory_gib <= 0
            or self.cpu_cores <= 0
        ):
            raise ValueError("sandbox memory and CPU limits must be positive")
        if not math.isfinite(self.timeout_seconds):
            raise ValueError("sandbox timeout must be finite")
        if self.pids_limit < 1 or self.tmpfs_mib < 1:
            raise ValueError("sandbox PID and tmpfs limits must be positive")
        if self.gpu_devices is not None and (
            not isinstance(self.gpu_devices, str)
            or not re.fullmatch(r"(?:all|[A-Za-z0-9_-]+(?:,[A-Za-z0-9_-]+)*)", self.gpu_devices)
        ):
            raise ValueError("gpu_devices must be 'all' or a comma-separated device ID list")


@dataclass(frozen=True)
class SandboxResult:
    status: str
    backend: str
    elapsed_seconds: float
    returncode: int | None
    response: dict[str, Any] | None
    response_sha256: str | None
    failure_code: str | None
    error: str | None
    stderr_tail: str
    fault_injection: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "elapsed_seconds": self.elapsed_seconds,
            "returncode": self.returncode,
            "response_sha256": self.response_sha256,
            "failure_code": self.failure_code,
            "error": self.error,
            "stderr_tail": self.stderr_tail,
            "fault_injection": self.fault_injection,
        }


class DockerCandidateSandbox:
    """Execute a candidate in a networkless, resource-limited Docker container."""

    backend_name = "docker"
    isolation_verified = True

    @property
    def isolation_evidence(self) -> dict[str, Any]:
        return {
            "container_isolation": True,
            "network_policy": "docker-network-none",
            "read_only_root": True,
            "resource_limits": True,
        }

    def __init__(
        self,
        *,
        image: str,
        limits: SandboxLimits | None = None,
        docker_executable: str = "docker",
        fault_injection: Mapping[int, str] | None = None,
    ):
        if not image.strip() or image.startswith("-"):
            raise ValueError("a Docker image name is required")
        if not docker_executable.strip():
            raise ValueError("docker_executable must be non-empty")
        self.image = image
        self.limits = limits or SandboxLimits()
        self.docker_executable = docker_executable
        self.fault_injection = {int(key): value for key, value in (fault_injection or {}).items()}
        invalid_faults = {
            value for value in self.fault_injection.values()
            if value not in _FAULT_INJECTION_KINDS
        }
        if invalid_faults:
            raise ValueError(f"unsupported candidate fault injection: {sorted(invalid_faults)}")
        if any(index < 0 for index in self.fault_injection):
            raise ValueError("fault injection batch indexes cannot be negative")

    def execute(
        self,
        *,
        command: Sequence[str],
        payload: Mapping[str, Any],
        mounts: Sequence[SandboxMount],
        output_dir: str | Path,
        environment: Mapping[str, str] | None = None,
    ) -> SandboxResult:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("candidate command must be a non-empty argv sequence")
        if any(not mount.read_only and mount.target != "/candidate-output" for mount in mounts):
            raise ValueError("only /candidate-output may be mounted read-write")
        targets = [mount.target for mount in mounts]
        if len(targets) != len(set(targets)):
            raise ValueError("sandbox mount targets must be unique")
        env = dict(environment or {})
        for key, value in env.items():
            if not _ENV_NAME.fullmatch(key) or not isinstance(value, str) or "\x00" in value:
                raise ValueError("sandbox environment contains an invalid entry")
            if key not in _SAFE_ENVIRONMENT:
                raise ValueError(f"sandbox environment variable is not allowlisted: {key}")

        output = Path(output_dir).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"candidate output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for mount in mounts:
            if output == mount.source or output in mount.source.parents or mount.source in output.parents:
                raise ValueError("candidate output must not overlap another mount source")
        mount_list = [*mounts]
        if any(mount.target == "/candidate-output" for mount in mount_list):
            raise ValueError("/candidate-output is a bounded tmpfs, not a bind mount")
        encoded_payload = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        if len(encoded_payload) > _MAX_REQUEST_BYTES:
            return SandboxResult(
                status="failed",
                backend=self.backend_name,
                elapsed_seconds=0.0,
                returncode=None,
                response=None,
                response_sha256=None,
                failure_code="request_too_large",
                error=f"candidate request exceeds {_MAX_REQUEST_BYTES} bytes",
                stderr_tail="",
            )

        batch_index = payload.get("batch_index")
        fault = self.fault_injection.get(batch_index) if isinstance(batch_index, int) else None
        execution_command = tuple(command)
        if fault == "timeout":
            execution_command = ("python3", "-c", "import time; time.sleep(3600)")
        elif fault == "nonzero_exit":
            execution_command = ("python3", "-c", "raise SystemExit(17)")
        elif fault == "invalid_json":
            execution_command = ("python3", "-c", "print('not-json')")
        elif fault == "write_protected_parent":
            execution_command = (
                "python3",
                "-c",
                "from pathlib import Path; "
                "Path('/parent-checkpoint/.candidate-write-probe').write_text('blocked')",
            )

        if fault in {"timeout", "nonzero_exit", "invalid_json"}:
            if fault == "timeout":
                return SandboxResult(
                    status="timed_out",
                    backend=self.backend_name,
                    elapsed_seconds=self.limits.timeout_seconds,
                    returncode=None,
                    response=None,
                    response_sha256=None,
                    failure_code="timeout",
                    error="injected candidate timeout",
                    stderr_tail="",
                    fault_injection=fault,
                )
        with tempfile.TemporaryDirectory(prefix="lego-candidate-") as temporary:
            cidfile = Path(temporary) / "container.id"
            argv = [
                self.docker_executable,
                "run",
                "--init",
                "--cidfile",
                str(cidfile),
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit",
                str(self.limits.pids_limit),
                "--memory",
                f"{int(self.limits.memory_gib * 1024)}m",
                "--cpus",
                str(self.limits.cpu_cores),
                "--tmpfs",
                f"/tmp:rw,noexec,nosuid,size={self.limits.tmpfs_mib}m",
                "--tmpfs",
                f"/candidate-output:rw,noexec,nosuid,size={self.limits.tmpfs_mib}m",
            ]
            if self.limits.gpu_devices is not None:
                argv.extend(["--gpus", f"device={self.limits.gpu_devices}"])
            for mount in mount_list:
                access = "readonly" if mount.read_only else "rw"
                argv.extend(
                    [
                        "--mount",
                        f"type=bind,src={mount.source},dst={mount.target},{access}",
                    ]
                )
            for key, value in sorted(env.items()):
                argv.extend(["--env", f"{key}={value}"])
            argv.extend(["--workdir", "/candidate", self.image, *execution_command])

            return self._execute_container(
                argv=argv,
                encoded_payload=encoded_payload,
                output=output,
                cidfile=cidfile,
                fault=fault,
            )

    def _execute_container(
        self,
        *,
        argv: Sequence[str],
        encoded_payload: bytes,
        output: Path,
        cidfile: Path,
        fault: str | None,
    ) -> SandboxResult:
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        stdout_data = bytearray()
        stderr_tail = bytearray()
        stdout_overflow = threading.Event()
        readers: list[threading.Thread] = []
        sender: threading.Thread | None = None
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_pipe, stderr_pipe, stdin_pipe = process.stdout, process.stderr, process.stdin

            def capture_stdout() -> None:
                while chunk := stdout_pipe.read(64 * 1024):
                    remaining = _MAX_STDOUT_BYTES - len(stdout_data)
                    if remaining > 0:
                        stdout_data.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        stdout_overflow.set()

            def capture_stderr() -> None:
                while chunk := stderr_pipe.read(64 * 1024):
                    stderr_tail.extend(chunk)
                    if len(stderr_tail) > _STDERR_TAIL_BYTES:
                        del stderr_tail[:-_STDERR_TAIL_BYTES]

            def send_payload() -> None:
                try:
                    stdin_pipe.write(encoded_payload)
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    try:
                        stdin_pipe.close()
                    except OSError:
                        pass

            readers = [
                threading.Thread(target=capture_stdout, daemon=True),
                threading.Thread(target=capture_stderr, daemon=True),
            ]
            for reader in readers:
                reader.start()
            sender = threading.Thread(target=send_payload, daemon=True)
            sender.start()

            timed_out = False
            while process.poll() is None:
                if stdout_overflow.is_set():
                    self._kill_container(cidfile)
                    try:
                        process.kill()
                    except OSError:
                        pass
                    break
                if time.monotonic() - started >= self.limits.timeout_seconds:
                    timed_out = True
                    self._kill_container(cidfile)
                    try:
                        process.kill()
                    except OSError:
                        pass
                    break
                time.sleep(0.05)
            returncode = process.wait()
            sender.join(timeout=5)
            for reader in readers:
                reader.join(timeout=5)
            elapsed = time.monotonic() - started
            stderr_text = bytes(stderr_tail).decode("utf-8", errors="replace")

            if timed_out:
                return SandboxResult(
                    status="timed_out",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=None,
                    response=None,
                    response_sha256=None,
                    failure_code="timeout",
                    error=f"candidate exceeded {self.limits.timeout_seconds}s",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )
            if stdout_overflow.is_set():
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="stdout_limit_exceeded",
                    error=f"candidate stdout exceeded {_MAX_STDOUT_BYTES} bytes",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )
            if returncode != 0:
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="nonzero_exit",
                    error=f"candidate exited with status {returncode}",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )

            stdout = bytes(stdout_data)
            try:
                response = json.loads(stdout)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="invalid_json",
                    error=f"candidate stdout was not one JSON object: {exc}",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )
            if not isinstance(response, dict):
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="invalid_response_type",
                    error="candidate stdout must decode to a JSON object",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )

            container_id = self._container_id(cidfile)
            if container_id is None:
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="output_copy_failed",
                    error="Docker did not write a container ID",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )
            try:
                copied = subprocess.run(
                    [
                        self.docker_executable,
                        "cp",
                        f"{container_id}:/candidate-output/.",
                        str(output),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="output_copy_failed",
                    error=f"could not copy bounded candidate output: {exc}",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )
            if copied.returncode != 0:
                copy_error = copied.stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL_BYTES:]
                return SandboxResult(
                    status="failed",
                    backend=self.backend_name,
                    elapsed_seconds=elapsed,
                    returncode=returncode,
                    response=None,
                    response_sha256=None,
                    failure_code="output_copy_failed",
                    error=f"docker cp exited with status {copied.returncode}: {copy_error}",
                    stderr_tail=stderr_text,
                    fault_injection=fault,
                )
            if fault == "invalid_protocol":
                response["protocol"] = "injected-invalid-protocol"
            elif fault == "wrong_parent":
                response["parent_version_id"] = "injected-wrong-parent"
            elif fault == "missing_prediction" and isinstance(response.get("predictions"), list):
                response["predictions"] = response["predictions"][:-1]
            elif fault == "invalid_label" and isinstance(response.get("predictions"), list):
                if response["predictions"]:
                    response["predictions"][0]["prediction"] = "injected-invalid-label"
            elif fault == "budget_overrun" and isinstance(response.get("cost"), dict):
                response["cost"]["training_steps"] = 10**9
            return SandboxResult(
                status="completed",
                backend=self.backend_name,
                elapsed_seconds=elapsed,
                returncode=returncode,
                response=response,
                response_sha256=hashlib.sha256(stdout).hexdigest(),
                failure_code=None,
                error=None,
                stderr_tail=stderr_text,
                fault_injection=fault,
            )
        except OSError as exc:
            return SandboxResult(
                status="failed",
                backend=self.backend_name,
                elapsed_seconds=time.monotonic() - started,
                returncode=None,
                response=None,
                response_sha256=None,
                failure_code="sandbox_unavailable",
                error=f"could not execute candidate sandbox: {exc}",
                stderr_tail=bytes(stderr_tail).decode("utf-8", errors="replace"),
                fault_injection=fault,
            )
        finally:
            if process is not None and process.poll() is None:
                self._kill_container(cidfile)
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()
            if sender is not None:
                sender.join(timeout=5)
            for reader in readers:
                reader.join(timeout=5)
            self._remove_container(cidfile)

    @staticmethod
    def _container_id(cidfile: Path) -> str | None:
        try:
            value = cidfile.read_text(encoding="ascii").strip()
        except OSError:
            return None
        return value or None

    def _remove_container(self, cidfile: Path) -> None:
        container_id = self._container_id(cidfile)
        if container_id is None:
            return
        try:
            subprocess.run(
                [self.docker_executable, "rm", "-f", container_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _kill_container(self, cidfile: Path) -> None:
        try:
            container_id = self._container_id(cidfile)
        except OSError:
            return
        if container_id is not None:
            try:
                subprocess.run(
                    [self.docker_executable, "kill", container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass


class KubernetesJobCandidateSandbox:
    """Execute candidates in restricted Kubernetes Jobs backed by a shared PVC."""

    backend_name = "kubernetes"

    def __init__(
        self,
        *,
        image: str,
        shared_root: str | Path,
        pvc_name: str,
        network_policy_name: str,
        namespace: str = "default",
        limits: SandboxLimits | None = None,
        kubectl_executable: str = "kubectl",
        gpu_count: int = 0,
        network_policy_enforcement_verified: bool = False,
        fault_injection: Mapping[int, str] | None = None,
    ):
        self.image = validate_immutable_image_reference(image)
        if "@sha256:" not in self.image:
            raise ValueError("Kubernetes image must use a pullable repository digest")
        if fault_injection:
            raise ValueError("Kubernetes fault_injection is not supported; use an empty schedule")
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise ValueError("Kubernetes gpu_count must be a non-negative integer")
        self.shared_root = Path(shared_root).resolve(strict=True)
        if not self.shared_root.is_dir():
            raise ValueError("shared_root must be a directory")
        for name, value in (("pvc_name", pvc_name), ("namespace", namespace),
                            ("network_policy_name", network_policy_name)):
            if not isinstance(value, str) or not re.fullmatch(
                r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?", value
            ):
                raise ValueError(f"invalid Kubernetes {name}")
        if not kubectl_executable.strip():
            raise ValueError("kubectl_executable must be non-empty")
        self.pvc_name = pvc_name
        self.namespace = namespace
        self.network_policy_name = network_policy_name
        self.limits = limits or SandboxLimits()
        if self.limits.gpu_devices is not None:
            raise ValueError("Kubernetes uses gpu_count, not physical gpu_devices")
        self.gpu_count = gpu_count
        self.network_policy_enforcement_verified = network_policy_enforcement_verified
        self.kubectl_executable = kubectl_executable
        self.fault_injection: dict[int, str] = {}
        self.isolation_verified = False
        self._execution_evidence: dict[str, Any] = {}
        self.cleanup_errors: list[str] = []

    @property
    def isolation_evidence(self) -> dict[str, Any]:
        return {
            "container_isolation": self.isolation_verified,
            "network_policy": self.network_policy_name,
            "read_only_root": True,
            "resource_limits": True,
            "shared_pvc": self.pvc_name,
            "network_enforcement_basis": "administrator_verified",
            "network_policy_enforcement_verified": self.network_policy_enforcement_verified is True,
            "pid_limit_enforced_by_runner": False,
            "cleanup_errors": list(self.cleanup_errors),
            **self._execution_evidence,
        }

    def run_configuration(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "shared_root": str(self.shared_root),
            "pvc_name": self.pvc_name,
            "network_policy_name": self.network_policy_name,
            "network_policy_enforcement_verified": self.network_policy_enforcement_verified,
            "gpu_count": self.gpu_count,
        }

    def _run(self, args: Sequence[str], *, input_data: bytes | None = None,
             timeout: float = 30) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.kubectl_executable, "-n", self.namespace, *args],
            input=input_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )

    def preflight(self) -> None:
        if self.network_policy_enforcement_verified is not True:
            raise RuntimeError(
                "administrator must verify CNI NetworkPolicy enforcement and set "
                "network_policy_enforcement_verified=true"
            )
        for check in (
            ("auth", "can-i", "create", "jobs"),
            ("auth", "can-i", "get", "jobs"),
            ("auth", "can-i", "create", "configmaps"),
            ("auth", "can-i", "get", "pods/log"),
            ("auth", "can-i", "get", "pods"),
            ("auth", "can-i", "list", "pods"),
            ("auth", "can-i", "create", "pods/exec"),
            ("auth", "can-i", "delete", "jobs"),
            ("auth", "can-i", "delete", "configmaps"),
        ):
            result = self._run(check)
            if result.returncode or result.stdout.decode("utf-8", errors="replace").strip() != "yes":
                raise RuntimeError(f"Kubernetes permission preflight failed: {' '.join(check)}")
        pvc = self._json(("get", "pvc", self.pvc_name, "-o", "json"))
        if pvc.get("status", {}).get("phase") != "Bound":
            raise RuntimeError("Kubernetes candidate PVC must be Bound")
        policies = self._json(("get", "networkpolicies", "-o", "json")).get("items", [])
        selected = [
            policy for policy in policies
            if self._selector_matches(
                policy.get("spec", {}).get("podSelector", {}),
                {"app.kubernetes.io/name": "lego-t1-candidate"},
            )
        ]
        expected = next(
            (policy for policy in selected
             if policy.get("metadata", {}).get("name") == self.network_policy_name), None
        )
        if (expected is None
                or "Egress" not in expected.get("spec", {}).get("policyTypes", [])
                or expected["spec"].get("egress", []) != []):
            raise RuntimeError("NetworkPolicy must select T1 candidate Pods and deny all egress")
        if any(policy.get("spec", {}).get("egress") for policy in policies):
            raise RuntimeError(
                "another NetworkPolicy allows egress; use a dedicated candidate namespace"
            )
        self._execution_evidence["network_policy_uid"] = expected.get("metadata", {}).get("uid")

    def _json(self, args: Sequence[str]) -> dict[str, Any]:
        result = self._run(args)
        if result.returncode:
            raise RuntimeError(
                f"kubectl {' '.join(args)} failed: "
                + result.stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL_BYTES:]
            )
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("kubectl response must be a JSON object")
        return value

    @staticmethod
    def _selector_matches(selector: Mapping[str, Any], labels: Mapping[str, str]) -> bool:
        if any(labels.get(key) != value for key, value in selector.get("matchLabels", {}).items()):
            return False
        for expression in selector.get("matchExpressions", []):
            key, operator = expression["key"], expression["operator"]
            values = expression.get("values", [])
            if operator == "In" and labels.get(key) not in values:
                return False
            if operator == "NotIn" and labels.get(key) in values:
                return False
            if operator == "Exists" and key not in labels:
                return False
            if operator == "DoesNotExist" and key in labels:
                return False
            if operator not in {"In", "NotIn", "Exists", "DoesNotExist"}:
                raise ValueError("unsupported NetworkPolicy selector")
        return True

    def validate_shared_path(self, path: str | Path) -> str:
        """Check a present or future orchestrator path against the PVC root."""
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.shared_root)
        except ValueError as exc:
            raise ValueError(f"Kubernetes candidate path must be inside shared_root: {resolved}") from exc
        if not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise ValueError("invalid PVC subPath")
        return relative.as_posix()

    def _pvc_subpath(self, path: Path) -> str:
        return self.validate_shared_path(path.resolve(strict=True))

    def execute(
        self, *, command: Sequence[str], payload: Mapping[str, Any],
        mounts: Sequence[SandboxMount], output_dir: str | Path,
        environment: Mapping[str, str] | None = None,
    ) -> SandboxResult:
        started = time.monotonic()
        self.isolation_verified = False
        self._execution_evidence = {}
        self.cleanup_errors = []
        job_name = config_name = None
        try:
            if not command or any(not isinstance(item, str) or not item for item in command):
                raise ValueError("candidate command must be a non-empty argv sequence")
            if any(not mount.read_only for mount in mounts):
                raise ValueError("Kubernetes source mounts must be read-only")
            allowed_targets = {"/candidate", "/parent-checkpoint", "/models/task"}
            if any(mount.target not in allowed_targets for mount in mounts):
                raise ValueError("Kubernetes mount target is not an approved candidate input")
            targets = [mount.target for mount in mounts]
            if len(targets) != len(set(targets)):
                raise ValueError("sandbox mount targets must be unique")
            env = dict(environment or {})
            for key, value in env.items():
                if (not _ENV_NAME.fullmatch(key) or not isinstance(value, str)
                        or "\x00" in value or key not in _SAFE_ENVIRONMENT):
                    raise ValueError(f"sandbox environment variable is not allowlisted: {key}")
            output = Path(output_dir).resolve()
            self.validate_shared_path(output)
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise FileExistsError(f"candidate output directory is not empty: {output}")
            mount_specs = []
            for index, mount in enumerate(mounts):
                if (output == mount.source or output in mount.source.parents
                        or mount.source in output.parents):
                    raise ValueError("candidate output must not overlap a source mount")
                subpath = self._pvc_subpath(mount.source)
                mount_specs.append({
                    "name": f"source-{index}", "target": mount.target,
                    "subPath": subpath, "readOnly": True,
                })
            encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            if len(encoded) > _MAX_KUBERNETES_REQUEST_BYTES:
                return SandboxResult(
                    "failed", self.backend_name, 0.0, None, None, None,
                    "request_too_large",
                    f"candidate request exceeds ConfigMap limit ({_MAX_KUBERNETES_REQUEST_BYTES} bytes)", "",
                )
            self.preflight()
            token = uuid.uuid4().hex[:12]
            job_name, config_name = f"lego-t1-{token}", f"lego-t1-input-{token}"
            labels = {"app.kubernetes.io/name": "lego-t1-candidate"}
            config = {
                "apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": config_name},
                "immutable": True,
                "binaryData": {"request.json": base64.b64encode(encoded).decode("ascii")},
            }
            volume_mounts = [{
                "name": spec["name"], "mountPath": spec["target"],
                "subPath": spec["subPath"], "readOnly": True,
            } for spec in mount_specs]
            volume_mounts.extend([
                {"name": "candidate-output", "mountPath": "/candidate-output"},
                {"name": "candidate-input", "mountPath": "/candidate-input", "readOnly": True},
                {"name": "candidate-tmp", "mountPath": "/tmp"},
            ])
            resources = {
                "requests": {"cpu": str(self.limits.cpu_cores), "memory": f"{self.limits.memory_gib}Gi"},
                "limits": {"cpu": str(self.limits.cpu_cores), "memory": f"{self.limits.memory_gib}Gi"},
            }
            if self.gpu_count:
                resources["limits"]["nvidia.com/gpu"] = self.gpu_count
                resources["requests"]["nvidia.com/gpu"] = self.gpu_count
            security = {
                "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                "runAsNonRoot": True, "runAsUser": 65532, "runAsGroup": 65532,
                "capabilities": {"drop": ["ALL"]},
            }
            env.setdefault("HF_HOME", "/tmp/huggingface")
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            container = {
                "name": "candidate", "image": self.image,
                "command": ["python3", "-I", "-B", "-c", _KUBERNETES_CAPTURE, *command],
                "workingDir": "/candidate",
                "resources": resources,
                "securityContext": security,
                "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
                "volumeMounts": volume_mounts,
            }
            volumes = [
                {"name": spec["name"], "persistentVolumeClaim": {
                    "claimName": self.pvc_name, "readOnly": True,
                }}
                for spec in mount_specs
            ]
            volumes.extend([
                {"name": "candidate-output", "emptyDir": {"medium": "Memory", "sizeLimit": f"{self.limits.tmpfs_mib}Mi"}},
                {"name": "candidate-input", "configMap": {"name": config_name}},
                {"name": "candidate-tmp", "emptyDir": {"medium": "Memory", "sizeLimit": f"{self.limits.tmpfs_mib}Mi"}},
            ])
            collector = {
                "name": "collector", "image": self.image,
                "command": ["python3", "-I", "-B", "-c", "import time; time.sleep(3600)"],
                "workingDir": "/", "securityContext": security,
                "resources": {
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                    "limits": {"cpu": "1", "memory": "256Mi"},
                },
                "volumeMounts": [{
                    "name": "candidate-output", "mountPath": "/candidate-output",
                    "readOnly": True,
                }],
            }
            job = {
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"name": job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": math.ceil(self.limits.timeout_seconds) + 180,
                    "ttlSecondsAfterFinished": 3600,
                    "template": {
                        "metadata": {"labels": labels},
                        "spec": {
                            "restartPolicy": "Never", "automountServiceAccountToken": False,
                            "enableServiceLinks": False,
                            "securityContext": {
                                "seccompProfile": {"type": "RuntimeDefault"},
                                "fsGroup": 65532,
                            },
                            "initContainers": [container],
                            "containers": [collector], "volumes": volumes,
                        },
                    },
                },
            }
            for manifest in (config, job):
                applied = self._run(("create", "-f", "-"), input_data=json.dumps(manifest).encode())
                if applied.returncode:
                    raise RuntimeError(applied.stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL_BYTES:])
            pod, terminated = self._wait_candidate(job_name)
            pod_name = pod["metadata"]["name"]
            logs = self._run((
                "logs", pod_name, "-c", "candidate",
                f"--limit-bytes={2 * _MAX_STDOUT_BYTES}",
            ))
            if logs.returncode:
                raise RuntimeError("could not read Kubernetes candidate logs")
            envelope = json.loads(logs.stdout)
            stderr = str(envelope.get("stderr_tail", ""))[-_STDERR_TAIL_BYTES:]
            returncode = envelope.get("returncode")
            if terminated.get("exitCode") != 0 or returncode != 0:
                return SandboxResult(
                    "failed", self.backend_name, time.monotonic() - started,
                    returncode, None, None, "nonzero_exit",
                    f"Kubernetes candidate exited with status {returncode}", stderr,
                )
            stdout = base64.b64decode(envelope["stdout_b64"], validate=True)
            if len(stdout) > _MAX_STDOUT_BYTES:
                return SandboxResult("failed", self.backend_name, time.monotonic()-started,
                                     0, None, None, "stdout_limit_exceeded",
                                     "candidate stdout exceeded output limit", stderr)
            response = json.loads(stdout)
            if not isinstance(response, dict):
                raise ValueError("candidate stdout must decode to a JSON object")
            image_status = next(
                item for item in pod["status"]["initContainerStatuses"]
                if item.get("name") == "candidate"
            )
            image_id = image_status.get("imageID", "")
            digest = self.image.split("@", 1)[1]
            if not image_id.endswith(digest):
                raise RuntimeError("candidate runtime image ID does not match its pinned digest")
            if not pod["metadata"].get("uid"):
                raise RuntimeError("candidate Pod UID is missing from execution evidence")
            self._copy_output(pod_name, output)
            self._execution_evidence.update({
                "job_name": job_name,
                "pod_uid": pod["metadata"].get("uid"),
                "image_id": image_id,
                "candidate_exit_code": terminated["exitCode"],
                "output_retrieved": True,
            })
            self.isolation_verified = True
            return SandboxResult(
                "completed", self.backend_name, time.monotonic()-started, 0, response,
                hashlib.sha256(stdout).hexdigest(), None, None, stderr,
            )
        except TimeoutError as exc:
            return SandboxResult(
                "timed_out", self.backend_name, time.monotonic() - started,
                None, None, None, "timeout", str(exc), "",
            )
        except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError,
                KeyError, TypeError, IndexError, StopIteration, tarfile.TarError) as exc:
            return SandboxResult(
                "failed", self.backend_name, time.monotonic()-started, None, None, None,
                "sandbox_unavailable" if isinstance(exc, (OSError, subprocess.TimeoutExpired)) else "sandbox_error",
                str(exc), "",
            )
        finally:
            for resource in (
                f"job/{job_name}" if job_name else None,
                f"configmap/{config_name}" if config_name else None,
            ):
                if resource is not None:
                    try:
                        deleted = self._run((
                            "delete", resource, "--ignore-not-found=true",
                            "--cascade=foreground", "--wait=true", "--timeout=20s",
                        ))
                        if deleted.returncode:
                            self.cleanup_errors.append(f"could not delete {resource}")
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        self.cleanup_errors.append(f"could not delete {resource}: {exc}")
            if self.cleanup_errors:
                self.isolation_verified = False

    def _wait_candidate(self, job_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + self.limits.timeout_seconds
        while time.monotonic() < deadline:
            pods = self._json((
                "get", "pods", "-l", f"job-name={job_name}", "-o", "json",
            )).get("items", [])
            for pod in pods:
                status = pod.get("status", {})
                for container in status.get("initContainerStatuses", []):
                    if container.get("name") != "candidate":
                        continue
                    terminated = container.get("state", {}).get("terminated")
                    if terminated is not None:
                        collector_ready = any(
                            entry.get("name") == "collector"
                            and "running" in entry.get("state", {})
                            for entry in status.get("containerStatuses", [])
                        )
                        if terminated.get("exitCode") != 0 or collector_ready:
                            return pod, terminated
                if status.get("phase") == "Failed":
                    raise RuntimeError(f"Kubernetes candidate Pod failed: {status.get('reason', '')}")
            job = self._json(("get", f"job/{job_name}", "-o", "json"))
            if job.get("status", {}).get("failed", 0):
                raise RuntimeError("Kubernetes candidate Job failed")
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError("Kubernetes candidate exceeded its scheduling/execution deadline")

    def _copy_output(self, pod_name: str, output: Path) -> None:
        limit = self.limits.tmpfs_mib * 1024 * 1024
        with tempfile.TemporaryFile() as archive:
            result = subprocess.run(
                [self.kubectl_executable, "-n", self.namespace, "exec", pod_name,
                 "-c", "collector", "--", "python3", "-I", "-B", "-c",
                 _KUBERNETES_ARCHIVE, str(limit)],
                stdout=archive, stderr=subprocess.PIPE, timeout=120, check=False,
            )
            if result.returncode:
                raise RuntimeError("could not retrieve Kubernetes candidate output")
            archive.seek(0, 2)
            if archive.tell() > limit + 16 * 1024 * 1024:
                raise ValueError("candidate output archive exceeds its storage limit")
            archive.seek(0)
            self._extract_output(archive, output, limit)

    @staticmethod
    def _extract_output(archive: Any, output: Path, limit: int) -> None:
        total = count = 0
        with tarfile.open(fileobj=archive, mode="r|") as contents:
            for member in contents:
                count += 1
                if count > 10000:
                    raise ValueError("candidate output contains too many files")
                relative = Path(member.name)
                if (relative.is_absolute() or relative.drive or ".." in relative.parts
                        or "\\" in member.name or ":" in member.name):
                    raise ValueError("candidate output path escapes its output directory")
                destination = (output / relative).resolve()
                if destination != output and output not in destination.parents:
                    raise ValueError("candidate output path escapes its output directory")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    total += member.size
                    if member.size < 0 or total > limit:
                        raise ValueError("candidate output exceeds its storage limit")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = contents.extractfile(member)
                    if source is None:
                        raise ValueError("candidate output file is missing")
                    with source, destination.open("xb") as handle:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            handle.write(block)
                else:
                    raise ValueError("candidate output may contain only regular files and directories")


def hash_directory(path: str | Path) -> str:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact hash target must be a directory")
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"artifact directories may not contain symlinks: {item}")
        if item.is_file():
            digest.update(str(item.relative_to(root)).replace(os.sep, "/").encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()
