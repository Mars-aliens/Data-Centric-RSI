from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
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
_MAX_STDOUT_BYTES = 4 * 1024 * 1024
_STDERR_TAIL_BYTES = 4000


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
