from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    item for item in (str(CODE_ROOT), existing_pythonpath) if item
)

from core_system.mle_core.store import ArtifactStore  # noqa: E402
from technologies.technology_one.candidate_sandbox import (  # noqa: E402
    DockerCandidateSandbox,
    SandboxLimits,
    validate_immutable_image_reference,
)
from technologies.technology_one.candidate_shadow import (  # noqa: E402
    LongHorizonCandidateShadow,
)
from technologies.technology_one.candidate_proposal import (  # noqa: E402
    CandidateProposalError,
    read_candidate_manifest,
)
from technologies.technology_one.long_horizon_runner import (  # noqa: E402
    LongHorizonQwenRunner,
)
from technologies.technology_one.long_horizon_task import (  # noqa: E402
    LongHorizonTaskConfig,
    prepare_helpsteer2_long_horizon_task,
)
from technologies.technology_one.server_adapters import JsonCommand  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            rows.append(value)
    return rows


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _docker_preflight(executable: str) -> None:
    if shutil.which(executable) is None:
        raise RuntimeError(f"Docker executable not found: {executable}")
    try:
        result = subprocess.run(
            [executable, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Docker daemon preflight failed") from exc
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"Docker daemon is unavailable: {message}")


def _image_preflight(executable: str, image: str) -> None:
    try:
        result = subprocess.run(
            [executable, "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("candidate image preflight failed") from exc
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"candidate image is not available locally: {message}")


def _check_gpu_separation(candidate_devices: str | None) -> None:
    if candidate_devices is None:
        return
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValueError(
            "set CUDA_VISIBLE_DEVICES explicitly for the stable branch before assigning "
            "candidate GPU devices"
        )
    stable_raw = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    if not stable_raw:
        return
    if stable_raw == "all" or candidate_devices == "all":
        raise ValueError(
            "stable and candidate branches cannot share an unrestricted GPU assignment; "
            "use distinct physical GPU IDs or separate MIG devices"
        )
    stable_ids = {item.strip() for item in stable_raw.split(",") if item.strip()}
    candidate_ids = {item.strip() for item in candidate_devices.split(",") if item.strip()}
    if stable_ids.intersection(candidate_ids):
        raise ValueError(
            "candidate GPU must be separate from the stable branch GPU; "
            "Docker does not enforce GPU memory quotas"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired HelpSteer2 stream with a Docker-isolated T1 candidate."
    )
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--task-config-json", required=True, type=Path)
    parser.add_argument("--candidate-config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stream-seed", type=int, default=17)
    parser.add_argument("--groups-per-batch", type=int, default=50)
    parser.add_argument("--feedback-delay", choices=(0, 2, 5), type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=_positive_int,
        help="schedule only the first N deterministic development batches (smoke runs)",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=8)
    parser.add_argument("--command-timeout-seconds", type=float, default=3600)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    task_config = _object(args.task_config_json)
    base_path = task_config.get("base_model_path")
    revision = task_config.get("base_model_revision")
    if not isinstance(base_path, str) or not Path(base_path).is_dir():
        parser.error("task config must point to an existing local base_model_path")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision.strip().lower() in {"latest", "main", "master", "unrecorded"}
        or (revision.startswith("<") and revision.endswith(">"))
    ):
        parser.error("task config must pin base_model_revision to an immutable model snapshot")

    candidate_config_path = args.candidate_config_json.resolve(strict=True)
    candidate_config = _object(candidate_config_path)
    candidate_dir_raw = candidate_config.get("candidate_dir")
    if not isinstance(candidate_dir_raw, str) or not candidate_dir_raw.strip():
        parser.error("candidate config must set candidate_dir")
    candidate_dir = (candidate_config_path.parent / candidate_dir_raw).resolve(strict=True)
    if not candidate_dir.is_dir():
        parser.error("candidate_dir must be a directory")
    command = candidate_config.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item for item in command
    ):
        parser.error("candidate config command must be a non-empty argv array")
    limits_raw = candidate_config.get("limits", {})
    if not isinstance(limits_raw, dict):
        parser.error("candidate config limits must be an object")
    limits = SandboxLimits(
        timeout_seconds=float(limits_raw.get("timeout_seconds", 600)),
        memory_gib=float(limits_raw.get("memory_gib", 16)),
        cpu_cores=float(limits_raw.get("cpu_cores", 4)),
        pids_limit=int(limits_raw.get("pids_limit", 256)),
        tmpfs_mib=int(limits_raw.get("tmpfs_mib", 1024)),
        gpu_devices=limits_raw.get("gpu_devices"),
    )
    sandbox_image = candidate_config.get("image")
    if not isinstance(sandbox_image, str) or not sandbox_image.strip():
        parser.error("candidate config must set a prebuilt sandbox image")
    try:
        sandbox_image = validate_immutable_image_reference(sandbox_image)
    except ValueError as exc:
        parser.error(str(exc))
    candidate_id = candidate_config.get("candidate_id")
    if not isinstance(candidate_id, str):
        parser.error("candidate config must set candidate_id")
    require_proposal_manifest = candidate_config.get("require_proposal_manifest", True)
    if not isinstance(require_proposal_manifest, bool):
        parser.error("require_proposal_manifest must be a boolean")
    try:
        proposal_manifest = read_candidate_manifest(candidate_dir)
    except CandidateProposalError as exc:
        if require_proposal_manifest or (candidate_dir / "candidate_proposal.json").exists():
            parser.error(str(exc))
        proposal_manifest = None
    if proposal_manifest is not None and proposal_manifest.get("candidate_id") != candidate_id:
        parser.error("candidate_id does not match candidate_proposal.json")
    raw_faults = candidate_config.get("fault_injection", {})
    if not isinstance(raw_faults, dict):
        parser.error("candidate config fault_injection must map batch indexes to fault kinds")
    try:
        fault_schedule = {int(key): value for key, value in raw_faults.items()}
    except (TypeError, ValueError):
        parser.error("fault_injection keys must be non-negative batch indexes")
    _docker_preflight(args.docker_executable)
    _image_preflight(args.docker_executable, sandbox_image)
    try:
        _check_gpu_separation(limits.gpu_devices)
    except ValueError as exc:
        parser.error(str(exc))

    task = prepare_helpsteer2_long_horizon_task(
        _read_jsonl(args.train_jsonl),
        _read_jsonl(args.validation_jsonl),
        config=LongHorizonTaskConfig(
            stream_seed=args.stream_seed,
            groups_per_batch=args.groups_per_batch,
            feedback_delay_batches=args.feedback_delay,
            max_batches=args.max_batches,
        ),
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "run_id": args.run_id,
                    "task_manifest": task.agent_view.manifest,
                    "development_batches": task.scheduler.batch_count,
                    "candidate_id": candidate_config.get("candidate_id"),
                    "candidate_dir": str(candidate_dir),
                    "proposal_manifest_sha256": (
                        hashlib.sha256(
                            (candidate_dir / "candidate_proposal.json").read_bytes()
                        ).hexdigest()
                        if proposal_manifest is not None
                        else None
                    ),
                    "proposal_base_version_id": (
                        proposal_manifest.get("base_version_id")
                        if proposal_manifest is not None
                        else None
                    ),
                    "sandbox_image": sandbox_image,
                    "sandbox_image_pinned": True,
                    "fault_injection": fault_schedule,
                    "task_model_id": task_config.get("base_model_id"),
                    "task_model_revision": revision,
                    "decision_model_id": candidate_config.get("decision_model_id"),
                    "decision_model_revision": candidate_config.get("decision_model_revision"),
                    "final_test_used": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    component_versions = candidate_config.get("component_versions", {})
    config_overrides = candidate_config.get("task_config_overrides", {})
    environment = candidate_config.get("safe_environment", {})
    if not all(isinstance(value, dict) for value in (component_versions, config_overrides, environment)):
        parser.error("component_versions, task_config_overrides, and safe_environment must be objects")
    sandbox = DockerCandidateSandbox(
        image=sandbox_image,
        limits=limits,
        docker_executable=args.docker_executable,
        fault_injection=fault_schedule,
    )
    store = ArtifactStore(args.output_dir / "t1-artifacts")
    shadow = LongHorizonCandidateShadow(
        candidate_id=candidate_id,
        candidate_dir=candidate_dir,
        output_dir=args.output_dir / "candidate-shadow",
        sandbox=sandbox,
        command=command,
        store=store,
        component_versions=component_versions,
        task_config_overrides=config_overrides,
        environment=environment,
        max_reported_training_steps=candidate_config.get("max_training_steps_per_batch"),
        require_proposal_manifest=require_proposal_manifest,
    )
    stable_command = JsonCommand.from_value(
        (
            sys.executable,
            "-m",
            "technologies.technology_one.long_horizon_qwen_command",
        ),
        timeout_seconds=args.command_timeout_seconds,
        cwd=str(PROJECT_ROOT),
    )

    def report_batch(batch: dict[str, Any]) -> None:
        candidate_trial = batch.get("candidate_trial", {})
        print(
            f"completed {batch['batch_index'] + 1}/{task.scheduler.batch_count}: "
            f"{batch['batch_id']} stable={batch['model_version_id']} "
            f"candidate={candidate_trial.get('status', 'not-run')}",
            file=sys.stderr,
            flush=True,
        )

    runner = LongHorizonQwenRunner(
        task,
        command=stable_command,
        config={
            **task_config,
            "seed": args.seed,
            "prediction_batch_size": args.prediction_batch_size,
        },
        output_dir=args.output_dir / "stable",
        run_id=args.run_id,
        seed=args.seed,
        prediction_batch_size=args.prediction_batch_size,
        on_batch=report_batch,
        candidate_shadow=shadow,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
