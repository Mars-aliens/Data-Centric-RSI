from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .data_validation import validate_records
from .protocol import JsonlTaskAdapter, load_protocol


PLACEHOLDER_PREFIXES = ("SET_", "server-", "unset", "null")
REQUIRED_METHODS = {
    "fixed-harness-react",
    "react-reflexion",
    "general-self-harness",
    "curateevo-adaptation",
    "independent-edit",
    "random-joint-edit",
    "dependency-aware-joint-edit",
    "one-batch-selection",
    "three-batch-selection",
    "same-memory-no-process-update",
}
BASELINE_METHODS = {"fixed-harness-react"}
REQUIRED_COMMANDS = {"policy", "trainer", "evaluator"}


def _action_system(profile: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return the canonical Action System block and whether a legacy alias was used."""
    if "action_system" in profile:
        return profile["action_system"], False
    return profile.get("execution", {}), "execution" in profile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(PLACEHOLDER_PREFIXES)


def _walk_placeholders(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk_placeholders(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_placeholders(child, f"{path}[{index}]"))
    elif _is_placeholder(value):
        found.append(path)
    return found


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_freeze_hashes(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
        if "path" in item and "sha256" in item
    }


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def run_preflight(profile_path: Path, root: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    action_system, used_legacy_execution = _action_system(profile)
    if used_legacy_execution:
        warnings.append("profile.execution is deprecated; rename it to profile.action_system")
    if not action_system:
        errors.append("profile.action_system is required")

    if profile.get("task_id") != "helpsteer2-preference":
        errors.append("profile.task_id must be helpsteer2-preference")
    if action_system.get("final_test_unlocked", False):
        errors.append("final_test must remain locked during preflight")
    if profile.get("stream", {}).get("candidate_selection_split") != "held_in":
        errors.append("candidate_selection_split must be held_in")
    if profile.get("stream", {}).get("regression_split") != "held_out":
        errors.append("regression_split must be held_out")
    if profile.get("stream", {}).get("final_report_split") != "final_test":
        errors.append("final_report_split must be final_test")

    methods = set(profile.get("methods", []))
    scope = str(profile.get("experiment_scope", "core_matrix"))
    required_methods = BASELINE_METHODS if scope == "technology_one_baseline" else REQUIRED_METHODS
    missing_methods = sorted(required_methods - methods)
    if missing_methods:
        errors.append(f"profile is missing required methods: {missing_methods}")
    known_methods = REQUIRED_METHODS | BASELINE_METHODS
    if methods - known_methods:
        warnings.append(f"profile contains non-core methods: {sorted(methods - known_methods)}")

    commands = profile.get("commands", {})
    missing_commands = sorted(REQUIRED_COMMANDS - set(commands))
    if missing_commands:
        errors.append(f"profile is missing commands: {missing_commands}")

    placeholders = _walk_placeholders(profile)
    if placeholders:
        errors.append(f"profile contains unresolved placeholders: {placeholders}")

    dataset = profile["dataset"]
    protocol_path = _resolve(root, str(dataset["protocol"]))
    train_path = _resolve(root, str(dataset["train_jsonl"]))
    final_test_path = _resolve(root, str(dataset["final_test_jsonl"]))
    freeze_path = _resolve(root, str(dataset["freeze_manifest"]))
    for path in (protocol_path, train_path, final_test_path, freeze_path):
        if not path.exists():
            errors.append(f"missing required file: {path}")

    dataset_summary: dict[str, Any] | None = None
    final_test_summary: dict[str, Any] | None = None
    hashes: dict[str, str] = {}
    if not errors or all(path.exists() for path in (protocol_path, train_path, final_test_path, freeze_path)):
        protocol = load_protocol(protocol_path)
        adapter = JsonlTaskAdapter(protocol)
        records = adapter.load(train_path)
        final_records = adapter.load(final_test_path)
        dataset_summary = validate_records(records, protocol).to_dict()
        final_test_summary = validate_records(final_records, protocol).to_dict()
        validate_records(records + final_records, protocol)
        hashes = {
            "train_jsonl": sha256_file(train_path),
            "final_test_jsonl": sha256_file(final_test_path),
            "protocol": sha256_file(protocol_path),
            "freeze_manifest": sha256_file(freeze_path),
        }
        frozen_hashes = _load_freeze_hashes(freeze_path)
        configured_train = _normalized_path(str(dataset["train_jsonl"]))
        frozen_train = next(
            (
                value
                for key, value in frozen_hashes.items()
                if _normalized_path(key).endswith(configured_train)
            ),
            None,
        )
        if frozen_train and frozen_train != hashes["train_jsonl"]:
            errors.append("train JSONL hash does not match freeze manifest")
        if not final_test_summary.get("split_counts", {}).get("final_test"):
            errors.append("final-test file must contain final_test records")

    return {
        "status": "failed" if errors else "ready_for_server_model_configuration",
        "profile": str(profile_path),
        "python": sys.version,
        "platform": platform.platform(),
        "errors": errors,
        "warnings": warnings,
        "dataset_summary": dataset_summary,
        "final_test_summary": final_test_summary,
        "hashes": hashes,
        "methods": profile.get("methods", []),
        "final_test_unlocked": action_system.get("final_test_unlocked", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight a LEGO server profile without running models.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.profile, args.root)
    serialized = json.dumps(report, ensure_ascii=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
