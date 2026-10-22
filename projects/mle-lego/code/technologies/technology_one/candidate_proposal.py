from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .candidate_sandbox import hash_directory


_IGNORED_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "data",
    "datasets",
    "logs",
    "models",
    "runs",
    "secrets",
}
_FORBIDDEN_PARTS = _IGNORED_NAMES | {".", ".."}
_ALLOWED_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_MAX_EDITS = 64
_MAX_FILE_BYTES = 256 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024
_MAX_CONTEXT_CHARS = 40_000
_MAX_OBSERVATION_BYTES = 24 * 1024
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_FORBIDDEN_OBSERVATION_KEYS = {
    "held_in",
    "held_out",
    "final_test",
    "test_labels",
    "unreleased_feedback",
    "gold",
}
_LABEL_OBSERVATION_KEYS = {"target", "targets", "score", "scores", "label", "labels"}
_IGNORED_SUFFIXES = (
    ".pem",
    ".key",
    ".token",
    ".secret",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".h5",
    ".model",
    ".jsonl",
    ".csv",
    ".tsv",
    ".parquet",
    ".arrow",
    ".feather",
    ".sqlite",
    ".db",
)


class CandidateProposalError(ValueError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _safe_relative_path(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise CandidateProposalError("edit path must be a non-empty POSIX relative path")
    if raw.startswith("/") or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise CandidateProposalError(f"edit path is not a normalized relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in _FORBIDDEN_PARTS for part in path.parts):
        raise CandidateProposalError(f"edit path escapes the candidate bundle: {raw!r}")
    if any(part.startswith(".env") for part in path.parts):
        raise CandidateProposalError("candidate edits cannot create or change environment files")
    if path.name == "candidate_proposal.json":
        raise CandidateProposalError("candidate_proposal.json is reserved for provenance")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise CandidateProposalError(f"candidate edit has a disallowed file type: {raw!r}")
    if any(path.name.lower().endswith(suffix) for suffix in (".pem", ".key", ".token")):
        raise CandidateProposalError("candidate edits cannot contain credentials")
    return path


def _ignored_path(relative: Path) -> bool:
    for part in relative.parts:
        lowered = part.lower()
        if part in _IGNORED_NAMES or lowered.startswith(".env"):
            return True
        if lowered in {
            "credentials",
            "credential",
            "secrets",
            "secret",
            "token",
            "tokens",
            "password",
            "passwords",
            "private_key",
            "id_rsa",
            "authorized_keys",
        }:
            return True
        if lowered.endswith(_IGNORED_SUFFIXES):
            return True
    return False


def _validate_source_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories = []
        for name in directories:
            relative = (current_path / name).relative_to(root)
            if _ignored_path(relative):
                continue
            if (current_path / name).is_symlink():
                raise CandidateProposalError(f"candidate source contains a symlink: {relative}")
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in files:
            relative = (current_path / name).relative_to(root)
            if _ignored_path(relative):
                continue
            if (current_path / name).is_symlink():
                raise CandidateProposalError(f"candidate source contains a symlink: {relative}")


def validate_candidate_proposal(
    raw: object,
    *,
    candidate_id: str,
    base_version_id: str,
) -> dict[str, Any]:
    if not _CANDIDATE_ID.fullmatch(candidate_id):
        raise CandidateProposalError("candidate_id has an invalid format")
    if not isinstance(base_version_id, str) or not base_version_id.strip():
        raise CandidateProposalError("base_version_id must be non-empty")
    if not isinstance(raw, dict):
        raise CandidateProposalError("proposal must be an object")
    rationale = raw.get("rationale")
    edits = raw.get("edits")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4000:
        raise CandidateProposalError("proposal rationale must contain 1 to 4000 characters")
    if not isinstance(edits, list) or not edits:
        raise CandidateProposalError("proposal edits must be a non-empty list")
    if len(edits) > _MAX_EDITS:
        raise CandidateProposalError(f"proposal exceeds the {_MAX_EDITS}-file edit limit")

    normalized: list[dict[str, str]] = []
    total_bytes = 0
    seen: set[str] = set()
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise CandidateProposalError(f"edit {index} must be an object")
        path = _safe_relative_path(edit.get("path"))
        path_text = path.as_posix()
        if path_text in seen:
            raise CandidateProposalError(f"duplicate edit path: {path_text}")
        seen.add(path_text)
        operation = edit.get("operation")
        if operation == "write":
            content = edit.get("content")
            if not isinstance(content, str):
                raise CandidateProposalError(f"write edit {path_text} needs string content")
            size = len(content.encode("utf-8"))
            if size > _MAX_FILE_BYTES:
                raise CandidateProposalError(f"edit {path_text} exceeds the per-file size limit")
            total_bytes += size
            normalized.append({"path": path_text, "operation": "write", "content": content})
        elif operation == "delete":
            if "content" in edit:
                raise CandidateProposalError(f"delete edit {path_text} cannot include content")
            normalized.append({"path": path_text, "operation": "delete"})
        else:
            raise CandidateProposalError(f"edit {path_text} operation must be write or delete")
    if total_bytes > _MAX_TOTAL_BYTES:
        raise CandidateProposalError("proposal exceeds the total patch size limit")
    return {
        "candidate_id": candidate_id,
        "base_version_id": base_version_id,
        "rationale": rationale.strip(),
        "edits": normalized,
    }


def _validate_observation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateProposalError("candidate observation must be an object")

    def inspect(item: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _FORBIDDEN_OBSERVATION_KEYS:
                    raise CandidateProposalError(
                        f"candidate observation contains a prohibited field: {key}"
                    )
                if (
                    normalized in _LABEL_OBSERVATION_KEYS
                    and "released_feedback" not in path
                ):
                    raise CandidateProposalError(
                        f"candidate observation contains labels outside released_feedback: {key}"
                    )
                inspect(nested, (*path, normalized))
        elif isinstance(item, (list, tuple)):
            for nested in item:
                inspect(nested, path)

    result = dict(value)
    inspect(result)
    try:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidateProposalError("candidate observation must be JSON serializable") from exc
    if len(encoded) > _MAX_OBSERVATION_BYTES:
        raise CandidateProposalError("candidate observation exceeds the 24 KiB size limit")
    return result


def _ignore_bundle(directory: str, names: list[str]) -> set[str]:
    parent = Path(directory)
    return {
        name
        for name in names
        if name == "candidate_proposal.json"
        or _ignored_path((parent / name).relative_to(parent))
    }


def _context_files(
    base_dir: Path,
    *,
    selected: Sequence[str] | None,
) -> list[dict[str, str]]:
    if selected:
        paths = [_safe_relative_path(item) for item in selected]
        candidates = [base_dir.joinpath(*path.parts) for path in paths]
    else:
        candidates = []
        for current, directories, files in os.walk(base_dir, topdown=True, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if not _ignored_path((current_path / name).relative_to(base_dir))
            ]
            candidates.extend(
                current_path / name
                for name in files
                if not _ignored_path((current_path / name).relative_to(base_dir))
                and (current_path / name).suffix.lower() in _ALLOWED_SUFFIXES
            )
        candidates.sort()
    result: list[dict[str, str]] = []
    used = 0
    for path in candidates:
        try:
            relative = path.relative_to(base_dir).as_posix()
        except ValueError as exc:
            raise CandidateProposalError("context file is outside the base bundle") from exc
        _safe_relative_path(relative)
        if not path.is_file() or path.is_symlink():
            raise CandidateProposalError(
                f"context file is missing, symlinked, or unsafe: {relative}"
            )
        text = path.read_text(encoding="utf-8")
        remaining = _MAX_CONTEXT_CHARS - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n[context truncated]"
        result.append({"path": relative, "content": text})
        used += len(text)
    if not result:
        raise CandidateProposalError("no source context is available to the Decision LLM")
    return result


def hash_candidate_sources(path: str | Path) -> str:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("candidate source hash target must be a directory")
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise CandidateProposalError(f"candidate bundle contains a symlink: {item}")
        if item.is_file() and item.name != "candidate_proposal.json":
            digest.update(item.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def read_candidate_manifest(candidate_dir: str | Path) -> dict[str, Any]:
    root = Path(candidate_dir).resolve(strict=True)
    manifest_path = root / "candidate_proposal.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateProposalError("candidate proposal manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "lego-t1-candidate-proposal-v1":
        raise CandidateProposalError("candidate proposal manifest has an unsupported schema")
    manifest_candidate_id = manifest.get("candidate_id")
    if not isinstance(manifest_candidate_id, str) or not _CANDIDATE_ID.fullmatch(
        manifest_candidate_id
    ):
        raise CandidateProposalError("candidate proposal manifest has an invalid candidate_id")
    if not isinstance(manifest.get("base_version_id"), str) or not manifest["base_version_id"].strip():
        raise CandidateProposalError("candidate proposal manifest has an invalid base_version_id")
    if manifest.get("candidate_source_sha256") != hash_candidate_sources(root):
        raise CandidateProposalError("candidate source files do not match the proposal manifest")
    return manifest


def build_candidate_prompt(
    *,
    base_dir: str | Path,
    observation: Mapping[str, Any],
    candidate_id: str,
    base_version_id: str,
    selected_context_files: Sequence[str] | None = None,
) -> tuple[str, str]:
    if not _CANDIDATE_ID.fullmatch(candidate_id):
        raise CandidateProposalError("candidate_id has an invalid format")
    if not isinstance(base_version_id, str) or not base_version_id.strip():
        raise CandidateProposalError("base_version_id must be non-empty")
    root = Path(base_dir).resolve(strict=True)
    safe_observation = _validate_observation(observation)
    context = _context_files(root, selected=selected_context_files)
    system = (
        "You propose one bounded Data System candidate modification for a long-running "
        "HelpSteer2 stream. Return JSON only with a top-level proposal object containing "
        "a non-empty rationale and a non-empty edits array. Each edit is either "
        "{path, operation:'write', content} or {path, operation:'delete'}. Paths are "
        "relative to the supplied candidate bundle. Do not include credentials, datasets, "
        "weights, run outputs, or environment files. The candidate will execute in a "
        "networkless resource-limited container and will not be promoted by this stage."
    )
    user = json.dumps(
        {
            "candidate_id": candidate_id,
            "base_version_id": base_version_id,
            "observation": safe_observation,
            "editable_bundle_files": context,
            "constraints": {
                "max_edits": _MAX_EDITS,
                "max_file_bytes": _MAX_FILE_BYTES,
                "max_total_patch_bytes": _MAX_TOTAL_BYTES,
                "candidate_must_remain_runnable": True,
                "current_batch_labels_are_not_available": True,
                "held_out_and_final_test_are_not_available": True,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return system, user


def write_candidate_bundle(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    proposal: Mapping[str, Any],
    candidate_id: str,
    base_version_id: str,
    decision_model: Mapping[str, Any],
    prompt_hashes: Mapping[str, str],
    proposal_cost: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(base_dir).resolve(strict=True)
    destination = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("base candidate bundle must be a directory")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("candidate input and output directories must not overlap")
    if destination.exists():
        raise FileExistsError(f"candidate output already exists: {destination}")
    _validate_source_tree(source)

    validated = validate_candidate_proposal(
        proposal,
        candidate_id=candidate_id,
        base_version_id=base_version_id,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(source, staging, ignore=_ignore_bundle, symlinks=False)
        base_bundle_hash = hash_directory(staging)
        changed_files: list[dict[str, str]] = []
        for edit in validated["edits"]:
            path = PurePosixPath(edit["path"])
            target = staging.joinpath(*path.parts)
            resolved = target.resolve()
            if staging.resolve() not in resolved.parents:
                raise CandidateProposalError("candidate edit resolved outside the output bundle")
            if edit["operation"] == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                content = edit["content"]
                target.write_text(content, encoding="utf-8", newline="\n")
                changed_files.append(
                    {
                        "path": edit["path"],
                        "operation": "write",
                        "sha256": _sha256(content.encode("utf-8")),
                    }
                )
            else:
                if target.exists():
                    if target.is_dir():
                        raise CandidateProposalError("candidate delete edits may only remove files")
                    target.unlink()
                changed_files.append(
                    {"path": edit["path"], "operation": "delete", "sha256": ""}
                )

        candidate_source_hash = hash_candidate_sources(staging)
        if candidate_source_hash == base_bundle_hash:
            raise CandidateProposalError("candidate proposal does not change the source bundle")

        manifest = {
            "schema": "lego-t1-candidate-proposal-v1",
            "candidate_id": candidate_id,
            "base_version_id": base_version_id,
            "rationale": validated["rationale"],
            "base_bundle_sha256": base_bundle_hash,
            "candidate_source_sha256": candidate_source_hash,
            "decision_model": {
                "provider": decision_model.get("provider", "openai_compatible"),
                "model_id": decision_model.get("model_id"),
                "revision": decision_model.get("revision"),
            },
            "prompt_sha256": dict(prompt_hashes),
            "proposal_cost": dict(proposal_cost),
            "edits": changed_files,
        }
        manifest_path = staging / "candidate_proposal.json"
        temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        if destination.exists():
            raise FileExistsError(f"candidate output already exists: {destination}")
        os.rename(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
