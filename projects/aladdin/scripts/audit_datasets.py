"""Print privacy-preserving schema, split, and group-leakage diagnostics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.download_datasets import ROOT, load_registry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def audit_helpsteer2(data_dir: Path) -> dict[str, Any]:
    train_path = data_dir / "train.jsonl.gz"
    validation_path = data_dir / "validation.jsonl.gz"
    required = {"prompt", "response", "helpfulness", "correctness", "coherence", "complexity", "verbosity"}
    prompt_hashes: dict[str, set[str]] = {}
    result: dict[str, Any] = {
        "dataset": load_registry()["helpsteer2"]["repo_id"],
        "revision": load_registry()["helpsteer2"]["revision"],
        "splits": {},
    }
    for split, path in (("train", train_path), ("validation", validation_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = 0
        malformed = 0
        prompt_counts: Counter[str] = Counter()
        prompt_set = set()
        for row in _iter_jsonl(path):
            rows += 1
            if not required.issubset(row) or not row.get("prompt") or not row.get("response"):
                malformed += 1
                continue
            if any(
                not isinstance(row.get(dimension), (int, float))
                or not math.isfinite(float(row[dimension]))
                or not 0 <= row[dimension] <= 4
                for dimension in required - {"prompt", "response"}
            ):
                malformed += 1
                continue
            normalized_prompt = " ".join(row["prompt"].split())
            prompt_hash = hashlib.sha256(normalized_prompt.encode()).hexdigest()
            prompt_set.add(prompt_hash)
            prompt_counts[prompt_hash] += 1
        prompt_hashes[split] = prompt_set
        result["splits"][split] = {
            "rows": rows,
            "unique_prompts": len(prompt_set),
            "repeated_prompt_rows": sum(n - 1 for n in prompt_counts.values()),
            "malformed_rows": malformed,
            "sha256": _sha256(path),
        }
    result["train_validation_prompt_overlap"] = len(prompt_hashes["train"] & prompt_hashes["validation"])
    return result


def audit_prism(raw_dir: Path) -> dict[str, Any]:
    manifest_path = raw_dir.parent.parent / "processed/prism/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run scripts.prepare_prism before auditing prepared PRISM splits")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "dataset": load_registry()["prism"]["repo_id"],
        "revision": load_registry()["prism"]["revision"],
        "manifest": manifest,
        "raw_file_presence": {
            name: (raw_dir / name).is_file()
            for name in ("conversations.jsonl", "utterances.jsonl", "metadata.jsonl")
        },
    }


def audit_prepared_helpsteer2(processed_dir: Path) -> dict[str, Any]:
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run scripts.prepare_helpsteer2 before auditing processed HelpSteer2")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = ("seed", "feedback_pool", "development", "final_test")
    groups: dict[str, set[str]] = {}
    row_counts = {}
    for role in roles:
        path = processed_dir / f"{role}.jsonl"
        label_path = processed_dir / "labels" / f"{role}-labels.jsonl"
        if not path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing prepared HelpSteer2 data or labels for {role}")
        data_rows = list(_iter_jsonl(path))
        label_rows = list(_iter_jsonl(label_path))
        if len(data_rows) != len(label_rows):
            raise ValueError(f"data/label row count mismatch for {role}")
        groups[role] = {str(row["source_group_id"]) for row in data_rows}
        row_counts[role] = len(data_rows)
        if role != "seed" and any("ratings" in row for row in data_rows):
            raise ValueError(f"ratings leaked into feature records for {role}")
        if any(set(row.get("ratings", {})) != {"helpfulness", "correctness", "coherence", "complexity", "verbosity"} for row in label_rows):
            raise ValueError(f"incomplete rating dimensions in {role} label sidecar")
    overlap = {
        f"{left}:{right}": len(groups[left] & groups[right])
        for i, left in enumerate(roles)
        for right in roles[i + 1 :]
    }
    if any(overlap.values()):
        raise ValueError(f"prepared HelpSteer2 prompt leakage detected: {overlap}")
    if manifest.get("official_train_validation_prompt_overlap") != 0:
        raise ValueError("prepared manifest reports official train/validation prompt overlap")
    return {
        "manifest": manifest,
        "rows_by_split": row_counts,
        "prompt_group_overlap_counts": overlap,
        "label_dimensions_complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--dataset", choices=["prism", "helpsteer2", "all"], default="all")
    args = parser.parse_args()
    root = Path(args.data_root)
    if not root.is_absolute():
        root = ROOT / root
    datasets = ["prism", "helpsteer2"] if args.dataset == "all" else [args.dataset]
    reports = {}
    for name in datasets:
        if name == "prism":
            reports[name] = audit_prism(root / "prism")
        else:
            reports[name] = audit_helpsteer2(root / "helpsteer2")
            processed = root.parent / "processed" / "helpsteer2"
            if processed.exists():
                reports[name]["prepared"] = audit_prepared_helpsteer2(processed)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
