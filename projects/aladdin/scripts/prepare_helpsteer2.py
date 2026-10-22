"""Create prompt-disjoint HelpSteer2 replay splits with labels in sidecars."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.download_datasets import ROOT, load_registry

DIMENSIONS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def _valid_ratings(row: dict[str, Any]) -> bool:
    for dimension in DIMENSIONS:
        value = row.get(dimension)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= 4:
            return False
    return True


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _prompt_key(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _train_role(prompt_hash: str, split_seed: int) -> str:
    bucket = int(hashlib.sha256(f"{split_seed}:{prompt_hash}".encode()).hexdigest(), 16) % 100
    return "seed" if bucket < 20 else "feedback_pool" if bucket < 60 else "development"


def prepare_helpsteer2(raw_dir: Path, output_dir: Path, *, split_seed: int = 17) -> dict[str, Any]:
    registry = load_registry()["helpsteer2"]
    download_manifest_path = raw_dir / "download-manifest.json"
    if not download_manifest_path.is_file():
        raise FileNotFoundError("pinned download-manifest.json is required; download HelpSteer2 first")
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    if download_manifest.get("repo_id") != registry["repo_id"] or download_manifest.get("revision") != registry["revision"]:
        raise ValueError("HelpSteer2 download manifest does not match the pinned dataset revision")

    train_path = raw_dir / "train.jsonl.gz"
    validation_path = raw_dir / "validation.jsonl.gz"
    expected_files = download_manifest.get("files", {})
    for path in (train_path, validation_path):
        expected_hash = expected_files.get(path.name, {}).get("sha256")
        if not expected_hash or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"download checksum mismatch or missing hash: {path.name}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_text: dict[str, str] = {}
    hashes_by_source: dict[str, set[str]] = {"train": set(), "final_test": set()}
    malformed = 0
    for source, split in ((train_path, "train"), (validation_path, "final_test")):
        for index, row in enumerate(_read_jsonl_gz(source)):
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip() or not isinstance(row.get("response"), str) or not row["response"].strip():
                malformed += 1
                continue
            if not _valid_ratings(row):
                malformed += 1
                continue
            prompt = row["prompt"].strip()
            key = _prompt_key(prompt)
            hashes_by_source[split].add(key)
            row = {**row, "_source_split": split, "_source_index": index, "_prompt_hash": key}
            grouped[key].append(row)
            prompt_text[key] = prompt

    overlap = hashes_by_source["train"] & hashes_by_source["final_test"]
    if overlap:
        raise ValueError(f"official train/validation prompt overlap found: {len(overlap)} prompts")

    data_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    role_by_hash = {
        key: "final_test" if rows[0]["_source_split"] == "final_test" else _train_role(key, split_seed)
        for key, rows in grouped.items()
    }
    for prompt_hash, rows in grouped.items():
        role = role_by_hash[prompt_hash]
        best = max(float(row["helpfulness"]) for row in rows)
        best_rows = [row for row in rows if float(row["helpfulness"]) == best]
        for row in rows:
            record_id = f"hs2-{hashlib.sha256((prompt_hash + ':' + str(row['_source_index'])).encode()).hexdigest()[:20]}"
            data_by_split[role].append(
                {
                    "record_id": record_id,
                    "input": prompt_text[prompt_hash],
                    "output": row["response"].strip(),
                    "source_group_id": prompt_hash,
                    "split": role,
                }
            )
            labels_by_split[role].append(
                {
                    "record_id": record_id,
                    "ratings": {dim: row[dim] for dim in DIMENSIONS},
                    "helpfulness_tie_for_best": float(row["helpfulness"]) == best,
                    "helpfulness_best_tie_count": len(best_rows),
                }
            )

    seed_data: list[dict[str, Any]] = []
    seed_labels: list[dict[str, Any]] = []
    for prompt_hash, rows in grouped.items():
        if role_by_hash[prompt_hash] != "seed":
            continue
        best = max(float(row["helpfulness"]) for row in rows)
        for row in rows:
            if float(row["helpfulness"]) != best:
                continue
            record_id = f"hs2-{hashlib.sha256((prompt_hash + ':' + str(row['_source_index'])).encode()).hexdigest()[:20]}"
            seed_data.append(
                {
                    "record_id": record_id,
                    "input": prompt_text[prompt_hash],
                    "output": row["response"].strip(),
                    "source_group_id": prompt_hash,
                    "split": "seed",
                }
            )
            seed_labels.append(
                {
                    "record_id": record_id,
                    "ratings": {dim: row[dim] for dim in DIMENSIONS},
                    "helpfulness_tie_for_best": True,
                }
            )
    data_by_split["seed"] = seed_data
    labels_by_split["seed"] = seed_labels

    for split in ("seed", "feedback_pool", "development", "final_test"):
        _write_jsonl(output_dir / f"{split}.jsonl", data_by_split[split])
        _write_jsonl(output_dir / "labels" / f"{split}-labels.jsonl", labels_by_split[split])

    user_groups = {split: {row["source_group_id"] for row in rows} for split, rows in data_by_split.items()}
    overlap_counts = {
        f"{left}:{right}": len(user_groups[left] & user_groups[right])
        for i, left in enumerate(user_groups)
        for right in list(user_groups)[i + 1 :]
    }
    manifest = {
        "status": "prepared",
        "dataset": registry["repo_id"],
        "revision": registry["revision"],
        "split_seed": split_seed,
        "split_unit": "normalized prompt hash; responses for one prompt remain together",
        "role_assignment": "train prompts: 20% seed, 40% feedback_pool, 40% development; official validation: final_test",
        "rows_by_split": {split: len(rows) for split, rows in data_by_split.items()},
        "prompt_groups_by_split": {split: len(groups) for split, groups in user_groups.items()},
        "prompt_group_overlap_counts": overlap_counts,
        "official_train_validation_prompt_overlap": len(overlap),
        "malformed_rows_excluded": malformed,
        "ratings_are_separate": True,
        "seed_rule": "retain all train responses tied for maximum helpfulness per seed prompt",
        "feedback_target": "higher helpfulness wins; equal helpfulness is a tie",
        "other_dimensions": list(DIMENSIONS[1:]),
        "final_test_is_official_validation": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/helpsteer2")
    parser.add_argument("--output-dir", default="data/processed/helpsteer2")
    parser.add_argument("--split-seed", type=int, default=17)
    args = parser.parse_args()
    raw_dir, output_dir = Path(args.raw_dir), Path(args.output_dir)
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    print(json.dumps(prepare_helpsteer2(raw_dir, output_dir, split_seed=args.split_seed), indent=2))


if __name__ == "__main__":
    main()
