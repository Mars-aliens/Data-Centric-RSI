"""Audit PRISM metadata and create user-disjoint, PII-screened local splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.download_datasets import ROOT, load_registry


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{number}") from exc


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pii_manual(value: Any) -> bool:
    try:
        return not math.isnan(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _metadata_safe(row: dict[str, Any] | None) -> bool:
    if row is None or not isinstance(row.get("pii_flag"), bool) or not isinstance(row.get("en_flag"), bool):
        return False
    if row.get("pii_flag") is True or _pii_manual(row.get("pii_manual_flag")):
        return False
    if row.get("en_flag") is not True:
        return False
    moderation = row.get("moderation_flag")
    if not isinstance(moderation, dict) or not isinstance(moderation.get("flagged"), bool):
        return False
    return not bool(moderation.get("flagged"))


def _assign_split(user_id: str, seed: int) -> str:
    fraction = int(hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest(), 16) / (1 << 256)
    if fraction < 0.20:
        return "seed"
    if fraction < 0.60:
        return "feedback_pool"
    if fraction < 0.80:
        return "development"
    return "final_test"


def prepare_prism(raw_dir: Path, output_dir: Path, *, split_seed: int = 17) -> dict[str, Any]:
    utterance_path = raw_dir / "utterances.jsonl"
    metadata_path = raw_dir / "metadata.jsonl"
    download_manifest_path = raw_dir / "download-manifest.json"
    if not utterance_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("PRISM utterances.jsonl and metadata.jsonl are required")
    if not download_manifest_path.is_file():
        raise FileNotFoundError("pinned download-manifest.json is required; run scripts.download_datasets first")
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    registry_entry = load_registry()["prism"]
    if download_manifest.get("repo_id") != registry_entry["repo_id"] or download_manifest.get("revision") != registry_entry["revision"]:
        raise ValueError("PRISM download manifest does not match the pinned dataset revision")
    for path in (utterance_path, metadata_path):
        expected = download_manifest.get("files", {}).get(path.name, {}).get("sha256")
        if not expected or _sha256(path) != expected:
            raise ValueError(f"download checksum mismatch or missing hash: {path.name}")

    metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    response_metadata: dict[str, dict[str, Any]] = {}
    metadata_rows = 0
    for row in _jsonl(metadata_path):
        metadata_rows += 1
        key = (
            str(row.get("user_id") or ""),
            str(row.get("conversation_id") or ""),
            str(row.get("interaction_id") or ""),
            str(row.get("column_id") or ""),
        )
        if row.get("column_id") == "model_response" and row.get("utterance_id"):
            response_metadata[str(row["utterance_id"])] = row
        else:
            metadata[key] = row

    by_conversation_turn: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    utterances: list[dict[str, Any]] = []
    for row in _jsonl(utterance_path):
        utterances.append(row)
        by_conversation_turn[(str(row.get("conversation_id") or ""), int(row.get("turn", 0)))].append(row)

    user_splits: dict[str, str] = {}
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    for row in utterances:
        user_id = str(row.get("user_id") or "")
        conversation_id = str(row.get("conversation_id") or "")
        interaction_id = str(row.get("interaction_id") or "")
        utterance_id = str(row.get("utterance_id") or "")
        if not all((user_id, conversation_id, interaction_id, utterance_id)):
            excluded["missing_identity_or_id"] += 1
            continue

        prompt_key = (user_id, conversation_id, interaction_id, "user_prompt")
        prompt_meta = metadata.get(prompt_key)
        response_meta = response_metadata.get(utterance_id)
        if not _metadata_safe(prompt_meta):
            excluded["prompt_missing_or_flagged"] += 1
            continue
        if not _metadata_safe(response_meta):
            excluded["response_missing_or_flagged"] += 1
            continue

        prompt = row.get("user_prompt")
        response = row.get("model_response")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
            excluded["empty_text"] += 1
            continue

        turn = int(row.get("turn", 0))
        context = []
        safe_context = True
        for prior_turn in range(turn):
            prior_rows = by_conversation_turn.get((conversation_id, prior_turn), [])
            if not prior_rows:
                safe_context = False
                break
            prior = prior_rows[0]
            prior_key = (
                str(prior.get("user_id") or ""),
                conversation_id,
                str(prior.get("interaction_id") or ""),
                "user_prompt",
            )
            if not _metadata_safe(metadata.get(prior_key)):
                safe_context = False
                break
            chosen = [candidate for candidate in prior_rows if candidate.get("if_chosen") is True]
            if not chosen:
                excluded["context_missing_chosen_response"] += 1
                safe_context = False
                break
            chosen_response = chosen[0]
            if not _metadata_safe(response_metadata.get(str(chosen_response.get("utterance_id") or ""))):
                safe_context = False
                break
            context.extend(
                [
                    {"role": "user", "content": str(prior.get("user_prompt") or "")},
                    {"role": "assistant", "content": str(chosen_response.get("model_response") or "")},
                ]
            )
        if not safe_context:
            excluded["context_missing_or_flagged"] += 1
            continue

        split = user_splits.setdefault(user_id, _assign_split(user_id, split_seed))
        if split == "seed" and row.get("if_chosen") is not True:
            continue
        record = {
            "record_id": utterance_id,
            "input": prompt.strip(),
            "context": context,
            "output": response.strip(),
            "source_group_id": user_id,
            "conversation_id": conversation_id,
            "interaction_id": interaction_id,
            "turn": turn,
            "split": split,
        }
        records[split].append(record)
        labels[split].append(
            {
                "record_id": utterance_id,
                "score": int(row["score"]) if row.get("score") is not None else None,
                "if_chosen": bool(row.get("if_chosen", False)),
            }
        )

    if not records["seed"] or not records["feedback_pool"] or not records["development"] or not records["final_test"]:
        raise ValueError("one or more user-level splits are empty; inspect the split seed or dataset")

    for split in ("seed", "feedback_pool", "development", "final_test"):
        _write_jsonl(output_dir / f"{split}.jsonl", records[split])
        _write_jsonl(output_dir / "labels" / f"{split}-labels.jsonl", labels[split])

    conversations_path = raw_dir / "conversations.jsonl"
    conversation_pool = []
    conversation_labels = []
    if conversations_path.is_file():
        for row in _jsonl(conversations_path):
            user_id = str(row.get("user_id") or "")
            conversation_id = str(row.get("conversation_id") or "")
            feedback = row.get("open_feedback")
            feedback_key = (user_id, conversation_id, "", "open_feedback")
            if user_splits.get(user_id) != "feedback_pool" or not isinstance(feedback, str) or not feedback.strip():
                continue
            if not _metadata_safe(metadata.get(feedback_key)):
                excluded["open_feedback_missing_or_flagged"] += 1
                continue
            conversation_pool.append(
                {
                    "query_id": conversation_id,
                    "conversation_id": conversation_id,
                    "source_group_id": user_id,
                    "conversation_type": row.get("conversation_type"),
                }
            )
            conversation_labels.append({"query_id": conversation_id, "open_feedback": feedback.strip()})
    _write_jsonl(output_dir / "conversation_feedback_pool.jsonl", conversation_pool)
    _write_jsonl(output_dir / "labels" / "conversation-feedback-labels.jsonl", conversation_labels)

    user_sets: dict[str, set[str]] = {
        split: {record["source_group_id"] for record in rows} for split, rows in records.items()
    }
    overlap = {
        f"{left}:{right}": len(user_sets[left] & user_sets[right])
        for i, left in enumerate(user_sets)
        for right in list(user_sets)[i + 1 :]
    }
    manifest = {
        "status": "prepared",
        "dataset": load_registry()["prism"]["repo_id"],
        "revision": load_registry()["prism"]["revision"],
        "split_seed": split_seed,
        "split_unit": "user_id; all conversations for a user remain in one split",
        "raw_utterance_rows": len(utterances),
        "raw_conversation_rows": sum(1 for _ in _jsonl(conversations_path)) if conversations_path.is_file() else 0,
        "metadata_rows": metadata_rows,
        "usable_candidate_rows": sum(map(len, records.values())),
        "rows_by_split": {split: len(rows) for split, rows in records.items()},
        "users_by_split": {split: len(user_sets[split]) for split in user_sets},
        "user_overlap_counts": overlap,
        "excluded_reasons": dict(excluded),
        "label_files_are_separate": True,
        "conversation_feedback_queries": len(conversation_pool),
        "survey_downloaded": False,
        "pii_and_moderation_filtered": True,
        "raw_file_sha256": {
            "utterances.jsonl": _sha256(utterance_path),
            "metadata.jsonl": _sha256(metadata_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/prism")
    parser.add_argument("--output-dir", default="data/processed/prism")
    parser.add_argument("--split-seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    print(json.dumps(prepare_prism(raw_dir, output_dir, split_seed=args.split_seed), indent=2))


if __name__ == "__main__":
    main()
