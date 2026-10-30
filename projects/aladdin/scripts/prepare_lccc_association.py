"""Prepare grouped Chinese continuation samples from the pinned LCCC-base files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "silver/lccc"
DATASET_REVISION = "5bd582fa28cd7143f2f9c852e08e23089d677c44"
FILES = {
    "train": "lccc_base_train.jsonl.gz",
    "development": "lccc_base_valid.jsonl.gz",
    "final_test": "lccc_base_test.jsonl.gz",
}
PRIORITY = {"final_test": 0, "development": 1, "train": 2}
CUT_RATIOS = (0.25, 0.5, 0.75)
MIN_FRAGMENT_CHARS = 4
HAN_PUNCT = r"\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
TOKEN_SPACE_RE = re.compile(
    rf"(?<=[{HAN_PUNCT}])\s+(?=[{HAN_PUNCT}.,!?;:])"
    rf"|(?<=[{HAN_PUNCT}.,!?;:])\s+(?=[{HAN_PUNCT}])"
)
DIRECT_IDENTIFIER_PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "url": re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE),
    "mainland_mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "mainland_id": re.compile(
        r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_utterance(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    return TOKEN_SPACE_RE.sub("", text)


def _identifier_categories(dialogue: list[str]) -> set[str]:
    text = "\n".join(dialogue)
    return {
        category
        for category, pattern in DIRECT_IDENTIFIER_PATTERNS.items()
        if pattern.search(text)
    }


def _read_dialogues(
    path: Path, audit: dict[str, int] | None = None
) -> Iterator[list[str]]:
    audit = audit if audit is not None else {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            audit["rows_seen"] = audit.get("rows_seen", 0) + 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                audit["invalid_json"] = audit.get("invalid_json", 0) + 1
                raise ValueError(f"invalid JSON at {path.name}:{line_number}") from exc
            dialogue = row.get("dialog") if isinstance(row, dict) else None
            if not isinstance(dialogue, list):
                audit["invalid_schema"] = audit.get("invalid_schema", 0) + 1
                continue
            if len(dialogue) < 2:
                audit["too_short"] = audit.get("too_short", 0) + 1
                continue
            if any(not isinstance(item, str) for item in dialogue):
                audit["non_string_utterance"] = audit.get("non_string_utterance", 0) + 1
                continue
            normalized = [_normalize_utterance(item) for item in dialogue]
            if any(not item for item in normalized):
                audit["empty_utterance"] = audit.get("empty_utterance", 0) + 1
                continue
            identifier_categories = _identifier_categories(normalized)
            if identifier_categories:
                audit["direct_identifier_dialogues_excluded"] = (
                    audit.get("direct_identifier_dialogues_excluded", 0) + 1
                )
                for category in identifier_categories:
                    key = f"identifier_category_{category}"
                    audit[key] = audit.get(key, 0) + 1
                continue
            audit["usable"] = audit.get("usable", 0) + 1
            yield normalized


def _dialogue_id(dialogue: list[str]) -> str:
    payload = json.dumps(dialogue, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_examples(
    dialogue: list[str], group_id: str, split: str
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    skipped_targets = 0
    for turn_index in range(1, len(dialogue)):
        target = dialogue[turn_index]
        cut_points = set()
        for ratio in CUT_RATIOS:
            cut = round(len(target) * ratio)
            if (
                MIN_FRAGMENT_CHARS <= cut
                and len(target) - cut >= MIN_FRAGMENT_CHARS
            ):
                cut_points.add(cut)
        if not cut_points:
            skipped_targets += 1
            continue

        context = dialogue[max(0, turn_index - 4) : turn_index]
        for cut in sorted(cut_points):
            prefix = target[:cut]
            continuation = target[cut:]
            record_id = hashlib.sha256(
                f"{group_id}:{turn_index}:{cut}".encode("utf-8")
            ).hexdigest()[:24]
            records.append(
                {
                    "schema_version": 1,
                    "record_id": f"lccc-{record_id}",
                    "source_group_id": f"lccc-{group_id}",
                    "source_dialogue_id": f"lccc-{group_id}",
                    "source_turn": turn_index,
                    "context": context,
                    "typed_prefix": prefix,
                    "input": (
                        "对话上下文：\n"
                        + "\n".join(context)
                        + "\n已输入："
                        + prefix
                    ),
                    "output": continuation,
                    "reference_continuation": continuation,
                    "cut_ratio": cut / len(target),
                    "source_split": split,
                    "split": split,
                }
            )
    return records, skipped_targets


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def _verify_download(raw_dir: Path) -> dict[str, Any]:
    manifest_path = raw_dir / "download-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("LCCC download-manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != "lccc"
        or manifest.get("repo_id") != DATASET_ID
        or manifest.get("revision") != DATASET_REVISION
        or manifest.get("terms_acknowledged") is not True
    ):
        raise ValueError("LCCC source, revision, or terms acknowledgement mismatch")
    for name in FILES.values():
        path = raw_dir / name
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise ValueError(f"LCCC file missing or hash mismatch: {name}")
    return manifest


def _create_database(path: Path, raw_dir: Path, split_seed: int) -> dict[str, int]:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE dialogues (
                group_id TEXT PRIMARY KEY,
                source_split TEXT NOT NULL,
                priority INTEGER NOT NULL,
                order_key TEXT NOT NULL,
                dialogue_json TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX dialogues_role ON dialogues(source_split, order_key)")
        raw_counts = {}
        row_audits = {}
        for split, filename in FILES.items():
            seen = 0
            row_audit: dict[str, int] = {}
            for dialogue in _read_dialogues(raw_dir / filename, row_audit):
                seen += 1
                group_id = _dialogue_id(dialogue)
                order_key = hashlib.sha256(
                    f"{split_seed}:{group_id}".encode("ascii")
                ).hexdigest()
                cursor = connection.execute(
                    """
                    INSERT INTO dialogues
                        (group_id, source_split, priority, order_key, dialogue_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(group_id) DO UPDATE SET
                        source_split = excluded.source_split,
                        priority = excluded.priority,
                        order_key = excluded.order_key,
                        dialogue_json = excluded.dialogue_json
                    WHERE excluded.priority < dialogues.priority
                    """,
                    (
                        group_id,
                        split,
                        PRIORITY[split],
                        order_key,
                        json.dumps(dialogue, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
            raw_counts[split] = seen
            row_audits[split] = row_audit
        connection.commit()
        final_counts = dict(
            connection.execute(
                "SELECT source_split, COUNT(*) FROM dialogues GROUP BY source_split"
            ).fetchall()
        )
        return {
            "raw_dialogues_by_source_split": raw_counts,
            "source_row_audits": row_audits,
            "unique_dialogues_by_final_source_split": final_counts,
            "exact_duplicates_removed": sum(raw_counts.values())
            - sum(final_counts.values()),
        }
    finally:
        connection.close()


def prepare_lccc_association(
    raw_dir: Path,
    output_dir: Path,
    *,
    split_seed: int = 17,
    seed_dialogues: int = 32,
    feedback_dialogues: int = 128,
    confirm_research_terms: bool = False,
    confirm_privacy_review: bool = False,
) -> dict[str, Any]:
    if not confirm_research_terms:
        raise ValueError("review LCCC terms and privacy requirements, then confirm")
    if not confirm_privacy_review:
        raise ValueError("review LCCC privacy handling before preparing samples")
    if seed_dialogues <= 0 or feedback_dialogues <= 0:
        raise ValueError("seed-dialogues and feedback-dialogues must be positive")
    _verify_download(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("output directory must be empty")

    fd = tempfile.NamedTemporaryFile(
        prefix="aladdin-lccc-", suffix=".sqlite3", dir=output_dir.parent, delete=False
    )
    database_path = Path(fd.name)
    fd.close()
    try:
        unique_counts = _create_database(database_path, raw_dir, split_seed)
        connection = sqlite3.connect(database_path)
        try:
            train_rows = connection.execute(
                """
                SELECT group_id, dialogue_json FROM dialogues
                WHERE source_split = 'train'
                ORDER BY order_key, group_id
                LIMIT ?
                """,
                (seed_dialogues + feedback_dialogues,),
            ).fetchall()
            if len(train_rows) < seed_dialogues + feedback_dialogues:
                raise ValueError("not enough unique train dialogues for requested pools")
            seed_rows = train_rows[:seed_dialogues]
            feedback_rows = train_rows[seed_dialogues:]

            role_rows = {
                "seed": seed_rows,
                "feedback_pool": feedback_rows,
                "development": connection.execute(
                    """
                    SELECT group_id, dialogue_json FROM dialogues
                    WHERE source_split = 'development' ORDER BY order_key, group_id
                    """
                ).fetchall(),
                "final_test": connection.execute(
                    """
                    SELECT group_id, dialogue_json FROM dialogues
                    WHERE source_split = 'final_test' ORDER BY order_key, group_id
                    """
                ).fetchall(),
            }
        finally:
            connection.close()

        group_sets = {
            role: {row[0] for row in rows} for role, rows in role_rows.items()
        }
        roles = list(group_sets)
        overlaps = {
            f"{left}__{right}": len(group_sets[left] & group_sets[right])
            for index, left in enumerate(roles)
            for right in roles[index + 1 :]
        }
        if any(overlaps.values()):
            raise ValueError(f"dialogue group overlap detected: {overlaps}")

        counts: dict[str, Any] = {}
        skipped_short_targets = 0
        for role, source_rows in role_rows.items():
            def examples():
                nonlocal skipped_short_targets
                for group_id, raw_dialogue in source_rows:
                    dialogue = json.loads(raw_dialogue)
                    records, skipped = _make_examples(
                        dialogue, group_id, role
                    )
                    skipped_short_targets += skipped
                    for record in records:
                        if role == "feedback_pool":
                            record.pop("output")
                            record.pop("reference_continuation")
                        yield record

            if role == "feedback_pool":
                label_rows = (
                    {
                        "record_id": record["record_id"],
                        "reference_continuation": record["reference_continuation"],
                        "source_group_id": record["source_group_id"],
                    }
                    for group_id, raw_dialogue in source_rows
                    for record in _make_examples(
                        json.loads(raw_dialogue), group_id, role
                    )[0]
                )
                counts["feedback_reference_labels"], labels_hash = _write_jsonl(
                    output_dir / "labels" / "feedback_pool-reference.jsonl",
                    label_rows,
                )
            filename = {
                "seed": "seed.jsonl",
                "feedback_pool": "feedback_pool.jsonl",
                "development": "development.jsonl",
                "final_test": "final_test.jsonl",
            }[role]
            counts[role], content_hash = _write_jsonl(
                output_dir / filename, examples()
            )
            counts.setdefault("file_sha256", {})[filename] = content_hash
            if role == "feedback_pool":
                counts["file_sha256"]["labels/feedback_pool-reference.jsonl"] = labels_hash

        manifest = {
            "status": "prepared",
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "split_seed": split_seed,
            "task": "lccc-chinese-contextual-continuation-v1",
            "normalization": "NFKC; remove whitespace between Han/CJK-punctuation characters",
            "cut_ratios": list(CUT_RATIOS),
            "minimum_prefix_and_suffix_chars": MIN_FRAGMENT_CHARS,
            "context_turn_limit": 4,
            "requested_seed_dialogues": seed_dialogues,
            "requested_feedback_dialogues": feedback_dialogues,
            "dialogue_counts": unique_counts,
            "dialogues_by_task_role": {
                role: len(rows) for role, rows in role_rows.items()
            },
            "task_rows_by_role": {
                role: counts[role]
                for role in ("seed", "feedback_pool", "development", "final_test")
            },
            "feedback_reference_label_rows": counts["feedback_reference_labels"],
            "skipped_short_targets": skipped_short_targets,
            "group_overlap_counts": overlaps,
            "final_test_emitted": True,
            "terms_acknowledged": True,
            "privacy_review_confirmed_by_operator": True,
            "direct_identifier_filtering": {
                "status": "heuristic_dialogue_exclusion_applied",
                "patterns": sorted(DIRECT_IDENTIFIER_PATTERNS),
                "limitations": (
                    "heuristics do not detect all personal information; "
                    "manual review remains required"
                ),
            },
            "near_duplicate_audit": "not_performed",
            "files_sha256": counts["file_sha256"],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        database_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/lccc")
    parser.add_argument("--output-dir", default="data/processed/lccc-association")
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seed-dialogues", type=int, default=32)
    parser.add_argument("--feedback-dialogues", type=int, default=128)
    parser.add_argument("--confirm-research-terms", action="store_true")
    parser.add_argument("--confirm-privacy-review", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest = prepare_lccc_association(
        raw_dir,
        output_dir,
        split_seed=args.split_seed,
        seed_dialogues=args.seed_dialogues,
        feedback_dialogues=args.feedback_dialogues,
        confirm_research_terms=args.confirm_research_terms,
        confirm_privacy_review=args.confirm_privacy_review,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
