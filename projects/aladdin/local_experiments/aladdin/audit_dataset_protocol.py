"""Audit a grouped preference JSONL before using it in an experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(path: Path) -> dict[str, object]:
    groups: dict[str, list[dict]] = defaultdict(list)
    rows = 0
    invalid = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid.append(line_number)
            continue
        rows += 1
        group_id = str(row.get("source_group_id", row.get("group_id", "")))
        if not group_id:
            invalid.append(line_number)
        groups[group_id].append(row)
    split_sets = {gid: {str(row.get("split", "")) for row in members}
                  for gid, members in groups.items()}
    cross_split = {gid: sorted(splits) for gid, splits in split_sets.items() if len(splits) > 1}
    conflicting = {
        gid: sorted({str(row.get("target", "")) for row in members})
        for gid, members in groups.items()
        if len({str(row.get("target", "")) for row in members}) > 2
    }
    return {
        "status": "error" if invalid or cross_split or conflicting else "ready",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": rows,
        "invalid_lines": invalid,
        "group_count": len(groups),
        "group_size_counts": dict(Counter(len(members) for members in groups.values())),
        "split_counts": dict(Counter(str(row.get("split", "")) for members in groups.values() for row in members)),
        "target_counts": dict(Counter(str(row.get("target", "")) for members in groups.values() for row in members)),
        "cross_split_groups": cross_split,
        "conflicting_target_groups": conflicting,
        "final_test_rows_loaded": sum(1 for members in groups.values() for row in members if row.get("split") == "final_test"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
