from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _bucket(seed: int, group_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def prepare(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    seed: int = 17,
    development_fraction: float = 0.3,
) -> dict[str, Any]:
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be in (0, 1)")
    rows: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {source_path}:{line_number}")
            rows.append(row)
    splits = {str(row.get("split")) for row in rows}
    if splits - {"seed", "development"}:
        raise ValueError(
            "source must contain only seed/development rows; "
            f"found {sorted(splits)}"
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "development":
            group_id = str(row.get("source_group_id") or row.get("record_id"))
            groups[group_id].append(row)
    if not groups:
        raise ValueError("source has no development groups to split")

    assigned: dict[str, str] = {}
    for group_id in sorted(groups):
        assigned[group_id] = (
            "development"
            if _bucket(seed, group_id) < development_fraction
            else "held_in"
        )
    if not {"development", "held_in"}.issubset(set(assigned.values())):
        ordered = sorted(assigned)
        assigned[ordered[0]] = "development"
        assigned[ordered[-1]] = "held_in"

    output_rows: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if row.get("split") == "development":
            group_id = str(row.get("source_group_id") or row.get("record_id"))
            row["split"] = assigned[group_id]
        output_rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")

    manifest = {
        "status": "ready",
        "source": str(source_path),
        "output": str(output_path),
        "split_seed": seed,
        "split_rule": "sha256(seed:source_group_id); complete source groups stay together",
        "development_fraction_by_group": development_fraction,
        "label_release_policy": "held_in target is hidden until offline feedback query",
        "record_counts": dict(Counter(str(row["split"]) for row in output_rows)),
        "group_counts": {
            split: sum(value == split for value in assigned.values())
            for split in ("development", "held_in")
        },
        "cross_split_source_groups": 0,
        "final_test_accessed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split normalized seed/development data for offline feedback replay."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--development-fraction", type=float, default=0.3)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                source_path=args.source,
                output_path=args.output,
                manifest_path=args.manifest,
                seed=args.seed,
                development_fraction=args.development_fraction,
            ),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
