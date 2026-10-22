from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from technologies.technology_one.candidate_proposal import (  # noqa: E402
    write_candidate_bundle,
)


_SMOKE_FILE = "technologies/technology_one/long_horizon_candidate_qwen_command.py"
_SMOKE_MARKER = "\n# Deterministic T1 infrastructure smoke fixture; no behavior change.\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a no-behavior-change candidate for T1 infrastructure smoke only."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-id", default="t1-infrastructure-smoke")
    parser.add_argument("--base-version-id", default="runtime-stable")
    args = parser.parse_args()

    source = args.source_dir.resolve(strict=True)
    source_file = source / _SMOKE_FILE
    original = source_file.read_text(encoding="utf-8")
    if _SMOKE_MARKER.strip() in original:
        parser.error("source already contains the smoke marker")
    manifest = write_candidate_bundle(
        base_dir=source,
        output_dir=args.output_dir,
        proposal={
            "rationale": "Exercise the isolated candidate runtime without changing behavior.",
            "edits": [
                {
                    "path": _SMOKE_FILE,
                    "operation": "write",
                    "content": original.rstrip("\n") + _SMOKE_MARKER,
                }
            ],
        },
        candidate_id=args.candidate_id,
        base_version_id=args.base_version_id,
        decision_model={
            "provider": "local_smoke_fixture",
            "model_id": "deterministic-no-behavior-change",
            "revision": "1",
        },
        prompt_hashes={
            "fixture_sha256": hashlib.sha256(_SMOKE_MARKER.encode("utf-8")).hexdigest()
        },
        proposal_cost={"llm_calls": 0, "llm_tokens": 0},
    )
    print(f"candidate_id: {manifest['candidate_id']}")
    print(f"candidate_source_sha256: {manifest['candidate_source_sha256']}")
    print(f"output_dir: {args.output_dir.resolve()}")
    print("research_candidate: false")


if __name__ == "__main__":
    main()
