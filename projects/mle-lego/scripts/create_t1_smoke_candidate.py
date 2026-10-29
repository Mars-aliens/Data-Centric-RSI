from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from technologies.technology_one.candidate_proposal import (  # noqa: E402
    write_candidate_bundle,
)


_STRATEGY_FILE = "candidate_strategy.json"
_STRATEGY_CONTENT = """{
  "schema": "lego-t1-data-strategy-v1",
  "data_policy": {
    "min_input_chars": 1,
    "deduplicate_inputs": true,
    "max_feedback_examples": null
  }
}
"""


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
    if not (source / "candidate_strategy.json").is_file():
        parser.error("source-dir must contain candidate_strategy.json")
    manifest = write_candidate_bundle(
        base_dir=source,
        output_dir=args.output_dir,
        proposal={
            "rationale": "Exercise the candidate data-strategy branch using the default policy.",
            "edits": [
                {
                    "path": _STRATEGY_FILE,
                    "operation": "write",
                    "content": _STRATEGY_CONTENT,
                }
            ],
        },
        candidate_id=args.candidate_id,
        base_version_id=args.base_version_id,
        decision_model={
            "provider": "local_smoke_fixture",
            "model_id": "deterministic-default-data-strategy",
            "revision": "1",
        },
        prompt_hashes={"fixture_sha256": hashlib.sha256(_STRATEGY_CONTENT.encode("utf-8")).hexdigest()},
        proposal_cost={"llm_calls": 0, "llm_tokens": 0},
    )
    print(f"candidate_id: {manifest['candidate_id']}")
    print(f"candidate_source_sha256: {manifest['candidate_source_sha256']}")
    print(f"output_dir: {args.output_dir.resolve()}")
    print("research_candidate: false")
    print("strategy: default data policy; trusted runner; no arbitrary code execution")


if __name__ == "__main__":
    main()
