from __future__ import annotations

import argparse
import json
from pathlib import Path

from integration.experiment_matrix import all_feature_configs


def summarize(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    runs: list[dict[str, object]] = []
    missing: list[str] = []
    for config in all_feature_configs():
        path = root_path / config.name / "summary.json"
        if not path.is_file():
            missing.append(config.name)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("features", {}).get("name") != config.name:
            raise ValueError(f"feature configuration mismatch: {path}")
        runs.append(value)
    if missing:
        raise ValueError(f"missing integration configurations: {missing}")
    completed = sum(item.get("status") == "complete" for item in runs)
    return {
        "status": "complete" if completed == len(runs) else "incomplete",
        "expected_config_count": len(all_feature_configs()),
        "completed_config_count": completed,
        "failed_config_count": len(runs) - completed,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the 2x2x2 integration matrix.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = summarize(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status: {value['status']}")
    print(f"completed: {value['completed_config_count']} / {value['expected_config_count']}")


if __name__ == "__main__":
    main()
