from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import build_adapter
from .protocol import load_configs, load_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate server baseline handoff configs without running models.")
    parser.add_argument("--specs", default=str(Path(__file__).with_name("baseline-specs.json")))
    parser.add_argument("--configs", default=str(Path(__file__).with_name("server-configs.local.json")))
    args = parser.parse_args()
    specs = load_specs(Path(args.specs))
    plans = []
    for config in load_configs(Path(args.configs)):
        spec = specs[config.baseline_key]
        plans.append(build_adapter(spec).dry_run(config))
    print(json.dumps(plans, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
