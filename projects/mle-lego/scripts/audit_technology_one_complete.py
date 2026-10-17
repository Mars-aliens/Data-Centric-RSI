from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from local_experiments.lego.technical_one_protocol import (
    TECHNOLOGY_ONE_SEEDS,
    audit_complete_technology_one,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the complete Technology One evidence package."
    )
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--reflexion-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(TECHNOLOGY_ONE_SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_complete_technology_one(
        component_root=args.component_root,
        independent_root=args.independent_root,
        dependency_root=args.dependency_root,
        fixed_root=args.fixed_root,
        reflexion_root=args.reflexion_root,
        expected_seeds=args.seeds,
    )
    rendered = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
