from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adaptive import run_all_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LEGO Harness strategy comparisons.")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "runs" / "strategies"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    results = run_all_strategies(Path(args.output_dir), seed=args.seed)
    print(json.dumps({name: result.to_dict() for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()

