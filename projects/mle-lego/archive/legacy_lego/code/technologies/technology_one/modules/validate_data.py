from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_validation import validate_records
from .protocol import JsonlTaskAdapter, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a normalized LEGO JSONL dataset.")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("data_protocol.json"),
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    records = JsonlTaskAdapter(protocol).load(args.path)
    summary = validate_records(records, protocol)
    print(json.dumps(summary.to_dict(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

