from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from local_experiments.aladdin.hh_rlhf_feedback import DATASET_ID, DATASET_REVISION


def prepare_split(split: str, output: Path) -> dict[str, Any]:
    """Materialize one official HF split without changing its chosen/rejected labels."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the Hugging Face datasets package on the server first") from exc
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=split, streaming=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        iterator = tqdm(dataset, desc=f"download {split}") if tqdm else dataset
        for row in iterator:
            if set(row) != {"chosen", "rejected"} and not {"chosen", "rejected"}.issubset(row):
                raise ValueError(f"unexpected HH-RLHF fields in {split}: {sorted(row)}")
            handle.write(json.dumps({"chosen": row["chosen"], "rejected": row["rejected"]}, ensure_ascii=False) + "\n")
            count += 1
    return {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": split,
        "output": str(output),
        "records": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a pinned Anthropic HH-RLHF split as JSONL.")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_split(args.split, args.output)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
