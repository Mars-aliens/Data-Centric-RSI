"""Download pinned Aladdin datasets into the Git-ignored local data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "datasets.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _file_metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def download(dataset: str, data_root: Path, *, confirm_research_terms: bool = False) -> Path:
    registry = load_registry()
    if dataset not in registry:
        raise ValueError(f"unknown dataset: {dataset}")
    if dataset == "prism" and not confirm_research_terms:
        raise ValueError(
            "PRISM contains participant text and model responses under mixed terms. "
            "Review configs/datasets.json and pass --confirm-research-terms."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install the data extra first: python -m pip install -r requirements-data.txt") from exc

    spec = registry[dataset]
    target = data_root / dataset
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        allow_patterns=spec["files"],
        local_dir=target,
    )
    missing = [name for name in spec["files"] if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"download completed without expected files: {missing}")
    manifest = {
        "dataset": dataset,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "files": {name: _file_metadata(target / name) for name in spec["files"]},
        "terms_acknowledged": bool(confirm_research_terms) if dataset == "prism" else True,
    }
    (target / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["prism", "helpsteer2", "all"], required=True)
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="path relative to the project root unless absolute (default: data/raw)",
    )
    parser.add_argument(
        "--confirm-research-terms",
        action="store_true",
        help="confirm that PRISM terms, privacy, and non-redistribution rules were reviewed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    choices = ["prism", "helpsteer2"] if args.dataset == "all" else [args.dataset]
    for dataset in choices:
        path = download(dataset, data_root, confirm_research_terms=args.confirm_research_terms)
        spec = load_registry()[dataset]
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "path": str(path),
                    "revision": spec["revision"],
                    "files": spec["files"],
                    "status": "downloaded",
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
