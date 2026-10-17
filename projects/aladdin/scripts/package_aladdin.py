from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "models", "runs", "logs", "checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pt", ".pth", ".safetensors", ".bin"}


def package_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name.endswith(".log"):
            continue
        paths.append(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Aladdin source repository.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = package_paths(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(f"wrote {args.output}")
    print(f"sha256 {digest}")
    print(f"files {len(paths)}")


if __name__ == "__main__":
    main()
