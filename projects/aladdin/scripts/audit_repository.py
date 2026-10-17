from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    self_path = Path(__file__).resolve()
    forbidden_names = {"data", "models", "checkpoints", "runs", "logs", "_archive", ".venv"}
    forbidden_tokens = ("local_experiments.lego", "LEGO_TASK_MODEL", "LEGO_DETERMINISTIC")
    failures: list[str] = []

    for path in root.rglob("*"):
        if path.resolve() == self_path:
            continue
        if path.name in {"audit_repository.py", "check_aladdin_install.py"} and path.parent.name == "scripts":
            continue
        if path.is_dir() and path.name in forbidden_names:
            failures.append(f"forbidden directory: {path.relative_to(root)}")
            continue
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in {".pyc", ".pt", ".pth", ".safetensors", ".bin"}:
            failures.append(f"forbidden artifact: {path.relative_to(root)}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden_tokens:
            if token in text:
                failures.append(f"forbidden token {token}: {path.relative_to(root)}")

    if failures:
        raise SystemExit("repository audit failed:\n" + "\n".join(failures))
    print("repository audit OK")


if __name__ == "__main__":
    main()
