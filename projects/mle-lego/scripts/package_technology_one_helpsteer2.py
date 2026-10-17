from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def selected_files(root: Path) -> list[Path]:
    core = root / "core_system"
    paths = [
        root / "requirements-integrated-aladdin.txt",
        root / "local_experiments" / "__init__.py",
        root / "local_experiments" / "aladdin" / "__init__.py",
        root / "local_experiments" / "aladdin" / "active_feedback_helpsteer2.py",
        root / "local_experiments" / "aladdin" / "active_feedback_helpsteer2_multiround.py",
        root / "local_experiments" / "aladdin" / "analyze_active_feedback_helpsteer2.py",
        root / "local_experiments" / "aladdin" / "helpsteer2_feedback.py",
        root / "local_experiments" / "aladdin" / "helpsteer2-technique-1-server-handoff.md",
        root / "local_experiments" / "tests" / "__init__.py",
        root / "local_experiments" / "tests" / "test_active_feedback_helpsteer2.py",
        root / "local_experiments" / "tests" / "test_active_feedback_helpsteer2_multiround.py",
        root / "local_experiments" / "tests" / "test_analyze_active_feedback_helpsteer2.py",
        core / "README.md",
    ]
    paths.extend(
        path
        for path in core.rglob("*.py")
        if not {"__pycache__", "tests"}.intersection(path.parts)
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package files: {missing}")
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the standalone HelpSteer2 technique-one experiment."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = selected_files(root)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"wrote {output}")
    print(f"sha256 {digest}")
    print(f"files {len(paths)}")


if __name__ == "__main__":
    main()
