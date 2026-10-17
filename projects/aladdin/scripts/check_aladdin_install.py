from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


CORE_MODULES = (
    "core_system.mle_core.models",
    "core_system.mle_core.runtime",
    "local_experiments.aladdin.helpsteer2_feedback",
    "local_experiments.aladdin.requirement_state",
    "local_experiments.aladdin.integrated_pipeline.system",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-qwen", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    self_path = Path(__file__).resolve()
    forbidden = ("local_experiments.lego", "LEGO_TASK_MODEL", "LEGO_DETERMINISTIC")
    violations: list[str] = []
    for path in root.rglob("*"):
        if path.resolve() == self_path:
            continue
        if path.name == "audit_repository.py" and path.parent.name == "scripts":
            continue
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if any(part in {"data", "runs", "logs", "checkpoints", "models"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(root)}: {token}")

    modules = list(CORE_MODULES)
    if args.with_qwen:
        modules.append("local_experiments.aladdin.qwen_preference")
    for module in modules:
        importlib.import_module(module)

    if violations:
        raise SystemExit("forbidden cross-project references:\n" + "\n".join(violations))
    print(f"Aladdin install OK: {len(modules)} modules imported; no LEGO runtime references")


if __name__ == "__main__":
    main()
