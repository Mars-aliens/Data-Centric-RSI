from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    item for item in (str(CODE_ROOT), existing_pythonpath) if item
)

from technologies.technology_one.long_horizon_runner import LongHorizonQwenRunner  # noqa: E402
from technologies.technology_one.long_horizon_task import (  # noqa: E402
    LongHorizonTaskConfig,
    prepare_helpsteer2_long_horizon_task,
)
from technologies.technology_one.server_adapters import JsonCommand  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed, feedback-delayed HelpSteer2 Qwen stream baseline."
    )
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stream-seed", type=int, default=17)
    parser.add_argument("--groups-per-batch", type=int, default=50)
    parser.add_argument("--feedback-delay", choices=(0, 2, 5), type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=_positive_int,
        help="schedule only the first N deterministic development batches (smoke runs)",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=8)
    parser.add_argument("--command-timeout-seconds", type=float, default=3600)
    args = parser.parse_args()

    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("--config-json must contain a JSON object")
    if not isinstance(config.get("base_model_path"), str) or not config["base_model_path"].strip():
        parser.error("config JSON must set base_model_path to a local Qwen model directory")
    base_revision = config.get("base_model_revision")
    if (
        not isinstance(base_revision, str)
        or not base_revision.strip()
        or base_revision.strip().lower() in {"latest", "main", "master", "unrecorded"}
        or (base_revision.strip().startswith("<") and base_revision.strip().endswith(">"))
    ):
        parser.error("config JSON must set base_model_revision to an immutable model revision")

    task = prepare_helpsteer2_long_horizon_task(
        _read_jsonl(args.train_jsonl),
        _read_jsonl(args.validation_jsonl),
        config=LongHorizonTaskConfig(
            stream_seed=args.stream_seed,
            groups_per_batch=args.groups_per_batch,
            feedback_delay_batches=args.feedback_delay,
            max_batches=args.max_batches,
        ),
    )
    command = JsonCommand.from_value(
        (
            sys.executable,
            "-m",
            "technologies.technology_one.long_horizon_qwen_command",
        ),
        timeout_seconds=args.command_timeout_seconds,
        cwd=str(PROJECT_ROOT),
    )
    completed = 0

    def report_batch(batch: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        print(
            f"completed {completed}/{task.scheduler.batch_count}: "
            f"{batch['batch_id']} model={batch['model_version_id']} "
            f"released_feedback={len(batch['released_feedback_ids'])}",
            file=sys.stderr,
            flush=True,
        )

    runner = LongHorizonQwenRunner(
        task,
        command=command,
        config=config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        seed=args.seed,
        prediction_batch_size=args.prediction_batch_size,
        on_batch=report_batch,
    )
    completed = int(runner.state["completed_batches"])
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
