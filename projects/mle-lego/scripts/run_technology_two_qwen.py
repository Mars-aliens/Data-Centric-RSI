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
os.environ["PYTHONPATH"] = os.pathsep.join(
    item for item in (str(CODE_ROOT), os.environ.get("PYTHONPATH", "")) if item
)

from technologies.technology_one.long_horizon_runner import LongHorizonQwenRunner  # noqa: E402
from technologies.technology_one.long_horizon_task import (  # noqa: E402
    LongHorizonTaskConfig,
    STREAM_REGIMES,
    prepare_helpsteer2_long_horizon_task,
)
from technologies.technology_one.server_adapters import JsonCommand  # noqa: E402
from technologies.technology_two.experiment_policy import ARMS, ExperienceTrainingPolicy  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one controlled T2 experience-lifecycle arm on HelpSteer2."
    )
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--task-config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stream-seed", type=int, default=17)
    parser.add_argument("--stream-regime", choices=STREAM_REGIMES, default="randomized")
    parser.add_argument("--groups-per-batch", type=int, default=50)
    parser.add_argument("--feedback-delay", choices=(0, 2, 5), type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--prediction-batch-size", type=int, default=8)
    parser.add_argument("--command-timeout-seconds", type=float, default=3600)
    args = parser.parse_args()

    config = json.loads(args.task_config_json.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("task config must contain a JSON object")
    model_path = config.get("base_model_path")
    revision = config.get("base_model_revision")
    if not isinstance(model_path, str) or not Path(model_path).is_dir():
        parser.error("task config must point to an existing local base_model_path")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision.strip().lower() in {"latest", "main", "master", "unrecorded"}
        or (revision.startswith("<") and revision.endswith(">"))
    ):
        parser.error("task config must pin an immutable base_model_revision")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    task = prepare_helpsteer2_long_horizon_task(
        _rows(args.train_jsonl),
        _rows(args.validation_jsonl),
        config=LongHorizonTaskConfig(
            stream_seed=args.stream_seed,
            stream_regime=args.stream_regime,
            groups_per_batch=args.groups_per_batch,
            feedback_delay_batches=args.feedback_delay,
            max_batches=args.max_batches,
        ),
    )
    command = JsonCommand.from_value(
        (sys.executable, "-m", "technologies.technology_one.long_horizon_qwen_command"),
        timeout_seconds=args.command_timeout_seconds,
        cwd=str(PROJECT_ROOT),
    )

    def report(batch: dict[str, Any]) -> None:
        audit = batch.get("training_data_policy", {})
        print(
            f"completed {batch['batch_index'] + 1}/{task.scheduler.batch_count}: "
            f"released={len(batch['released_feedback_ids'])} "
            f"experience={audit.get('experience_status', 'not-used')} "
            f"selected={audit.get('records_selected', 0)}",
            file=sys.stderr,
            flush=True,
        )

    runner = LongHorizonQwenRunner(
        task,
        command=command,
        config={**config, "seed": args.seed, "prediction_batch_size": args.prediction_batch_size},
        output_dir=args.output_dir,
        run_id=args.run_id,
        seed=args.seed,
        prediction_batch_size=args.prediction_batch_size,
        on_batch=report,
        training_data_policy=ExperienceTrainingPolicy(args.arm),
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
