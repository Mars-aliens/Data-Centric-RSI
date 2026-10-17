from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .real_data_runner import run_fixed_real_baseline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_projection(events_path: Path) -> dict[str, Any]:
    config: dict[str, Any] | None = None
    status = "unknown"
    metrics_by_split: dict[str, Any] = {}
    final_metrics: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []

    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            if event_type == "run_created":
                config = dict(payload["config"])
                config.pop("run_id", None)
            elif event_type == "evaluation_completed":
                metrics_by_split[str(payload["split"])] = payload["metrics"]
                costs.append(payload.get("cost", {}))
            elif event_type == "operation_completed":
                output = payload.get("output_artifact")
                if output:
                    artifacts.append(output)
                costs.append(payload.get("cost", {}))
            elif event_type == "training_completed":
                checkpoint = payload.get("checkpoint_artifact")
                if checkpoint:
                    artifacts.append(checkpoint)
                costs.append(payload.get("cost", {}))
            elif event_type == "run_completed":
                status = "completed"
                final_metrics = payload.get("metrics")
            elif event_type == "run_failed":
                status = "failed"

    if config is None:
        raise ValueError(f"Run event log has no run_created event: {events_path}")

    return {
        "config": config,
        "status": status,
        "metrics_by_split": metrics_by_split,
        "final_metrics": final_metrics,
        "artifacts": sorted(
            artifacts,
            key=lambda item: (item.get("kind", ""), item.get("version", ""), item.get("content_hash", "")),
        ),
        "costs": costs,
    }


def load_run_projection(run_root: Path, run_id: str) -> dict[str, Any]:
    return _event_projection(run_root / run_id / "events.jsonl")


def compare_run_projections(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    differences: list[str] = []
    for key in ("config", "status", "metrics_by_split", "final_metrics", "artifacts", "costs"):
        left_value = left.get(key)
        right_value = right.get(key)
        if key == "config":
            left_value = dict(left_value or {})
            right_value = dict(right_value or {})
            left_value.pop("run_id", None)
            right_value.pop("run_id", None)
        if left_value != right_value:
            differences.append(key)
    return differences


def build_freeze_manifest(
    *,
    dataset_path: Path,
    final_test_path: Path | None,
    protocol_path: Path,
    config_path: Path,
    requirements_path: Path,
    run_ids: list[str],
    runs_dir: Path,
    reproducible: bool,
) -> dict[str, Any]:
    files = [dataset_path, protocol_path, config_path, requirements_path]
    if final_test_path is not None:
        files.append(final_test_path)
    file_manifest = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    return {
        "manifest_version": "lego-baseline-freeze-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": "fixed-harness-react",
        "baseline_pipeline": "fixed-baseline-pipeline",
        "implementation_status": "deterministic-scaffold",
        "reproducible": reproducible,
        "python": sys.version,
        "platform": platform.platform(),
        "files": file_manifest,
        "runs": [load_run_projection(runs_dir, run_id) for run_id in run_ids],
    }


def _run_pair(args: argparse.Namespace) -> tuple[str, str]:
    run_ids = [f"{args.run_id_prefix}-a", f"{args.run_id_prefix}-b"]
    for run_id in run_ids:
        run_fixed_real_baseline(
            args.dataset,
            args.output_dir,
            run_id,
            final_test_path=args.final_test,
            report_final_test=args.report_final_test,
        )
    return run_ids[0], run_ids[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and freeze the fixed LEGO baseline.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--final-test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id-prefix", default="baseline-repro")
    parser.add_argument("--run-a")
    parser.add_argument("--run-b")
    parser.add_argument("--report-final-test", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.run_a) != bool(args.run_b):
        parser.error("--run-a and --run-b must be supplied together")
    if args.report_final_test and args.final_test is None:
        parser.error("--report-final-test requires --final-test")

    if args.run_a and args.run_b:
        run_ids = [args.run_a, args.run_b]
    else:
        run_ids = list(_run_pair(args))

    projections = [load_run_projection(args.output_dir, run_id) for run_id in run_ids]
    differences = compare_run_projections(projections[0], projections[1])
    manifest = build_freeze_manifest(
        dataset_path=args.dataset,
        final_test_path=args.final_test,
        protocol_path=Path(__file__).with_name("data_protocol.json"),
        config_path=Path(__file__).with_name("config.json"),
        requirements_path=Path("requirements-research.txt"),
        run_ids=run_ids,
        runs_dir=args.output_dir,
        reproducible=not differences,
    )
    manifest["comparison"] = {
        "run_a": run_ids[0],
        "run_b": run_ids[1],
        "differences": differences,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["comparison"], ensure_ascii=True, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
