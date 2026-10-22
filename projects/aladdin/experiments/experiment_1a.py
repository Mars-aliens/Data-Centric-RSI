"""Run the complete static synthesis comparison (experiment 1A)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.evaluation.qwen_sft import (
    QwenSFTConfig,
    build_1a_training_records,
    train_and_evaluate,
)
from experiments.synthesis.generator import OpenAICompatibleGenerator
from experiments.synthesis.strategies import normalize_seed, synthesize

PRISM_REPOSITORY = "HannahRoseKirk/prism-alignment"
PRISM_REVISION = "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_number}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_seed_groups(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if any(row.get("split") not in (None, "seed") for row in records):
        raise ValueError("seed input contains records outside the seed split")
    normalized = [normalize_seed(row, index) for index, row in enumerate(records)]
    groups = sorted({row["source_group_id"] for row in normalized})
    if limit <= 0:
        raise ValueError("seed_group_limit must be positive")
    if limit > len(groups):
        raise ValueError(f"requested {limit} seed groups but only {len(groups)} are available")
    rng = random.Random(seed)
    rng.shuffle(groups)
    selected = set(groups[:limit])
    return [row for row in normalized if row["source_group_id"] in selected]


def select_generation_parents(
    seeds: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in seeds:
        groups.setdefault(row["source_group_id"], []).append(row)
    rng = random.Random(seed)
    parents = []
    for group in sorted(groups):
        rows = groups[group]
        parents.append(rows[rng.randrange(len(rows))])
    return parents


def _load_development(data_path: Path, labels_path: Path | None) -> list[dict[str, Any]]:
    if "final_test" in data_path.name.lower() or "final-test" in str(data_path).lower():
        raise ValueError("experiment 1A cannot read final_test data")
    if labels_path is None:
        raise ValueError("experiment 1A requires the development label sidecar")
    records = _jsonl(data_path)
    if any(row.get("split") == "final_test" for row in records):
        raise ValueError("development data contains final_test records")
    if any(row.get("split") != "development" for row in records):
        raise ValueError("experiment 1A accepts only records marked split=development")
    record_ids = [str(row.get("record_id") or "") for row in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("development records must have unique, non-empty record_id values")
    if "final_test" in labels_path.name.lower() or "final-test" in str(labels_path).lower():
        raise ValueError("experiment 1A cannot read final_test labels")
    label_rows = _jsonl(labels_path)
    labels = {str(row["record_id"]): row for row in label_rows if row.get("record_id")}
    if len(labels) != len(label_rows):
        raise ValueError("development label sidecar has missing or duplicate record_id values")
    missing = [str(record.get("record_id")) for record in records if str(record.get("record_id")) not in labels]
    if missing:
        raise ValueError(f"development labels are missing for {len(missing)} records")
    extra = set(labels) - set(record_ids)
    if extra:
        raise ValueError(f"development label sidecar contains {len(extra)} records outside development data")
    for record in records:
        label = labels.get(str(record.get("record_id")))
        if not isinstance(label.get("if_chosen"), bool):
            raise ValueError(f"invalid if_chosen label for record {record.get('record_id')}")
        record["if_chosen"] = label["if_chosen"]
        record["score"] = label.get("score")
    return records


def select_development_groups(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    groups = sorted({str(row.get("source_group_id") or "") for row in records})
    groups = [group for group in groups if group]
    if limit <= 0 or limit >= len(groups):
        return records
    rng = random.Random(seed)
    rng.shuffle(groups)
    selected = set(groups[:limit])
    return [row for row in records if str(row.get("source_group_id") or "") in selected]


def _run_synthesis(
    seeds: list[dict[str, Any]],
    strategy: str,
    args: argparse.Namespace,
    output: Path,
    run_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    generator = None
    if strategy != "seed-only":
        if not args.endpoint or not args.model:
            raise ValueError("generated strategies require --endpoint and --model")
        generator = OpenAICompatibleGenerator(
            args.endpoint,
            args.model,
            api_key=os.environ.get(args.api_key_env),
            timeout_seconds=args.timeout_seconds,
        )
    if strategy == "seed-only":
        rows = synthesize(seeds, strategy=strategy, count=0, seed_value=run_seed)
    else:
        generation_parents = select_generation_parents(seeds, run_seed)
        rows = []
        while len(rows) < args.max_candidates_per_strategy:
            if generator.usage.reported and (
                generator.usage.prompt_tokens + generator.usage.completion_tokens
            ) >= args.generation_token_budget:
                break
            if generator.usage.requests and generator.usage.complete_usage_requests != generator.usage.requests:
                raise RuntimeError(
                    "formal token-budget comparison requires token usage on every API response"
                )
            next_row = synthesize(
                generation_parents,
                strategy=strategy,
                count=1,
                generator=generator,
                seed_value=run_seed + len(rows),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                index_offset=len(rows),
            )
            rows.extend(next_row)
            if generator.usage.requests != generator.usage.complete_usage_requests:
                raise RuntimeError(
                    "endpoint omitted token usage; cannot verify the shared generation-token budget"
                )
        if not rows:
            raise RuntimeError(f"{strategy} produced no candidate within the generation budget")
        used_tokens = generator.usage.prompt_tokens + generator.usage.completion_tokens
        if used_tokens < args.generation_token_budget:
            raise RuntimeError(
                f"candidate safety cap reached at {used_tokens} of "
                f"{args.generation_token_budget} generation tokens; increase --max-candidates-per-strategy"
            )
    _write_jsonl(output, rows)
    usage = getattr(generator, "usage", None)
    manifest = {
        "status": "completed",
        "strategy": strategy,
        "candidate_count": 0 if strategy == "seed-only" else len(rows),
        "seed_records_emitted": len(rows) if strategy == "seed-only" else 0,
        "unique_input_output_pairs": len(
            {(str(row.get("input", "")).strip(), str(row.get("output", "")).strip()) for row in rows}
        ),
        "source_groups_covered": len({str(row.get("source_group_id", "")) for row in rows}),
        "generation_calls": sum(int(row.get("generation_calls", 0)) for row in rows),
        "requests": getattr(usage, "requests", 0),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reported_usage": getattr(usage, "reported", False),
        "reported_requests": getattr(usage, "reported_requests", 0),
        "complete_usage_requests": getattr(usage, "complete_usage_requests", 0),
        "generation_token_budget": args.generation_token_budget if generator else 0,
        "generation_tokens_used": (
            getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
            if usage
            else 0
        ),
        "generation_budget_overshoot_tokens": max(
            0,
            getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
            - args.generation_token_budget,
        ) if generator else 0,
        "model": args.model if generator else None,
        "model_revision": MODEL_REVISION if generator else None,
        "endpoint": args.endpoint if generator else None,
        "novel_input_count": len(
            {
                " ".join(str(row.get("input", "")).split()).casefold()
                for row in rows
                if row.get("input")
            }
            - {
                " ".join(str(row.get("input", "")).split()).casefold()
                for row in seeds
                if row.get("input")
            }
        ) if generator else 0,
        "novel_input_rate": (
            len(
                {
                    " ".join(str(row.get("input", "")).split()).casefold()
                    for row in rows
                    if row.get("input")
                }
                - {
                    " ".join(str(row.get("input", "")).split()).casefold()
                    for row in seeds
                    if row.get("input")
                }
            ) / len(rows)
        ) if generator and rows else 0.0,
        "generation_seconds": time.monotonic() - started,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return rows, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    seed_path = Path(args.seed_data)
    development_path = Path(args.development_data)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "result.json").exists():
        raise FileExistsError("output already contains result.json; use a new run directory")
    unexpected = [
        path.name
        for path in output_root.iterdir()
        if path.name != "run.log"
    ]
    if unexpected:
        raise FileExistsError(
            f"output directory is not fresh; unexpected entries: {sorted(unexpected)}"
        )
    if args.generation_token_budget <= 0 or args.max_candidates_per_strategy <= 0:
        raise ValueError("generation_token_budget and max_candidates_per_strategy must be positive")
    if args.train_token_budget <= 0 or args.max_steps <= 0 or args.train_records <= 0:
        raise ValueError("training token budget, max steps, and train records must be positive")
    if args.development_group_limit <= 0:
        raise ValueError("development_group_limit must be positive")
    if not 0 < args.synthetic_train_records < args.train_records:
        raise ValueError("synthetic-train-records must be positive and less than train-records")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("use at least two unique seeds for paired strategy comparisons")
    if len(args.strategies) < 2 or len(set(args.strategies)) != len(args.strategies):
        raise ValueError("formal experiment 1A requires at least two unique strategies")
    if "seed-only" not in args.strategies:
        raise ValueError("formal experiment 1A requires the seed-only baseline")
    unknown = set(args.strategies) - {"seed-only", "fixed-prompt", "self-instruct", "evol-instruct"}
    if unknown:
        raise ValueError(f"unknown synthesis strategies: {sorted(unknown)}")
    if "evol-instruct" in args.strategies and not args.evol_task_compatible:
        raise ValueError("Evol-Instruct requires explicit task compatibility confirmation")
    development = _load_development(
        development_path,
        Path(args.development_labels) if args.development_labels else None,
    )
    all_development_count = len(development)
    development = select_development_groups(
        development, args.development_group_limit, args.development_selection_seed
    )
    seed_records = _jsonl(seed_path)
    if any(row.get("split") != "seed" for row in seed_records):
        raise ValueError("seed data contains records outside the seed split")
    split_manifest_path = seed_path.parent / "manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError("prepared PRISM manifest.json is required beside seed.jsonl")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("status") != "prepared":
        raise ValueError("prepared dataset manifest is not marked status=prepared")
    if split_manifest.get("dataset") != PRISM_REPOSITORY:
        raise ValueError("experiment 1A expects the pinned PRISM dataset")
    if split_manifest.get("revision") != PRISM_REVISION:
        raise ValueError("prepared PRISM data does not match the pinned dataset revision")
    seed_groups = {str(row.get("source_group_id") or "") for row in seed_records}
    development_groups = {str(row.get("source_group_id") or "") for row in development}
    overlap = seed_groups & development_groups
    if overlap:
        raise ValueError(f"seed/development source-group leakage detected: {len(overlap)} groups")
    results = []
    result_path = output_root / "result.json"
    run_id = getattr(args, "run_id", None) or uuid.uuid4().hex
    config_snapshot = {key: value for key, value in vars(args).items() if key != "run_id"}
    run_document = {
        "status": "running",
        "run_id": run_id,
        "protocol": "aladdin-experiment-1a-v1",
        "configuration": config_snapshot,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "runner_sha256": _sha256(Path(__file__)),
            "training_module_sha256": _sha256(Path(__file__).parent / "evaluation" / "qwen_sft.py"),
            "synthesis_module_sha256": _sha256(Path(__file__).parent / "synthesis" / "strategies.py"),
        },
        "results": results,
    }
    _write_json(result_path, run_document)
    for run_seed in args.seeds:
        run_seeds = select_seed_groups(seed_records, args.seed_group_limit, run_seed)
        ordered_strategies = list(args.strategies)
        random.Random(run_seed).shuffle(ordered_strategies)
        for strategy in ordered_strategies:
            strategy_dir = output_root / strategy / f"seed-{run_seed}"
            strategy_dir.mkdir(parents=True, exist_ok=True)
            candidates, generation = _run_synthesis(
                run_seeds, strategy, args, strategy_dir / "candidates.jsonl", run_seed
            )
            train_records = build_1a_training_records(
                run_seeds,
                candidates,
                strategy=strategy,
                train_records=args.train_records,
                synthetic_train_records=args.synthetic_train_records,
                seed=run_seed,
            )
            _write_jsonl(strategy_dir / "train.jsonl", train_records)
            adapter_path = strategy_dir / "adapter" if args.save_adapters else None
            config = QwenSFTConfig(
                base_model=args.base_model,
                model_revision=args.model_revision,
                train_token_budget=args.train_token_budget,
                require_cuda=not args.allow_cpu_training,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                eval_batch_size=args.eval_batch_size,
                seed=run_seed,
                save_adapter=str(adapter_path) if adapter_path else None,
            )
            metrics = train_and_evaluate(train_records, development, config)
            if not metrics.get("pairwise_comparisons"):
                raise RuntimeError(
                    "development set has no chosen-vs-rejected pairs; check its labels and grouping"
                )
            source_counts = {}
            for row in train_records:
                source = row.get("synthesis_strategy", "seed")
                source_counts[source] = source_counts.get(source, 0) + 1
            results.append({
                "strategy": strategy,
                "seed": run_seed,
                "selected_seed_group_count": len(
                    {row["source_group_id"] for row in run_seeds}
                ),
                "selected_seed_groups_sha256": hashlib.sha256(
                    "\n".join(sorted({row["source_group_id"] for row in run_seeds})).encode("utf-8")
                ).hexdigest(),
                "generation": generation,
                "training_source_counts": source_counts,
                "training_unique_record_ids": len(
                    {str(row.get("record_id") or "") for row in train_records}
                ),
                "training_repeated_rows": len(train_records)
                - len({str(row.get("record_id") or "") for row in train_records}),
                "synthetic_training_slots": 0
                if strategy == "seed-only"
                else args.synthetic_train_records,
                "seed_group_count": len({row["source_group_id"] for row in run_seeds}),
                "seed_row_count": len(run_seeds),
                "training": metrics,
                "candidate_path": str(strategy_dir / "candidates.jsonl"),
                "training_path": str(strategy_dir / "train.jsonl"),
            })
            run_document["results"] = results
            _write_json(result_path, run_document)
            print(json.dumps({"strategy": strategy, "seed": run_seed, "metrics": metrics}, ensure_ascii=False))

    expected_results = len(args.seeds) * len(args.strategies)
    if len(results) != expected_results:
        raise RuntimeError(f"incomplete matrix: got {len(results)} of {expected_results} runs")
    summary = {
        "status": "completed",
        "run_id": run_id,
        "protocol": "aladdin-experiment-1a-v1",
        "data": {
            "seed_path": seed_path.name,
            "seed_sha256": _sha256(seed_path),
            "split_manifest_sha256": _sha256(split_manifest_path),
            "dataset": split_manifest.get("dataset"),
            "dataset_revision": split_manifest.get("revision"),
            "dataset_split_seed": split_manifest.get("split_seed"),
            "seed_groups_requested_per_run": args.seed_group_limit,
            "seed_rows_available": len(seed_records),
            "development_path": development_path.name,
            "development_sha256": _sha256(development_path),
            "development_labels_sha256": _sha256(Path(args.development_labels))
            if args.development_labels
            else None,
            "development_rows": len(development),
            "development_rows_available": all_development_count,
            "development_groups_selected": len(
                {str(row.get("source_group_id") or "") for row in development}
            ),
            "development_groups_sha256": hashlib.sha256(
                "\n".join(
                    sorted({str(row.get("source_group_id") or "") for row in development})
                ).encode("utf-8")
            ).hexdigest(),
            "development_group_limit": args.development_group_limit,
            "development_selection_seed": args.development_selection_seed,
            "final_test_accessed": False,
        },
        "configuration": config_snapshot,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "runner_sha256": _sha256(Path(__file__)),
            "training_module_sha256": _sha256(Path(__file__).parent / "evaluation" / "qwen_sft.py"),
            "synthesis_module_sha256": _sha256(Path(__file__).parent / "synthesis" / "strategies.py"),
        },
        "results": results,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["data"]["seed_groups_selected_by_run_seed"] = {
        str(seed): len(
            {
                row["source_group_id"]
                for row in select_seed_groups(seed_records, args.seed_group_limit, seed)
            }
        )
        for seed in args.seeds
    }
    _write_json(result_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-data", required=True)
    parser.add_argument("--development-data", required=True)
    parser.add_argument("--development-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--strategies", nargs="+", default=["seed-only", "fixed-prompt", "self-instruct"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 29])
    parser.add_argument("--evol-task-compatible", action="store_true")
    parser.add_argument("--seed-group-limit", type=int, default=32)
    parser.add_argument("--development-group-limit", type=int, default=64)
    parser.add_argument("--development-selection-seed", type=int, default=101)
    parser.add_argument("--max-candidates-per-strategy", type=int, default=128)
    parser.add_argument("--generation-token-budget", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--train-token-budget", type=int, default=100_000)
    parser.add_argument("--allow-cpu-training", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-records", type=int, default=256)
    parser.add_argument("--synthetic-train-records", type=int, default=128)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    invalid = set(args.strategies) - {"seed-only", "fixed-prompt", "self-instruct", "evol-instruct"}
    if invalid:
        raise ValueError(f"unknown synthesis strategies: {sorted(invalid)}")
    if len(args.strategies) < 2 or len(set(args.strategies)) != len(args.strategies):
        raise ValueError("formal experiment 1A requires at least two unique strategies")
    if "seed-only" not in args.strategies:
        raise ValueError("formal experiment 1A requires the seed-only baseline")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("use at least two unique seeds for paired strategy comparisons")
    if "evol-instruct" in args.strategies and not args.evol_task_compatible:
        raise ValueError("Evol-Instruct requires --evol-task-compatible after task compatibility review")
    if args.model != MODEL_ID:
        raise ValueError(f"experiment 1A generation model is frozen to {MODEL_ID}")
    base_model = str(args.base_model)
    if base_model != MODEL_ID and MODEL_REVISION not in base_model:
        raise ValueError(
            "experiment 1A task model must be the frozen Hugging Face ID or a "
            f"local snapshot containing revision {MODEL_REVISION}"
        )
    if args.model_revision != MODEL_REVISION:
        raise ValueError(f"experiment 1A model revision is frozen to {MODEL_REVISION}")
    if not 0 < args.synthetic_train_records < args.train_records:
        raise ValueError("synthetic-train-records must be positive and less than train-records")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.run_id = uuid.uuid4().hex
    try:
        run(args)
    except Exception as exc:
        result_path = Path(args.output) / "result.json"
        if result_path.is_file():
            try:
                current = json.loads(result_path.read_text(encoding="utf-8"))
                if current.get("run_id") == args.run_id and current.get("status") == "running":
                    current["status"] = "failed"
                    current["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    _write_json(result_path, current)
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        raise


if __name__ == "__main__":
    main()
