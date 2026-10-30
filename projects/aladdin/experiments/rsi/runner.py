"""Run the feedback-driven, multi-round Aladdin system comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from experiments.cleaning.rules import CleaningPolicy, clean_candidates
from experiments.evaluation.qwen_sft import (
    QwenSFTConfig,
    build_training_schedule,
    train_and_evaluate,
)
from experiments.feedback.annotation import (
    ISSUES,
    JUDGMENTS,
    collect_interactive_feedback,
    read_feedback,
)
from experiments.feedback.selection import (
    PreferenceUncertaintyModel,
    select_queries,
    split_development_groups,
)
from experiments.feedback.model_feedback import (
    MODEL_ID as FEEDBACK_MODEL_ID,
    MODEL_REVISION as FEEDBACK_MODEL_REVISION,
    PROMPT_VERSION as FEEDBACK_PROMPT_VERSION,
    collect_model_feedback,
)
from experiments.rsi.policies import initial_policy_state, update_policy_state
from experiments.synthesis.generator import OpenAICompatibleGenerator
from experiments.synthesis.strategies import synthesize


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PRISM_REPOSITORY = "HannahRoseKirk/prism-alignment"
PRISM_REVISION = "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6"


@dataclass(frozen=True)
class Arm:
    name: str
    selector: str
    update_synthesis: bool
    update_cleaning: bool
    update_selection: bool


ARMS = {
    "fixed_policy": Arm("fixed_policy", "feedback-value", False, False, False),
    "recursive_policy": Arm("recursive_policy", "feedback-value", True, True, True),
    "freeze_synthesis": Arm("freeze_synthesis", "feedback-value", False, True, True),
    "freeze_cleaning": Arm("freeze_cleaning", "feedback-value", True, False, True),
    "freeze_selection": Arm("freeze_selection", "feedback-value", True, True, False),
    "query_random": Arm("query_random", "random", True, True, False),
    "query_uncertainty": Arm("query_uncertainty", "uncertainty", True, True, False),
    "query_feedback_value": Arm("query_feedback_value", "feedback-value", True, True, False),
}

COMPARISONS = {
    "overall": ("fixed_policy", "recursive_policy"),
    "ablation": ("recursive_policy", "freeze_synthesis", "freeze_cleaning", "freeze_selection"),
    "selection": ("query_random", "query_uncertainty", "query_feedback_value"),
    "all": tuple(ARMS),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_values(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _safe_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname += f":{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _attach_development_labels(
    records: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = {str(row.get("record_id") or ""): row for row in label_rows}
    record_ids = [str(row.get("record_id") or "") for row in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("development records must have unique non-empty record_id values")
    if len(labels) != len(label_rows) or set(labels) != set(record_ids):
        raise ValueError("development label sidecar must match development records exactly")
    result = []
    for record in records:
        label = labels[str(record["record_id"])]
        if not isinstance(label.get("if_chosen"), bool):
            raise ValueError(f"invalid if_chosen label for {record['record_id']}")
        result.append({**record, "if_chosen": label["if_chosen"], "score": label.get("score")})
    return result


def _load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    seed_path = _resolve_path(config["seed_data"])
    development_path = _resolve_path(config["development_data"])
    labels_path = _resolve_path(config["development_labels"])
    for path in (seed_path, development_path, labels_path):
        if "final_test" in str(path).casefold() or "final-test" in str(path).casefold():
            raise ValueError("the feedback-driven runner cannot read final_test data or labels")
        if not path.is_file():
            raise FileNotFoundError(path)
    seed_records = _read_jsonl(seed_path)
    development_records = _read_jsonl(development_path)
    development_labels = _read_jsonl(labels_path)
    if any(row.get("split") != "seed" for row in seed_records):
        raise ValueError("seed input must contain only split=seed records")
    if any(row.get("split") != "development" for row in development_records):
        raise ValueError("development input must contain only split=development records")
    manifest_path = seed_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("prepared PRISM manifest.json is required beside seed.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "prepared"
        or manifest.get("dataset") != PRISM_REPOSITORY
        or manifest.get("revision") != PRISM_REVISION
    ):
        raise ValueError("prepared data must match the pinned PRISM dataset revision")
    seed_groups = {str(row.get("source_group_id") or "") for row in seed_records}
    development_groups = {str(row.get("source_group_id") or "") for row in development_records}
    if not seed_groups or "" in seed_groups or seed_groups & development_groups:
        raise ValueError("seed/development source-group separation is invalid")
    labelled_development = _attach_development_labels(development_records, development_labels)
    calibration, evaluation = split_development_groups(
        labelled_development,
        calibration_fraction=float(config.get("calibration_fraction", 0.35)),
        seed=int(config.get("calibration_seed", 1701)),
    )
    if not {row["if_chosen"] for row in calibration} == {False, True}:
        raise ValueError("selector calibration split must contain chosen and rejected labels")
    if not {row["if_chosen"] for row in evaluation} == {False, True}:
        raise ValueError("development evaluation split must contain chosen and rejected labels")
    return {
        "seed_path": seed_path,
        "development_path": development_path,
        "labels_path": labels_path,
        "manifest_path": manifest_path,
        "seed_records": seed_records,
        "calibration_records": calibration,
        "evaluation_records": evaluation,
        "seed_groups": seed_groups,
    }


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ("seed_data", "development_data", "development_labels", "output", "endpoint")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"missing required config values: {', '.join(missing)}")
    if config.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"generation model is frozen to {MODEL_ID}")
    if config.get("model_revision", MODEL_REVISION) != MODEL_REVISION:
        raise ValueError(f"model revision is frozen to {MODEL_REVISION}")
    base_model = str(config.get("base_model") or MODEL_ID)
    if base_model != MODEL_ID and MODEL_REVISION not in base_model:
        raise ValueError("base_model must be Qwen3-8B or a local snapshot of the pinned revision")
    comparison = str(config.get("comparison", "overall"))
    if comparison not in COMPARISONS:
        raise ValueError(f"comparison must be one of {', '.join(COMPARISONS)}")
    run_purpose = str(config.get("run_purpose", "pilot"))
    if run_purpose not in {"pilot", "confirmatory"}:
        raise ValueError("run_purpose must be 'pilot' or 'confirmatory'")
    config = {**_default_config(), **config}
    if config["feedback_mode"] not in {"human", "model"}:
        raise ValueError("feedback_mode must be human or model")
    if config["feedback_mode"] == "model":
        if config["feedback_model"] != FEEDBACK_MODEL_ID or config["feedback_model_revision"] != FEEDBACK_MODEL_REVISION:
            raise ValueError("model feedback requires the pinned Qwen3-32B revision")
        if not config["feedback_endpoint"]:
            raise ValueError("feedback_endpoint is required for model feedback")
        if not isinstance(config["feedback_max_tokens"], int) or isinstance(config["feedback_max_tokens"], bool) or config["feedback_max_tokens"] <= 0:
            raise ValueError("feedback_max_tokens must be a positive integer")
        if config["feedback_temperature"] != 0:
            raise ValueError("model feedback temperature must be 0")
    positive_integer_values = (
        "rounds",
        "seed_group_limit",
        "candidate_count_per_arm_round",
        "query_count_per_arm_round",
        "generation_token_budget_per_arm_round",
        "max_tokens",
        "train_records",
        "train_token_budget",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "eval_batch_size",
        "max_length",
        "lora_r",
        "lora_alpha",
    )
    for key in positive_integer_values:
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("feedback_budget_seconds_per_arm_round", "estimated_review_seconds", "timeout_seconds"):
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("temperature", "learning_rate", "lora_dropout", "selector_learning_rate"):
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if not 0.0 <= config["temperature"] <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= config["lora_dropout"] < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not 0.0 < config["selector_learning_rate"] <= 1.0:
        raise ValueError("selector_learning_rate must be in (0, 1]")
    if config["query_count_per_arm_round"] > config["candidate_count_per_arm_round"]:
        raise ValueError("query_count_per_arm_round cannot exceed candidate_count_per_arm_round")
    if config["max_tokens"] < 2:
        raise ValueError("max_tokens must be at least 2 for task-input and response generation")
    CleaningPolicy(
        near_duplicate_threshold=float(config["cleaning_near_duplicate_threshold"]),
        repeated_ngram_threshold=float(config["cleaning_repeated_ngram_threshold"]),
    ).validate()
    seeds = config.get("seeds", [17])
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(value, int) or isinstance(value, bool) for value in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be a non-empty list of unique integers")
    quarantine_fraction = config["quarantine_review_fraction"]
    if (
        not isinstance(quarantine_fraction, (int, float))
        or isinstance(quarantine_fraction, bool)
        or not math.isfinite(quarantine_fraction)
        or not 0.0 <= quarantine_fraction <= 1.0
    ):
        raise ValueError("quarantine_review_fraction must be between 0 and 1")
    config["comparison"] = comparison
    config["arms"] = list(COMPARISONS[comparison])
    config["base_model"] = base_model
    config["seeds"] = [int(value) for value in seeds]
    return config


def _default_config() -> dict[str, Any]:
    return {
        "comparison": "overall",
        "run_purpose": "pilot",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model": MODEL_ID,
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 300,
        "rounds": 2,
        "seeds": [17],
        "seed_group_limit": 32,
        "candidate_count_per_arm_round": 32,
        "query_count_per_arm_round": 16,
        "quarantine_review_fraction": 0.25,
        "feedback_budget_seconds_per_arm_round": 1800,
        "feedback_mode": "human",
        "feedback_endpoint": "http://127.0.0.1:8001/v1/chat/completions",
        "feedback_model": FEEDBACK_MODEL_ID,
        "feedback_model_revision": FEEDBACK_MODEL_REVISION,
        "feedback_api_key_env": "OPENAI_API_KEY",
        "feedback_max_tokens": 512,
        "feedback_temperature": 0,
        "estimated_review_seconds": 90,
        "calibration_fraction": 0.35,
        "calibration_seed": 1701,
        "generation_token_budget_per_arm_round": 30000,
        "max_tokens": 512,
        "temperature": 0.7,
        "train_records": 256,
        "train_token_budget": 100000,
        "max_steps": 1000,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "eval_batch_size": 8,
        "max_length": 1024,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "cleaning_near_duplicate_threshold": 0.92,
        "cleaning_repeated_ngram_threshold": 0.65,
        "allow_cpu_training": False,
        "save_adapters": True,
        "selector_learning_rate": 0.2,
    }


def _make_generator(config: dict[str, Any]) -> OpenAICompatibleGenerator:
    api_key = os.environ.get(str(config["api_key_env"]))
    return OpenAICompatibleGenerator(
        str(config["endpoint"]),
        str(config["model"]),
        api_key=api_key,
        timeout_seconds=float(config["timeout_seconds"]),
    )


def _generate_candidates(
    seeds: list[dict[str, Any]],
    *,
    generator: Any,
    count: int,
    seed: int,
    round_index: int,
    arm: Arm,
    policy_state: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    usage = getattr(generator, "usage", None)
    token_budget = int(config["generation_token_budget_per_arm_round"])
    for index in range(count):
        used = (
            int(getattr(usage, "prompt_tokens", 0)) + int(getattr(usage, "completion_tokens", 0))
            if usage is not None
            else 0
        )
        if used >= token_budget:
            break
        record = synthesize(
            seeds,
            strategy="self-instruct",
            count=1,
            generator=generator,
            seed_value=seed + round_index * 100_003 + index,
            max_tokens=int(config["max_tokens"]),
            temperature=float(config["temperature"]),
            index_offset=index,
            policy_guidance=policy_state["synthesis"].get("guidance", []),
        )[0]
        stable = hashlib.sha256(
            f"{seed}:{round_index}:{index}:{record['parent_record_id']}".encode("utf-8")
        ).hexdigest()[:24]
        generated.append(
            {
                **record,
                "record_id": f"{arm.name}-{seed}-r{round_index}-{stable}",
                "round": round_index,
                "split": "generated",
                "policy_version": policy_state["synthesis"]["version"],
                "generation_model": config["model"],
                "generation_revision": config["model_revision"],
            }
        )
    if not generated:
        raise RuntimeError("Self-Instruct produced no candidate before its generation budget ended")
    used = (
        int(getattr(usage, "prompt_tokens", 0)) + int(getattr(usage, "completion_tokens", 0))
        if usage is not None
        else 0
    )
    usage_info = {
        "requests": int(getattr(usage, "requests", 0)),
        "complete_usage_requests": int(getattr(usage, "complete_usage_requests", 0)),
        "reported_usage": bool(getattr(usage, "reported", False)),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0)),
        "token_budget": token_budget,
        "token_overshoot": max(0, used - token_budget),
        "candidate_count": len(generated),
    }
    if usage_info["requests"] and usage_info["complete_usage_requests"] != usage_info["requests"]:
        raise RuntimeError("generation endpoint must return prompt and completion tokens on every request")
    if usage_info["token_overshoot"]:
        raise RuntimeError(
            f"generation token budget exceeded: {used} > {token_budget}; reduce candidate count or max_tokens"
        )
    return generated, usage_info


def _clone_shared_candidates(
    candidates: list[dict[str, Any]], *, arm: Arm, seed: int, round_index: int
) -> list[dict[str, Any]]:
    cloned = []
    for index, candidate in enumerate(candidates):
        stable = hashlib.sha256(
            f"{seed}:{round_index}:{index}:{candidate['parent_record_id']}".encode("utf-8")
        ).hexdigest()[:24]
        cloned.append(
            {
                **candidate,
                "record_id": f"{arm.name}-{seed}-r{round_index}-{stable}",
                "shared_candidate_pool": "query-selection-round-0",
                "shared_source_record_id": candidate["record_id"],
            }
        )
    return cloned


def _feedback_training_records(
    candidates: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate_by_id = {str(row["record_id"]): row for row in candidates}
    output = []
    for event in events:
        candidate = candidate_by_id.get(str(event.get("record_id") or ""))
        if candidate is None:
            raise ValueError(f"feedback references an unknown candidate: {event.get('record_id')}")
        judgment = event.get("judgment")
        corrected = str(event.get("corrected_output") or "").strip()
        if corrected:
            target = corrected
        elif judgment == "usable":
            target = str(candidate.get("output") or "").strip()
        else:
            continue
        if not target:
            continue
        output.append(
            {
                **candidate,
                "record_id": f"feedback-{event['query_id']}",
                "output": target,
                "feedback_query_id": event["query_id"],
                "feedback_judgment": judgment,
                "feedback_source": event["feedback_source"],
                "synthesis_strategy": event["feedback_source"] + "-feedback",
            }
        )
    return output


def _make_training_set(
    seed_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = [
        {**row, "synthesis_strategy": "seed"}
        for row in seed_records
        if str(row.get("input") or "").strip() and str(row.get("output") or "").strip()
    ]
    pool.extend(feedback_records)
    if not pool:
        raise ValueError("training pool has no non-empty input/output records")
    return build_training_schedule(pool, count, seed=seed)


def _training_config(config: dict[str, Any], *, seed: int, adapter_path: Path | None) -> QwenSFTConfig:
    return QwenSFTConfig(
        base_model=str(config["base_model"]),
        model_revision=str(config["model_revision"]),
        train_token_budget=int(config["train_token_budget"]),
        require_cuda=not bool(config["allow_cpu_training"]),
        max_steps=int(config["max_steps"]),
        batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_length=int(config["max_length"]),
        learning_rate=float(config["learning_rate"]),
        lora_r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        eval_batch_size=int(config["eval_batch_size"]),
        seed=seed,
        save_adapter=str(adapter_path) if adapter_path else None,
    )


def _append_results(path: Path, document: dict[str, Any]) -> None:
    _write_json(path, document)


def _configured_initial_state(config: dict[str, Any]) -> dict[str, Any]:
    state = initial_policy_state()
    state["cleaning"]["near_duplicate_threshold"] = float(
        config["cleaning_near_duplicate_threshold"]
    )
    state["cleaning"]["repeated_ngram_threshold"] = float(
        config["cleaning_repeated_ngram_threshold"]
    )
    return state


def _find_round_result(
    document: dict[str, Any], arm_name: str, seed: int, round_index: int
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in document["arms"][arm_name]["rounds"]
            if int(row["seed"]) == seed and int(row["round"]) == round_index
        ),
        None,
    )


def _restore_arm_history(
    output_root: Path,
    document: dict[str, Any],
    config: dict[str, Any],
    *,
    arm_name: str,
    seed: int,
    before_round: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = _configured_initial_state(config)
    accumulated_feedback: list[dict[str, Any]] = []
    for round_index in range(before_round):
        if _find_round_result(document, arm_name, seed, round_index) is None:
            raise ValueError(
                f"cannot resume {arm_name}/seed-{seed}: round {round_index} is not complete"
            )
        round_dir = output_root / f"seed-{seed}" / arm_name / f"round-{round_index}"
        state_path = round_dir / "policy-state-next.json"
        generated_path = round_dir / "generated.jsonl"
        feedback_path = round_dir / "feedback.jsonl"
        if not all(path.is_file() for path in (state_path, generated_path, feedback_path)):
            raise FileNotFoundError(f"resume checkpoint is incomplete: {round_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        events = read_feedback(feedback_path, expected_source="model_simulated" if config["feedback_mode"] == "model" else "live_human")
        accumulated_feedback.extend(
            _feedback_training_records(_read_jsonl(generated_path), events)
        )
    return state, accumulated_feedback


def run_feedback_rsi(
    raw_config: dict[str, Any],
    *,
    resume: bool = False,
    generator_factory: Callable[[dict[str, Any]], Any] | None = None,
    trainer: Callable[..., dict[str, Any]] | None = None,
    feedback_collector: Callable[..., list[dict[str, Any]]] | None = None,
    annotator_id: str | None = None,
) -> dict[str, Any]:
    """Execute selected system arms, collecting new feedback only after selection."""
    config = _validate_config(raw_config)
    data = _load_inputs(config)
    output_root = _resolve_path(str(config["output"]))
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result.json"
    if annotator_id is None and feedback_collector is None and config["feedback_mode"] == "human":
        annotator_id = os.environ.get("ALADDIN_ANNOTATOR_ID", "").strip()

    generator_factory = generator_factory or _make_generator
    trainer = trainer or train_and_evaluate
    use_default_collector = feedback_collector is None
    feedback_collector = feedback_collector or collect_interactive_feedback
    uncertainty_model = PreferenceUncertaintyModel().fit(
        data["calibration_records"], seed=int(config["calibration_seed"])
    )
    arm_definitions = [ARMS[name] for name in config["arms"]]
    run_id = uuid.uuid4().hex
    config_snapshot = {
        key: value
        for key, value in config.items()
        if key not in {"endpoint", "feedback_endpoint", "api_key", "token", "annotator_id"}
    }
    config_snapshot["endpoint"] = _safe_endpoint(str(config["endpoint"]))
    config_snapshot["feedback_endpoint"] = _safe_endpoint(str(config["feedback_endpoint"]))
    document: dict[str, Any] = {
        "status": "running",
        "protocol": "aladdin-feedback-driven-rsi-v1",
        "run_id": run_id,
        "comparison": config["comparison"],
        "configuration": config_snapshot,
        "feedback_boundary": {
            "phase_one_human_feedback": False,
            "phase_two_human_feedback": False,
            "phase_three_live_feedback": config["feedback_mode"] == "human",
            "phase_three_model_feedback": config["feedback_mode"] == "model",
            "feedback_prompt_version": FEEDBACK_PROMPT_VERSION if config["feedback_mode"] == "model" else None,
            "feedback_updates_next_round_only": True,
        },
        "data": {
            "dataset": PRISM_REPOSITORY,
            "dataset_revision": PRISM_REVISION,
            "seed_sha256": _sha256(data["seed_path"]),
            "development_sha256": _sha256(data["development_path"]),
            "development_labels_sha256": _sha256(data["labels_path"]),
            "split_manifest_sha256": _sha256(data["manifest_path"]),
            "seed_groups": len(data["seed_groups"]),
            "seed_groups_sha256": _hash_values(data["seed_groups"]),
            "selector_calibration_rows": len(data["calibration_records"]),
            "selector_calibration_groups": len({row["source_group_id"] for row in data["calibration_records"]}),
            "selector_calibration_groups_sha256": _hash_values({row["source_group_id"] for row in data["calibration_records"]}),
            "development_evaluation_rows": len(data["evaluation_records"]),
            "development_evaluation_groups": len({row["source_group_id"] for row in data["evaluation_records"]}),
            "development_evaluation_groups_sha256": _hash_values({row["source_group_id"] for row in data["evaluation_records"]}),
            "final_test_accessed": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "runner_sha256": _sha256(Path(__file__)),
            "feedback_implementation_sha256": _sha256(ROOT / "experiments/feedback/model_feedback.py"),
        },
        "arms": {arm.name: {"definition": asdict(arm), "rounds": []} for arm in arm_definitions},
        "completed_at_utc": None,
    }
    if result_path.exists():
        if not resume:
            raise FileExistsError(
                f"output already contains a run; pass --resume to continue it: {output_root}"
            )
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            raise ValueError("completed runs cannot be resumed")
        if existing.get("protocol") != document["protocol"]:
            raise ValueError("existing output uses a different experiment protocol")
        if existing.get("configuration") != document["configuration"]:
            raise ValueError("resume configuration differs from the original run")
        if existing.get("data") != document["data"]:
            raise ValueError("resume data or split manifest differs from the original run")
        if existing.get("runtime", {}).get("runner_sha256") != document["runtime"]["runner_sha256"]:
            raise ValueError("runner code changed since the interrupted run; use a new output directory")
        if existing.get("runtime", {}).get("feedback_implementation_sha256") != document["runtime"]["feedback_implementation_sha256"]:
            raise ValueError("feedback implementation changed; use a new output directory")
        if set(existing.get("arms", {})) != set(document["arms"]):
            raise ValueError("existing run contains a different comparison matrix")
        document = existing
        document["status"] = "running"
        document.pop("error", None)
        document["completed_at_utc"] = None
        run_id = str(document["run_id"])
    elif resume:
        raise FileNotFoundError(f"cannot resume because result.json is missing: {result_path}")
    elif any(output_root.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_root}")
    _append_results(result_path, document)

    try:
        for run_seed in config["seeds"]:
            seed_rows = _select_seed_groups(data["seed_records"], int(config["seed_group_limit"]), run_seed)
            for round_index in range(int(config["rounds"])):
                queues: list[dict[str, Any]] = []
                arm_candidates: dict[str, list[dict[str, Any]]] = {}
                generation_info: dict[str, dict[str, Any]] = {}
                cleaner_info: dict[str, dict[str, Any]] = {}
                states: dict[str, dict[str, Any]] = {}
                accumulated_feedback: dict[str, list[dict[str, Any]]] = {}
                shared_initial_candidates: list[dict[str, Any]] | None = None
                shared_initial_usage: dict[str, Any] | None = None
                initial_pool_owner = arm_definitions[0].name
                for arm in arm_definitions:
                    arm_round_dir = output_root / f"seed-{run_seed}" / arm.name / f"round-{round_index}"
                    arm_round_dir.mkdir(parents=True, exist_ok=True)
                    state, prior_feedback = _restore_arm_history(
                        output_root,
                        document,
                        config,
                        arm_name=arm.name,
                        seed=run_seed,
                        before_round=round_index,
                    )
                    states[arm.name] = state
                    accumulated_feedback[arm.name] = prior_feedback
                    completed = _find_round_result(document, arm.name, run_seed, round_index)
                    generated_path = arm_round_dir / "generated.jsonl"
                    generation_info_path = arm_round_dir / "generation.json"
                    if completed is not None:
                        if not generated_path.is_file():
                            raise FileNotFoundError(f"resume checkpoint is incomplete: {generated_path}")
                        candidates = _read_jsonl(generated_path)
                        usage = completed["generation"]
                        cleaner_info[arm.name] = completed["cleaning"]
                    elif generated_path.is_file() and generation_info_path.is_file():
                        candidates = _read_jsonl(generated_path)
                        usage = json.loads(generation_info_path.read_text(encoding="utf-8"))
                    else:
                        share_candidates = round_index == 0
                        if share_candidates and arm.name != initial_pool_owner:
                            if shared_initial_candidates is None or shared_initial_usage is None:
                                raise RuntimeError("the first comparison arm must generate the shared round-zero candidate pool")
                            candidates = _clone_shared_candidates(
                                shared_initial_candidates,
                                arm=arm,
                                seed=run_seed,
                                round_index=round_index,
                            )
                            usage = {
                                "requests": 0,
                                "complete_usage_requests": 0,
                                    "reported_usage": shared_initial_usage["reported_usage"],
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                    "token_budget": shared_initial_usage["token_budget"],
                                "token_overshoot": 0,
                                "candidate_count": len(candidates),
                                "shared_candidate_pool": "comparison-initial-round-0",
                                "generation_cost_owner": initial_pool_owner,
                            }
                        else:
                            generator = generator_factory(config)
                            candidates, usage = _generate_candidates(
                                seed_rows,
                                generator=generator,
                                count=int(config["candidate_count_per_arm_round"]),
                                seed=run_seed,
                                round_index=round_index,
                                arm=arm,
                                policy_state=state,
                                config=config,
                            )
                            if share_candidates:
                                usage = {
                                    **usage,
                                    "shared_candidate_pool": "comparison-initial-round-0",
                                    "generation_cost_owner": initial_pool_owner,
                                }
                                shared_initial_candidates = candidates
                                shared_initial_usage = usage
                        _write_jsonl(generated_path, candidates)
                        _write_json(generation_info_path, usage)
                    if (
                        round_index == 0
                        and arm.name == initial_pool_owner
                    ):
                        shared_initial_candidates = candidates
                        shared_initial_usage = usage
                    arm_candidates[arm.name] = candidates
                    generation_info[arm.name] = usage

                    cleaner_policy = CleaningPolicy(
                        version=int(state["cleaning"]["version"]),
                        near_duplicate_threshold=float(state["cleaning"]["near_duplicate_threshold"]),
                        repeated_ngram_threshold=float(state["cleaning"]["repeated_ngram_threshold"]),
                    )
                    cleaned, quarantined, decisions = clean_candidates(
                        [*seed_rows, *candidates], cleaner_policy
                    )
                    candidate_ids = {str(row["record_id"]) for row in candidates}
                    review_pool = [
                        row for row in [*cleaned, *quarantined]
                        if str(row.get("record_id") or "") in candidate_ids
                    ]
                    _write_jsonl(arm_round_dir / "retained.jsonl", [row for row in cleaned if row.get("record_id") in candidate_ids])
                    _write_jsonl(arm_round_dir / "quarantine.jsonl", [row for row in quarantined if row.get("record_id") in candidate_ids])
                    _write_jsonl(arm_round_dir / "cleaning-decisions.jsonl", decisions)
                    computed_cleaner_info = {
                        "input_records_including_seed": len(seed_rows) + len(candidates),
                        "candidate_records": len(candidates),
                        "retained_candidates": sum(row.get("cleaning", {}).get("status") == "retained" for row in review_pool),
                        "quarantined_candidates": sum(row.get("cleaning", {}).get("status") == "quarantined" for row in review_pool),
                        "policy": asdict(cleaner_policy),
                    }
                    if completed is None:
                        cleaner_info[arm.name] = computed_cleaner_info
                        queries = select_queries(
                            review_pool,
                            method=arm.selector,
                            count=int(config["query_count_per_arm_round"]),
                            seed=run_seed + round_index,
                            run_id=run_id,
                            arm_id=arm.name,
                            round_index=round_index,
                            uncertainty_model=uncertainty_model,
                            estimated_review_seconds=float(config["estimated_review_seconds"]),
                            impact_weight=float(state["selection"]["impact_weight"]),
                            quarantine_fraction=float(config["quarantine_review_fraction"]),
                        )
                    else:
                        queries = _read_jsonl(arm_round_dir / "query-queue.jsonl")
                    queues.extend(queries)
                    _write_jsonl(arm_round_dir / "query-queue.jsonl", queries)

                random.Random(run_seed + round_index + 901).shuffle(queues)
                if config["feedback_mode"] == "model":
                    queue_counts = {arm.name: sum(query.get("arm_id") == arm.name for query in queues) for arm in arm_definitions}
                    if any(count != config["query_count_per_arm_round"] for count in queue_counts.values()):
                        raise ValueError(f"each arm must supply the configured number of model feedback queries: {queue_counts}")
                feedback_path = output_root / f"seed-{run_seed}" / "feedback" / f"round-{round_index}.jsonl"
                expected_source = "model_simulated" if config["feedback_mode"] == "model" else "live_human"
                stored_events = read_feedback(feedback_path, expected_source=expected_source) if feedback_path.exists() else []
                stored_ids = {str(event["query_id"]) for event in stored_events}
                pending_queries = [query for query in queues if str(query["query_id"]) not in stored_ids]
                if pending_queries and use_default_collector and config["feedback_mode"] == "human" and not annotator_id:
                    raise ValueError("set ALADDIN_ANNOTATOR_ID to a pseudonymous reviewer ID")
                if not pending_queries:
                    events = stored_events
                elif use_default_collector and config["feedback_mode"] == "model":
                    feedback_generator = OpenAICompatibleGenerator(
                        str(config["feedback_endpoint"]), str(config["feedback_model"]),
                        api_key=os.environ.get(str(config["feedback_api_key_env"])),
                        timeout_seconds=float(config["timeout_seconds"]),
                    )
                    events = collect_model_feedback(
                        queues, output_path=feedback_path, generator=feedback_generator,
                        model_revision=str(config["feedback_model_revision"]),
                        max_tokens=int(config["feedback_max_tokens"]),
                        temperature=float(config["feedback_temperature"]),
                    )
                elif use_default_collector:
                    events = feedback_collector(
                        queues,
                        annotator_id=annotator_id,
                        budget_seconds_per_arm=float(config["feedback_budget_seconds_per_arm_round"]),
                        output_path=feedback_path,
                    )
                else:
                    pending_path = (
                        feedback_path
                        if not stored_events
                        else feedback_path.with_name(f"{feedback_path.stem}.pending.jsonl")
                    )
                    collected = feedback_collector(
                        pending_queries,
                        annotator_id=annotator_id or "test-reviewer",
                        budget_seconds_per_arm=float(config["feedback_budget_seconds_per_arm_round"]),
                        output_path=pending_path,
                    )
                    pending_events = read_feedback(pending_path, expected_source=expected_source) if pending_path.exists() else collected
                    events = [*stored_events, *pending_events]
                    _write_jsonl(feedback_path, events)
                    if pending_path != feedback_path and pending_path.exists():
                        pending_path.unlink()
                if feedback_path.exists():
                    events = read_feedback(feedback_path, expected_source=expected_source)
                for event in events:
                    if event.get("feedback_source") != expected_source:
                        raise ValueError("feedback source does not match configured feedback mode")
                    duration = event.get("duration_seconds")
                    if (
                        not isinstance(duration, (int, float))
                        or isinstance(duration, bool)
                        or not math.isfinite(duration)
                        or duration < 0
                    ):
                        raise ValueError("feedback event has an invalid measured duration")
                    if event.get("judgment") not in set(JUDGMENTS.values()):
                        raise ValueError("feedback event has an invalid judgment")
                    if event.get("issue_code") not in set(ISSUES.values()):
                        raise ValueError("feedback event has an invalid issue code")
                    if not isinstance(event.get("actionable"), bool):
                        raise ValueError("feedback event actionable flag must be boolean")
                    if not str(event.get("annotator_id") or "").strip():
                        raise ValueError("feedback event must include a pseudonymous annotator ID")
                    if event.get("requirement_clarity") not in {"clear", "partly_unclear", "unclear", "not_applicable"}:
                        raise ValueError("feedback event has an invalid requirement-clarity value")
                if len({str(event.get("query_id") or "") for event in events}) != len(events):
                    raise ValueError("feedback collector returned duplicate query IDs")
                query_by_id = {str(query["query_id"]): query for query in queues}
                if len(query_by_id) != len(queues):
                    raise ValueError("selected queue contains duplicate query IDs")
                for event in events:
                    query = query_by_id.get(str(event.get("query_id") or ""))
                    if query is None:
                        raise ValueError("feedback event references a query that was not selected")
                    candidate = query["candidate"]
                    if (
                        event.get("record_id") != query.get("record_id")
                        or event.get("arm_id") != query.get("arm_id")
                        or event.get("source_group_id") != query.get("source_group_id")
                        or event.get("selection_method") != query.get("selection_method")
                        or event.get("round") != query.get("round")
                        or candidate.get("record_id") != query.get("record_id")
                    ):
                        raise ValueError("feedback event metadata does not match its selected candidate")
                events_by_arm = {arm.name: [] for arm in arm_definitions}
                for event in events:
                    arm_name = str(event.get("arm_id") or "")
                    if arm_name not in events_by_arm:
                        raise ValueError(f"feedback event references unknown arm: {arm_name}")
                    events_by_arm[arm_name].append(event)

                attempt_path = feedback_path.with_name(feedback_path.stem + ".model-attempts.jsonl")
                attempts = _read_jsonl(attempt_path) if config["feedback_mode"] == "model" and attempt_path.exists() else []

                for arm in arm_definitions:
                    arm_round_dir = output_root / f"seed-{run_seed}" / arm.name / f"round-{round_index}"
                    arm_events = events_by_arm[arm.name]
                    arm_attempts = [row for row in attempts if row.get("arm_id") == arm.name]
                    arm_queries = [row for row in queues if row.get("arm_id") == arm.name]
                    query_ids = {str(row["query_id"]) for row in arm_queries}
                    if any(str(event.get("query_id") or "") not in query_ids for event in arm_events):
                        raise ValueError(f"feedback contains a query not selected for arm {arm.name}")
                    _write_jsonl(arm_round_dir / "feedback.jsonl", arm_events)
                    completed = _find_round_result(document, arm.name, run_seed, round_index)
                    if completed is not None:
                        states[arm.name] = json.loads(
                            (arm_round_dir / "policy-state-next.json").read_text(encoding="utf-8")
                        )
                        accumulated_feedback[arm.name].extend(
                            _feedback_training_records(arm_candidates[arm.name], arm_events)
                        )
                        continue
                    feedback_rows = _feedback_training_records(arm_candidates[arm.name], arm_events)
                    accumulated_feedback[arm.name].extend(feedback_rows)
                    next_state, policy_event = update_policy_state(
                        states[arm.name],
                        arm_events,
                        update_synthesis=arm.update_synthesis,
                        update_cleaning=arm.update_cleaning,
                        update_selection=arm.update_selection,
                        selector_learning_rate=float(config["selector_learning_rate"]),
                    )
                    _write_json(arm_round_dir / "policy-update.json", policy_event)
                    _write_json(arm_round_dir / "policy-state-next.json", next_state)

                    training_rows = _make_training_set(
                        seed_rows,
                        accumulated_feedback[arm.name],
                        count=int(config["train_records"]),
                        seed=run_seed + round_index,
                    )
                    _write_jsonl(arm_round_dir / "training.jsonl", training_rows)
                    adapter_path = arm_round_dir / "adapter" if config["save_adapters"] else None
                    training_metrics = trainer(
                        training_rows,
                        data["evaluation_records"],
                        _training_config(
                            config,
                            seed=run_seed + round_index,
                            adapter_path=adapter_path,
                        ),
                    )
                    if int(training_metrics.get("pairwise_comparisons", 0)) <= 0:
                        raise RuntimeError("development evaluation produced no chosen/rejected comparisons")
                    review_seconds = sum(float(row.get("duration_seconds") or 0) for row in arm_events)
                    round_result = {
                        "seed": run_seed,
                        "round": round_index,
                        "generation": generation_info[arm.name],
                        "cleaning": cleaner_info[arm.name],
                        "selected_queries": len(arm_queries),
                        "selected_quarantined_queries": sum(
                            bool(row.get("candidate_was_quarantined")) for row in arm_queries
                        ),
                        "selected_retained_queries": sum(
                            not bool(row.get("candidate_was_quarantined")) for row in arm_queries
                        ),
                        "completed_feedback_events": len(arm_events),
                        "feedback_source": expected_source,
                        "feedback_seconds": review_seconds if expected_source == "live_human" else None,
                        "feedback_budget_seconds": config["feedback_budget_seconds_per_arm_round"] if expected_source == "live_human" else None,
                        "feedback_budget_overrun_seconds": max(
                            0.0,
                            review_seconds - float(config["feedback_budget_seconds_per_arm_round"]),
                        ) if expected_source == "live_human" else None,
                        "model_feedback_seconds": sum(float(row["duration_seconds"]) for row in arm_attempts) if expected_source == "model_simulated" else None,
                        "model_feedback_requests": sum(int(row["requests"]) for row in arm_attempts) if expected_source == "model_simulated" else None,
                        "model_feedback_failed_attempts": sum(row["status"] == "error" for row in arm_attempts) if expected_source == "model_simulated" else None,
                        "model_feedback_prompt_tokens": sum(int(row["prompt_tokens"]) for row in arm_attempts) if expected_source == "model_simulated" else None,
                        "model_feedback_completion_tokens": sum(int(row["completion_tokens"]) for row in arm_attempts) if expected_source == "model_simulated" else None,
                        "actionable_feedback_count": sum(bool(row.get("actionable")) for row in arm_events),
                        "policy_update": policy_event,
                        "training_feedback_examples_cumulative": len(accumulated_feedback[arm.name]),
                        "training": training_metrics,
                    }
                    states[arm.name] = next_state
                    document["arms"][arm.name]["rounds"].append(round_result)
                    _append_results(result_path, document)
                    print(json.dumps({"arm": arm.name, "seed": run_seed, "round": round_index, "metrics": training_metrics}, ensure_ascii=False))

        document["status"] = "completed"
        document["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        document["summary"] = _summarize_arms(document["arms"])
        _append_results(result_path, document)
        return document
    except Exception as exc:
        document["status"] = "failed"
        document["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _append_results(result_path, document)
        raise


def _select_seed_groups(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("seed_group_limit must be positive")
    groups = sorted({str(row.get("source_group_id") or "") for row in records})
    if not groups or "" in groups:
        raise ValueError("seed records must have non-empty source_group_id values")
    if limit > len(groups):
        raise ValueError(f"requested {limit} seed groups but only {len(groups)} are available")
    random.Random(seed).shuffle(groups)
    selected = set(groups[:limit])
    return [row for row in records if str(row["source_group_id"]) in selected]


def _summarize_arms(arms: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for arm_name, value in arms.items():
        rounds = value["rounds"]
        accuracies = [
            float(row["training"]["pairwise_accuracy"])
            for row in rounds
            if row["training"].get("pairwise_accuracy") is not None
        ]
        summary[arm_name] = {
            "rounds": len(rounds),
            "mean_development_pairwise_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "feedback_seconds": sum(float(row["feedback_seconds"] or 0) for row in rounds) if any(row["feedback_seconds"] is not None for row in rounds) else None,
            "model_feedback_seconds": sum(float(row.get("model_feedback_seconds") or 0) for row in rounds) if any(row.get("model_feedback_seconds") is not None for row in rounds) else None,
            "model_feedback_requests": sum(int(row.get("model_feedback_requests") or 0) for row in rounds),
            "model_feedback_failed_attempts": sum(int(row.get("model_feedback_failed_attempts") or 0) for row in rounds),
            "model_feedback_prompt_tokens": sum(int(row.get("model_feedback_prompt_tokens") or 0) for row in rounds),
            "model_feedback_completion_tokens": sum(int(row.get("model_feedback_completion_tokens") or 0) for row in rounds),
            "feedback_events": sum(int(row["completed_feedback_events"]) for row in rounds),
            "actionable_feedback_events": sum(int(row["actionable_feedback_count"]) for row in rounds),
            "policy_updates": sum(row["policy_update"]["status"] == "updated" for row in rounds),
        }
    return summary
