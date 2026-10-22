from __future__ import annotations

"""Qwen causal-LM preference classifier shared by trainer and evaluator."""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .progress import ProgressReporter


LABEL_TO_TEXT = {"A": "preferred", "B": "rejected", "C": "tie"}
TEXT_TO_LABEL = {value: key for key, value in LABEL_TO_TEXT.items()}


def base_model_path(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    value = config.get("base_model_path") or os.environ.get("LEGO_TASK_MODEL_BASE_PATH")
    if not isinstance(value, str) or not value or value.startswith("SET_"):
        raise ValueError(
            "Set task_model.training_config.base_model_path or "
            "LEGO_TASK_MODEL_BASE_PATH to a local Hugging Face causal-LM model directory"
        )
    return value


def device() -> torch.device:
    requested = os.environ.get("LEGO_TASK_MODEL_DEVICE", "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _dtype(target: torch.device) -> torch.dtype:
    if target.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _configure_memory_limit(target: torch.device, config: dict[str, Any] | None = None) -> None:
    """Keep the LEGO process below its shared-GPU budget before allocation."""
    if target.type != "cuda":
        return
    config = config or {}
    raw_limit = config.get("max_gpu_memory_gib") or os.environ.get(
        "LEGO_TASK_MODEL_MAX_GPU_MEMORY_GIB"
    )
    if raw_limit in (None, ""):
        return
    limit_gib = float(raw_limit)
    if limit_gib <= 0:
        raise ValueError("max_gpu_memory_gib must be positive")
    total_gib = torch.cuda.get_device_properties(target).total_memory / (1024**3)
    fraction = min(0.95, limit_gib / total_gib)
    # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to local index 0.
    device_index = target.index if target.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    print(
        f"[LEGO] CUDA memory limit: {limit_gib:.1f} GiB "
        f"({fraction:.3f} of {total_gib:.1f} GiB)",
        file=sys.stderr,
        flush=True,
    )


def prompt_for(record: dict[str, Any]) -> str:
    return (
        "Classify the response quality relative to the task prompt.\n"
        "Return exactly one label: A=preferred, B=rejected, C=tie.\n"
        f"{record['input']}\n"
        "Label:"
    )


def _records(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("records must be a list")
    result = [item for item in raw if isinstance(item, dict) and item.get("target") is not None]
    if not result:
        raise ValueError("records must contain at least one labeled record")
    return result


def _label_for_target(target: Any) -> str:
    text = str(target).strip().lower()
    try:
        return TEXT_TO_LABEL[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported HelpSteer2 target: {target!r}") from exc


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # The supervision token and evaluation position are at the end of the
    # prompt; preserve them when a long conversation exceeds max_length.
    tokenizer.truncation_side = "left"
    return tokenizer


def label_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for label in LABEL_TO_TEXT:
        encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"Task label {label!r} is not one tokenizer token")
        ids[label] = int(encoded[0])
    return ids


def load_base(path: str, target: torch.device):
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=_dtype(target),
    ).to(target)


def load_adapter_checkpoint(checkpoint: dict[str, Any], target: torch.device):
    path = str(checkpoint["base_model_path"])
    tokenizer = load_tokenizer(path)
    model = load_base(path, target)
    model = PeftModel.from_pretrained(
        model,
        str(checkpoint["adapter_path"]),
        local_files_only=True,
    ).to(target)
    model.eval()
    return tokenizer, model, {str(k): int(v) for k, v in checkpoint["label_ids"].items()}


def _configure_reproducibility(seed: int, *, deterministic: bool) -> None:
    """Reset all trainer RNGs before model construction and optionally force deterministic CUDA ops."""
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def _training_recipe(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": float(config.get("learning_rate", 2e-4)),
        "max_steps": int(config.get("max_steps", 100)),
        "gradient_accumulation_steps": max(
            1, int(config.get("gradient_accumulation_steps", 1))
        ),
        "batch_size": max(
            1, int(config.get("per_device_batch_size", config.get("batch_size", 4)))
        ),
        "max_length": int(config.get("max_length", 1024)),
        "lora_r": int(config.get("lora_r", 8)),
        "lora_alpha": int(config.get("lora_alpha", 16)),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "target_modules": list(config.get("target_modules", ["q_proj", "v_proj"])),
    }


def _load_trainable_model(
    base_path: str,
    target: torch.device,
    *,
    config: dict[str, Any],
    initial_checkpoint: dict[str, Any] | None,
    current_label_ids: dict[str, int],
    recipe_hash: str,
):
    parent_adapter_path: Path | None = None
    parent_metadata: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        checkpoint = initial_checkpoint
        if checkpoint.get("backend") != "qwen-lora-preference-v1":
            raise ValueError("parent checkpoint uses an incompatible backend")
        try:
            parent_base = Path(str(checkpoint["base_model_path"])).resolve(strict=True)
            current_base = Path(base_path).resolve(strict=True)
            parent_adapter_path = Path(str(checkpoint["adapter_path"])).resolve(strict=True)
        except (KeyError, OSError) as exc:
            raise ValueError("parent checkpoint paths must exist") from exc
        if parent_base != current_base:
            raise ValueError("parent checkpoint uses a different base model path")
        current_revision = str(config.get("base_model_revision", "unrecorded"))
        parent_revision = str(checkpoint.get("base_model_revision", "unrecorded"))
        if parent_revision != current_revision:
            raise ValueError("parent checkpoint uses a different base model revision")
        current_model_id = config.get("base_model_id")
        parent_model_id = checkpoint.get("base_model_id")
        if current_model_id and parent_model_id and parent_model_id != current_model_id:
            raise ValueError("parent checkpoint uses a different base model ID")
        if not parent_adapter_path.is_dir():
            raise ValueError("parent adapter path must be a directory")
        actual_hash = hash_directory(parent_adapter_path)
        if actual_hash != checkpoint.get("adapter_hash"):
            raise ValueError("parent adapter hash does not match its checkpoint manifest")
        if checkpoint.get("label_ids") != current_label_ids:
            raise ValueError("parent checkpoint label token mapping does not match")
        if checkpoint.get("training_recipe_hash") != recipe_hash:
            raise ValueError("parent checkpoint training recipe does not match the current run")
        parent_metadata = {
            "task_model_version": str(checkpoint.get("task_model_version", actual_hash[:16])),
            "adapter_hash": actual_hash,
        }

    base_model = load_base(base_path, target)
    if initial_checkpoint is None:
        return get_peft_model(
            base_model,
            LoraConfig(
                r=int(config.get("lora_r", 8)),
                lora_alpha=int(config.get("lora_alpha", 16)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                target_modules=list(config.get("target_modules", ["q_proj", "v_proj"])),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ), None

    assert parent_adapter_path is not None and parent_metadata is not None
    return PeftModel.from_pretrained(
        base_model,
        str(parent_adapter_path),
        is_trainable=True,
        local_files_only=True,
    ), parent_metadata


def train_checkpoint(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    initial_checkpoint: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int | float]]:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"checkpoint output directory is not empty: {output_dir}")
    deterministic = bool(
        config.get("deterministic", os.environ.get("LEGO_DETERMINISTIC", "0") == "1")
    )
    _configure_reproducibility(seed, deterministic=deterministic)
    target = device()
    _configure_memory_limit(target, config)
    base_path = base_model_path(config)
    model_id = str(config.get("base_model_id") or Path(base_path).name)
    print(f"[LEGO] Task Model {model_id}: loading on {target}...", file=sys.stderr, flush=True)
    tokenizer = load_tokenizer(base_path)
    tokenizer.padding_side = "right"
    ids = label_ids(tokenizer)
    recipe = _training_recipe(config)
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model, parent_checkpoint = _load_trainable_model(
        base_path,
        target,
        config=config,
        initial_checkpoint=initial_checkpoint,
        current_label_ids=ids,
        recipe_hash=recipe_hash,
    )
    model.config.use_cache = False
    model.train()
    learning_rate = recipe["learning_rate"]
    max_steps = recipe["max_steps"]
    gradient_accumulation = recipe["gradient_accumulation_steps"]
    batch_size = recipe["batch_size"]
    max_length = recipe["max_length"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    examples = 0
    progress = ProgressReporter(f"Task Model {model_id} training", max_steps, unit="steps")
    def make_batch(batch_records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences: list[list[int]] = []
        sequence_labels: list[list[int]] = []
        for record in batch_records:
            target_ids = tokenizer(
                _label_for_target(record["target"]), add_special_tokens=False
            )["input_ids"]
            prompt_ids = tokenizer(
                prompt_for(record), add_special_tokens=False, truncation=True,
                max_length=max(1, max_length - len(target_ids)),
            )["input_ids"]
            sequence = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            sequences.append(sequence)
            sequence_labels.append(labels)
        width = max(len(sequence) for sequence in sequences)
        pad_id = int(tokenizer.pad_token_id)
        input_values = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
        label_values = [labels + [-100] * (width - len(labels)) for labels in sequence_labels]
        attention_values = [
            [1] * len(sequence) + [0] * (width - len(sequence))
            for sequence in sequences
        ]
        return {
            "input_ids": torch.tensor(input_values, device=target),
            "attention_mask": torch.tensor(attention_values, device=target),
            "labels": torch.tensor(label_values, device=target),
        }

    while step < max_steps:
        order = torch.randperm(len(records), generator=generator).tolist()
        for offset in range(0, len(order), batch_size):
            batch_records = [records[index] for index in order[offset : offset + batch_size]]
            batch = make_batch(batch_records)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=target.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                loss = model(**batch).loss / gradient_accumulation
            loss.backward()
            examples += len(batch_records)
            if (offset // batch_size + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.report(step)
                if step >= max_steps:
                    break
        if not records:
            break
    if step < max_steps and examples:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    progress.finish()
    print(f"[LEGO] Task Model {model_id}: saving LoRA adapter...", file=sys.stderr, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.eval()
    elapsed = time.perf_counter() - start
    content_hash = hash_directory(output_dir)
    checkpoint = {
        "backend": "qwen-lora-preference-v1",
        "base_model_id": model_id,
        "base_model_path": base_path,
        "base_model_revision": str(config.get("base_model_revision", "unrecorded")),
        "adapter_path": str(output_dir.resolve()),
        "adapter_hash": content_hash,
        "task_model_version": f"qwen-lora-{content_hash[:16]}",
        "parent_checkpoint": parent_checkpoint,
        "training_recipe_hash": recipe_hash,
        "training_recipe": recipe,
        "label_ids": ids,
        "label_map": LABEL_TO_TEXT,
        "seed": seed,
        "deterministic": deterministic,
        "training_steps": step,
        "max_length": max_length,
    }
    return checkpoint, {
        "training_steps": step,
        "training_records_seen": examples,
        "training_seconds": elapsed,
    }


def predict(
    checkpoint: dict[str, Any], records: list[dict[str, Any]], *, batch_size: int = 8
) -> list[str]:
    target = device()
    _configure_memory_limit(target)
    print(f"[LEGO] Qwen evaluator: loading checkpoint on {target}...", file=sys.stderr, flush=True)
    tokenizer, model, ids = load_adapter_checkpoint(checkpoint, target)
    labels = list(LABEL_TO_TEXT)
    token_ids = torch.tensor([ids[label] for label in labels], device=target)
    predictions: list[str] = []
    progress = ProgressReporter("Qwen evaluator", len(records), unit="records")
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch_records = records[offset : offset + batch_size]
            encoded = tokenizer(
                [prompt_for(record) for record in batch_records],
                return_tensors="pt", padding=True, truncation=True,
                max_length=int(checkpoint.get("max_length", 1024)),
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            logits = model(**encoded).logits[:, -1, :]
            scores = logits.index_select(1, token_ids)
            predictions.extend(labels[int(index)] for index in scores.argmax(dim=1).tolist())
            progress.report(len(predictions))
    progress.finish()
    return predictions


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()
