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

# Some NumPy 1.x builds expose this module only after explicit import, while
# accelerate accesses it through ``numpy._core.multiarray`` during import.
try:
    import numpy._core.multiarray  # noqa: F401
except ImportError:
    pass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .progress import ProgressReporter


LABEL_TO_TEXT = {"A": "preferred", "B": "rejected", "C": "tie"}
TEXT_TO_LABEL = {value: key for key, value in LABEL_TO_TEXT.items()}


def base_model_path(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    value = config.get("base_model_path") or os.environ.get("ALADDIN_TASK_MODEL_BASE_PATH")
    if not isinstance(value, str) or not value or value.startswith("SET_"):
        raise ValueError(
            "Set task_model.training_config.base_model_path or "
            "ALADDIN_TASK_MODEL_BASE_PATH to a local Qwen model directory"
        )
    return value


def device() -> torch.device:
    requested = os.environ.get("ALADDIN_TASK_MODEL_DEVICE", "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _dtype(target: torch.device) -> torch.dtype:
    if target.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _configure_memory_limit(target: torch.device, config: dict[str, Any] | None = None) -> None:
    """Keep the Aladdin process below its shared-GPU budget before allocation."""
    if target.type != "cuda":
        return
    config = config or {}
    raw_limit = config.get("max_gpu_memory_gib") or os.environ.get(
        "ALADDIN_TASK_MODEL_MAX_GPU_MEMORY_GIB"
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
        f"[ALADDIN] CUDA memory limit: {limit_gib:.1f} GiB "
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


def train_checkpoint(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, int | float]]:
    deterministic = bool(
        config.get("deterministic", os.environ.get("ALADDIN_DETERMINISTIC", "0") == "1")
    )
    _configure_reproducibility(seed, deterministic=deterministic)
    target = device()
    _configure_memory_limit(target, config)
    base_path = base_model_path(config)
    print(f"[ALADDIN] Qwen Task Model: loading base model on {target}...", file=sys.stderr, flush=True)
    tokenizer = load_tokenizer(base_path)
    tokenizer.padding_side = "right"
    ids = label_ids(tokenizer)
    model = load_base(base_path, target)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config.get("lora_r", 8)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=list(config.get("target_modules", ["q_proj", "v_proj"])),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    learning_rate = float(config.get("learning_rate", 2e-4))
    max_steps = int(config.get("max_steps", 100))
    gradient_accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    batch_size = max(1, int(config.get("per_device_batch_size", config.get("batch_size", 4))))
    max_length = int(config.get("max_length", 1024))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    examples = 0
    progress = ProgressReporter("Qwen Task Model training", max_steps, unit="steps")
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
    print("[ALADDIN] Qwen Task Model: saving LoRA adapter...", file=sys.stderr, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.eval()
    elapsed = time.perf_counter() - start
    content_hash = hash_directory(output_dir)
    checkpoint = {
        "backend": "qwen-lora-preference-v1",
        "base_model_path": base_path,
        "adapter_path": str(output_dir.resolve()),
        "adapter_hash": content_hash,
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
    print(f"[ALADDIN] Qwen evaluator: loading checkpoint on {target}...", file=sys.stderr, flush=True)
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


def predict_scores(
    checkpoint: dict[str, Any], records: list[dict[str, Any]], *, batch_size: int = 8
) -> list[dict[str, float]]:
    """Return A/B/C logits so pairwise evaluation can match the TF-IDF metric."""
    target = device()
    _configure_memory_limit(target)
    tokenizer, model, ids = load_adapter_checkpoint(checkpoint, target)
    labels = list(LABEL_TO_TEXT)
    token_ids = torch.tensor([ids[label] for label in labels], device=target)
    scores: list[dict[str, float]] = []
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
            selected = logits.index_select(1, token_ids).detach().float().cpu()
            scores.extend(
                {label: float(row[index]) for index, label in enumerate(labels)}
                for row in selected
            )
    del model, tokenizer
    return scores


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()
