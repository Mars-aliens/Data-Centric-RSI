"""Small, auditable Qwen LoRA SFT and development evaluation runner.

The module imports torch/transformers lazily so dataset and protocol tests stay
CPU-only. It evaluates only the development records supplied by the caller;
the final-test split is deliberately rejected by the public entry point.
"""

from __future__ import annotations

import gc
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QwenSFTConfig:
    base_model: str
    model_revision: str | None = None
    train_token_budget: int = 100_000
    require_cuda: bool = True
    max_steps: int = 1_000
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_length: int = 1024
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    eval_batch_size: int = 8
    seed: int = 17
    save_adapter: str | None = None


def format_prompt(record: dict[str, Any], tokenizer: Any | None = None) -> str:
    """Format a record without exposing source labels to the task model."""
    context = record.get("context") or []
    if (
        tokenizer is not None
        and getattr(tokenizer, "chat_template", None)
        and callable(getattr(tokenizer, "apply_chat_template", None))
    ):
        messages = []
        for turn in context:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "user").lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": str(record.get("input") or "").strip()})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as exc:
            raise ValueError("task tokenizer chat template could not format the record") from exc
    lines = []
    for turn in context:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user").capitalize()
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"User: {str(record.get('input') or '').strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_training_schedule(
    records: list[dict[str, Any]], target_count: int, *, seed: int = 17
) -> list[dict[str, Any]]:
    """Choose exactly target_count records, cycling only when necessary.

    Equalizing the number of training examples is important for S0, which has
    no generated candidates, and the generated strategies.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not records:
        raise ValueError("at least one training record is required")
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        group = str(record.get("source_group_id") or record.get("record_id") or index)
        buckets[group].append(record)
    groups = sorted(buckets)
    rng.shuffle(groups)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    positions = {group: 0 for group in groups}
    selected = []
    while len(selected) < target_count:
        for group in groups:
            bucket = buckets[group]
            position = positions[group]
            if position >= len(bucket):
                rng.shuffle(bucket)
                position = 0
            selected.append(bucket[position])
            positions[group] = position + 1
            if len(selected) == target_count:
                break
    return selected


def build_1a_training_records(
    seeds: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    strategy: str,
    train_records: int,
    synthetic_train_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    if strategy == "seed-only":
        return build_training_schedule(seeds, train_records, seed=seed)
    if not 0 < synthetic_train_records < train_records:
        raise ValueError("synthetic_train_records must be positive and less than train_records")
    return build_training_schedule(
        seeds, train_records - synthetic_train_records, seed=seed
    ) + build_training_schedule(candidates, synthetic_train_records, seed=seed + 1)


def _make_examples(tokenizer: Any, records: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        prompt = format_prompt(record, tokenizer)
        output = str(record.get("output") or "").strip()
        if not prompt or not output:
            continue
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
            truncation=False,
        )["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            output_ids = output_ids + [eos_id]
        output_ids = output_ids[: max(1, max_length - 1)]
        available_prompt = max_length - len(output_ids)
        prompt_ids = prompt_ids[-available_prompt:]
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [0]
            output_ids = output_ids[: max_length - len(prompt_ids)]
        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids
        if any(label != -100 for label in labels):
            examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "record_id": str(record.get("record_id") or ""),
                }
            )
    if not examples:
        raise ValueError("no record contains a target response after tokenization")
    return examples


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, masks = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        masks.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def _model_commit(model: Any, base_model: str) -> str | None:
    """Find the resolved commit for a Hub load or a cached snapshot path."""
    candidates = [
        getattr(getattr(model, "config", None), "_commit_hash", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
        base_model,
    ]
    for candidate in candidates:
        match = re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", str(candidate or ""))
        if match:
            return match.group(0)
    return None


def _score_batch(model: Any, batch: dict[str, Any], device: Any) -> list[float]:
    import torch

    inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = inputs["labels"][:, 1:]
    mask = target.ne(-100)
    safe_target = target.masked_fill(~mask, 0)
    token_scores = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    token_scores = token_scores.masked_fill(~mask, 0.0)
    totals = token_scores.sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return (totals / counts).detach().cpu().tolist()


def pairwise_preference_metrics(
    scored_records: list[tuple[dict[str, Any], float]],
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[tuple[dict[str, Any], float]]] = {}
    for record, score in scored_records:
        key = tuple(
            str(record.get(field) or "")
            for field in ("source_group_id", "conversation_id", "interaction_id")
        )
        if not key[2]:
            key += (" ".join(str(record.get("input") or "").split()),)
        groups.setdefault(key, []).append((record, score))

    comparisons = 0
    credit = 0.0
    for items in groups.values():
        winners = [(row, score) for row, score in items if row.get("if_chosen") is True]
        losers = [(row, score) for row, score in items if row.get("if_chosen") is False]
        for _, winner_score in winners:
            for _, loser_score in losers:
                comparisons += 1
                credit += 1.0 if winner_score > loser_score else 0.5 if winner_score == loser_score else 0.0
    return {
        "pairwise_comparisons": comparisons,
        "pairwise_accuracy": credit / comparisons if comparisons else None,
    }


def evaluate_development(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    device: Any,
) -> dict[str, Any]:
    """Score development responses and compute chosen-vs-rejected accuracy."""
    examples = _make_examples(tokenizer, records, max_length)
    scores: list[float] = []
    for start in range(0, len(examples), batch_size):
        batch = _collate(examples[start : start + batch_size], tokenizer.pad_token_id)
        scores.extend(_score_batch(model, batch, device))

    # The example list and records differ only for malformed/overlong rows.
    scored_records = []
    record_by_id = {str(record.get("record_id") or ""): record for record in records}
    for example, score in zip(examples, scores):
        record = record_by_id.get(example["record_id"])
        if record and record.get("output") and str(record.get("input") or "").strip():
            scored_records.append((record, score))
    preference = pairwise_preference_metrics(scored_records)

    mean_logprob = sum(scores) / len(scores) if scores else float("nan")
    return {
        "scored_records": len(scores),
        "mean_target_logprob": mean_logprob,
        "mean_target_nll": -mean_logprob if not math.isnan(mean_logprob) else float("nan"),
        **preference,
    }


def train_and_evaluate(
    train_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    config: QwenSFTConfig,
) -> dict[str, Any]:
    """Train one LoRA adapter and evaluate it on development records."""
    if not train_records or not development_records:
        raise ValueError("training and development records must be non-empty")
    if config.train_token_budget <= 0 or config.max_steps <= 0 or config.batch_size <= 0:
        raise ValueError("train_token_budget, max_steps, and batch_size must be positive")
    if config.gradient_accumulation_steps <= 0 or config.eval_batch_size <= 0:
        raise ValueError("gradient_accumulation_steps and eval_batch_size must be positive")
    if config.max_length < 2:
        raise ValueError("max_length must be at least 2")

    import random

    random.seed(config.seed)
    import torch
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("experiment 1A Qwen training requires CUDA; select a GPU or explicitly allow CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, use_fast=True, revision=config.model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=dtype, revision=config.model_revision
    )
    if config.model_revision and re.fullmatch(r"[0-9a-f]{40}", config.model_revision):
        resolved_commit = _model_commit(model, config.base_model)
        if resolved_commit != config.model_revision:
            raise RuntimeError(
                "loaded task-model commit does not match the frozen model revision: "
                f"expected {config.model_revision}, got {resolved_commit!r}"
            )
    model.to(device)
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    examples = _make_examples(tokenizer, train_records, config.max_length)
    loader = DataLoader(
        examples,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=lambda rows: _collate(rows, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    tokens_seen = 0
    supervised_tokens_seen = 0
    optimizer_steps = 0
    micro_batches_seen = 0
    accumulated_batches = 0
    while tokens_seen < config.train_token_budget and optimizer_steps < config.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = {key: value.to(device) for key, value in batch.items()}
        result = model(**inputs)
        result.loss.backward()
        micro_batches_seen += 1
        tokens_seen += int(inputs["attention_mask"].sum().item())
        supervised_tokens_seen += int(inputs["labels"].ne(-100).sum().item())
        accumulated_batches += 1
        if accumulated_batches >= config.gradient_accumulation_steps or tokens_seen >= config.train_token_budget:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_batches)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulated_batches = 0
        loss_total += float(result.loss.detach().cpu())

    if tokens_seen < config.train_token_budget:
        raise RuntimeError(
            f"training step safety cap reached at {tokens_seen} of {config.train_token_budget} tokens"
        )

    model.eval()
    metrics = evaluate_development(
        model,
        tokenizer,
        development_records,
        batch_size=config.eval_batch_size,
        max_length=config.max_length,
        device=device,
    )
    if config.save_adapter:
        model.save_pretrained(config.save_adapter)
        tokenizer.save_pretrained(config.save_adapter)
    elapsed = time.monotonic() - started
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    metrics.update(
        {
            "status": "completed",
            "training_steps": optimizer_steps,
            "training_token_budget": config.train_token_budget,
            "training_tokens_seen": tokens_seen,
            "training_token_overshoot": tokens_seen - config.train_token_budget,
            "supervised_training_tokens_seen": supervised_tokens_seen,
            "training_examples": len(train_records),
            "tokenized_training_examples": len(examples),
            "supervised_training_tokens": sum(
                sum(label != -100 for label in example["labels"]) for example in examples
            ),
            "mean_training_loss": loss_total / max(1, micro_batches_seen),
            "training_seconds": elapsed,
            "device": str(device),
            "device_name": device_name,
            "model_revision": config.model_revision,
            "resolved_model_commit": _model_commit(model, config.base_model),
            "model_architecture": getattr(model.config, "model_type", None),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "config": asdict(config),
        }
    )
    del model, optimizer, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics
