"""Generate and collect a separate human blind comparison on final-test inputs."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.evaluation.qwen_sft import format_prompt


PREFERENCES = {
    "1": "A",
    "2": "B",
    "3": "both_acceptable",
    "4": "neither_acceptable",
    "5": "cannot_judge",
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
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def select_final_test_prompts(
    records: list[dict[str, Any]], *, max_items: int, seed: int
) -> list[dict[str, Any]]:
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if not records or any(row.get("split") != "final_test" for row in records):
        raise ValueError("blind evaluation requires only split=final_test records")
    interactions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        user = str(row.get("source_group_id") or "")
        conversation = str(row.get("conversation_id") or "")
        interaction = str(row.get("interaction_id") or row.get("record_id") or "")
        if not user or not interaction:
            raise ValueError("final-test rows need source_group_id and interaction_id")
        key = (user, conversation, interaction)
        if key not in interactions:
            input_text = str(row.get("input") or "").strip()
            if not input_text:
                continue
            interactions[key] = {
                "record_id": str(row.get("record_id") or ""),
                "source_group_id": user,
                "conversation_id": conversation,
                "interaction_id": interaction,
                "input": input_text,
                "context": row.get("context") or [],
                "split": "final_test",
            }
    prompts = list(interactions.values())
    if len(prompts) < max_items:
        raise ValueError(f"requested {max_items} final-test prompts but only {len(prompts)} are available")
    random.Random(seed).shuffle(prompts)
    return prompts[:max_items]


def _generate_with_adapter(
    prompts: list[dict[str, Any]],
    *,
    base_model: str,
    model_revision: str,
    adapter_path: str | Path,
    max_new_tokens: int,
    require_cuda: bool = True,
) -> dict[str, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("final blind generation requires CUDA")
    if max_new_tokens <= 0 or max_new_tokens >= 4096:
        raise ValueError("max_new_tokens must be between 1 and 4095")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, revision=model_revision, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device)
    model.eval()
    results = {}
    for record in prompts:
        prompt = format_prompt(record, tokenizer)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - max_new_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError(f"model generated an empty final-test response for {record['record_id']}")
        results[str(record["record_id"])] = text
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def prepare_blind_pairwise_queue(
    *,
    final_test_path: str | Path,
    adapter_a: str | Path,
    adapter_b: str | Path,
    system_a: str,
    system_b: str,
    output_dir: str | Path,
    max_items: int,
    seed: int,
    base_model: str,
    model_revision: str,
    max_new_tokens: int = 256,
    require_cuda: bool = True,
    generator: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    source = Path(final_test_path)
    if "final_test" not in source.name.casefold() and "final-test" not in source.name.casefold():
        raise ValueError("final evaluation input path must explicitly identify final_test")
    if system_a == system_b or not system_a or not system_b:
        raise ValueError("two distinct system names are required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"blind evaluation output directory must be empty: {destination}")
    prompts = select_final_test_prompts(_read_jsonl(source), max_items=max_items, seed=seed)
    generator = generator or _generate_with_adapter
    outputs_a = generator(
        prompts,
        base_model=base_model,
        model_revision=model_revision,
        adapter_path=adapter_a,
        max_new_tokens=max_new_tokens,
        require_cuda=require_cuda,
    )
    outputs_b = generator(
        prompts,
        base_model=base_model,
        model_revision=model_revision,
        adapter_path=adapter_b,
        max_new_tokens=max_new_tokens,
        require_cuda=require_cuda,
    )
    randomizer = random.Random(seed + 1)
    queue, key = [], []
    for record in prompts:
        record_id = str(record["record_id"])
        if record_id not in outputs_a or record_id not in outputs_b:
            raise ValueError(f"model output missing for final-test record {record_id}")
        query_id = hashlib.sha256(f"{seed}:{record_id}:{system_a}:{system_b}".encode()).hexdigest()[:24]
        if randomizer.randrange(2):
            response_a, response_b = outputs_b[record_id], outputs_a[record_id]
            key_a, key_b = system_b, system_a
        else:
            response_a, response_b = outputs_a[record_id], outputs_b[record_id]
            key_a, key_b = system_a, system_b
        queue.append(
            {
                "schema_version": 1,
                "query_id": query_id,
                "source_group_id": record["source_group_id"],
                "input": record["input"],
                "context": record["context"],
                "candidate_a": response_a,
                "candidate_b": response_b,
            }
        )
        key.append(
            {
                "query_id": query_id,
                "record_id": record_id,
                "source_group_id": record["source_group_id"],
                "system_a": key_a,
                "system_b": key_b,
            }
        )
    _write_jsonl(destination / "blind-queue.jsonl", queue)
    _write_jsonl(destination / "blind-key.jsonl", key)
    manifest = {
        "status": "ready_for_blind_human_evaluation",
        "final_test_accessed": True,
        "final_test_path_name": source.name,
        "final_test_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_seed": seed,
        "sampled_prompts": len(prompts),
        "sampled_user_groups": len({row["source_group_id"] for row in prompts}),
        "systems": sorted({system_a, system_b}),
        "labels_read": False,
        "queue_path": str(destination / "blind-queue.jsonl"),
        "key_path": str(destination / "blind-key.jsonl"),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _show_pair(query: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    for turn in query.get("context") or []:
        if isinstance(turn, dict):
            print(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
        else:
            print(f"对话: {turn}")
    print(f"输入：{query.get('input', '')}")
    print(f"A：{query.get('candidate_a', '')}")
    print(f"B：{query.get('candidate_b', '')}")
    print("=" * 72)


def collect_blind_pairwise_feedback(
    queue_path: str | Path,
    *,
    output_path: str | Path,
    annotator_id: str,
    input_fn: Callable[[str], str] = input,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    if not annotator_id.strip():
        raise ValueError("annotator_id must be non-empty")
    queue = _read_jsonl(Path(queue_path))
    queue_by_id = {str(row.get("query_id") or ""): row for row in queue}
    if "" in queue_by_id or len(queue_by_id) != len(queue):
        raise ValueError("blind queue must have unique non-empty query IDs")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events = _read_jsonl(destination) if destination.exists() else []
    existing_ids = set()
    for event in events:
        query_id = str(event.get("query_id") or "")
        duration = event.get("duration_seconds")
        if (
            event.get("evaluation_source") != "final_test_blind_human"
            or event.get("annotator_id") != annotator_id.strip()
            or query_id not in queue_by_id
            or event.get("preference") not in set(PREFERENCES.values())
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
            or query_id in existing_ids
        ):
            raise ValueError("existing blind-evaluation file is invalid or belongs to another queue/reviewer")
        existing_ids.add(query_id)
    for query in queue:
        if str(query["query_id"]) in existing_ids:
            continue
        start = clock()
        started_at = datetime.now(timezone.utc).isoformat()
        _show_pair(query)
        while True:
            choice = input_fn(
                "评价：1 A 更好，2 B 更好，3 都可以，4 都不好，5 无法判断 [1/2/3/4/5]: "
            ).strip()
            if choice in PREFERENCES:
                break
            print("请输入 1 到 5。")
        comment = input_fn("可选：评价原因，留空跳过： ").strip()
        duration = max(0.0, clock() - start)
        event = {
            "schema_version": 1,
            "evaluation_source": "final_test_blind_human",
            "query_id": query["query_id"],
            "source_group_id": query["source_group_id"],
            "annotator_id": annotator_id.strip(),
            "preference": PREFERENCES[choice],
            "comment": comment or None,
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
        }
        events.append(event)
        existing_ids.add(str(query["query_id"]))
        with destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    return events


def analyze_blind_pairwise_feedback(
    feedback_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]], *, seed: int = 17,
    expected_source: str = "final_test_blind_human",
) -> dict[str, Any]:
    key_by_query = {str(row.get("query_id") or ""): row for row in key_rows}
    if len(key_by_query) != len(key_rows):
        raise ValueError("blind key has duplicate query IDs")
    system_names = sorted({str(row[name]) for row in key_rows for name in ("system_a", "system_b")})
    if len(system_names) != 2:
        raise ValueError("blind key must map exactly two systems")
    seen = set()
    ratings_by_query: dict[str, list[str]] = defaultdict(list)
    outcomes = {name: {"wins": 0, "losses": 0} for name in system_names}
    category_counts = {name: 0 for name in PREFERENCES.values()}
    group_results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total_seconds = 0.0
    for event in feedback_rows:
        if event.get("evaluation_source") != expected_source:
            raise ValueError("feedback file contains non-final-test evaluations")
        query_id = str(event.get("query_id") or "")
        identity = (query_id, str(event.get("annotator_id") or ""))
        if identity in seen:
            raise ValueError("duplicate blind evaluation from the same annotator and query")
        seen.add(identity)
        key = key_by_query.get(query_id)
        if key is None:
            raise ValueError(f"blind key is missing query {query_id}")
        if event.get("source_group_id") != key.get("source_group_id"):
            raise ValueError(f"blind evaluation source group does not match query {query_id}")
        preference = event.get("preference")
        if preference not in category_counts:
            raise ValueError(f"unknown preference label: {preference}")
        category_counts[preference] += 1
        if preference != "cannot_judge":
            ratings_by_query[query_id].append(str(preference))
        duration = event.get("duration_seconds")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("blind evaluation contains an invalid measured duration")
        total_seconds += float(duration)
        if preference not in {"A", "B"}:
            continue
        winner = str(key["system_a"] if preference == "A" else key["system_b"])
        loser = str(key["system_b"] if preference == "A" else key["system_a"])
        outcomes[winner]["wins"] += 1
        outcomes[loser]["losses"] += 1
        group = str(key.get("source_group_id") or event.get("source_group_id") or "")
        group_results[group][0 if winner == system_names[0] else 1] += 1

    decisive = sum(row["wins"] for row in outcomes.values())
    wins = outcomes[system_names[0]]["wins"]
    group_rates = []
    for values in group_results.values():
        total = sum(values)
        if total:
            group_rates.append(values[0] / total)
    interval = _cluster_bootstrap_mean(group_rates, seed=seed) if len(group_rates) >= 2 else None
    interrater = _interrater_agreement(ratings_by_query)
    return {
        "status": "analyzed",
        "evaluation_source": expected_source,
        "final_test_accessed": True,
        "labels_used": False,
        "systems": outcomes,
        "preference_counts": category_counts,
        "decisive_comparisons": decisive,
        "system_0_decisive_win_rate": wins / decisive if decisive else None,
        "user_groups_with_decisive_votes": len(group_rates),
        "user_group_macro_win_rate_ci95_bootstrap": interval,
        "feedback_seconds": total_seconds,
        "annotator_count": len({str(row.get("annotator_id") or "") for row in feedback_rows}),
        "inter_annotator_agreement": interrater,
    }


def _interrater_agreement(ratings_by_query: dict[str, list[str]]) -> dict[str, Any]:
    comparable = [values for values in ratings_by_query.values() if len(values) >= 2]
    pairs = [
        (left, right)
        for values in comparable
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ]
    raw_agreement = (
        sum(left == right for left, right in pairs) / len(pairs) if pairs else None
    )
    counts: dict[str, int] = defaultdict(int)
    observed_disagreement = 0.0
    coincidence_total = 0
    for values in comparable:
        for value in values:
            counts[value] += 1
        mismatched_pairs = sum(
            left != right
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        )
        observed_disagreement += 2.0 * mismatched_pairs / (len(values) - 1)
        coincidence_total += len(values)

    total_ratings = sum(counts.values())
    expected_disagreement = None
    alpha = None
    if total_ratings > 1:
        expected_disagreement = 1.0 - sum(
            count * (count - 1) for count in counts.values()
        ) / (total_ratings * (total_ratings - 1))
        if expected_disagreement > 0 and coincidence_total:
            alpha = 1.0 - (observed_disagreement / coincidence_total) / expected_disagreement
    return {
        "method": "nominal Krippendorff alpha; cannot_judge excluded",
        "queries_with_multiple_ratings": len(comparable),
        "pairwise_rating_count": len(pairs),
        "pairwise_raw_agreement": raw_agreement,
        "krippendorff_alpha_nominal": alpha,
        "ratings_in_alpha": total_ratings,
    }


def _cluster_bootstrap_mean(values: list[float], *, seed: int, samples: int = 2000) -> list[float]:
    randomizer = random.Random(seed)
    means = []
    for _ in range(samples):
        selected = [values[randomizer.randrange(len(values))] for _ in values]
        means.append(sum(selected) / len(selected))
    means.sort()
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]
