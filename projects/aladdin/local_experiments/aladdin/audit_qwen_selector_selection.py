from __future__ import annotations

"""Audit selected HH-RLHF groups without retraining or new LLM judgments."""

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from local_experiments.aladdin.hh_rlhf_qwen_lora_label_diagnostic import (
    QWEN_ALADDIN,
    _sample_groups,
)
from local_experiments.aladdin.hh_rlhf_feedback import (
    _groups_for_split_seed,
    load_hh_rlhf_groups,
    split_audit,
)
from local_experiments.aladdin.hh_rlhf_qwen_lora_pilot import (
    _record_preference_margins,
)
from local_experiments.aladdin.helpsteer2_feedback import (
    _group_novelty_tokens,
    _tokens,
)
from local_experiments.aladdin.qwen_preference import (
    _configure_memory_limit,
    device,
    hash_directory,
    label_ids,
    load_tokenizer,
)

BASELINE = "random"
STRATEGY = QWEN_ALADDIN
T95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
    8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
    14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
    20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
    26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}
METRICS = (
    "llm_feedback_accuracy",
    "selector_pairwise_accuracy_vs_official_label",
    "mean_absolute_selector_margin",
    "mean_selector_uncertainty",
    "unique_prompt_rate",
    "mean_pairwise_prompt_jaccard",
    "mean_novelty_to_seed_prompts",
    "mean_chosen_response_words",
    "mean_rejected_response_words",
)


def _response_text(record: Any) -> str:
    return str(record.input).split("\n\nResponse: ", 1)[-1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_metrics(
    groups: list[Any],
    judgments: list[dict[str, Any]],
    seed_prompts: list[str],
    group_margins: dict[str, float],
    *,
    include_group_audit: bool = True,
) -> dict[str, Any]:
    judgment_by_id = {item["group_id"]: item for item in judgments}
    tokens = [_tokens(group.prompt) for group in groups]
    similarities = []
    for index, left in enumerate(tokens):
        for right in tokens[index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 0.0)
    seed_token_sets = [_tokens(prompt) for prompt in seed_prompts]
    chosen_words = []
    rejected_words = []
    group_audit = []
    for group in groups:
        by_candidate = {
            str(record.record_id).rsplit("-", 1)[-1]: record
            for record in group.hidden_records
        }
        chosen_words.append(len(_response_text(by_candidate["chosen"]).split()))
        rejected_words.append(len(_response_text(by_candidate["rejected"]).split()))
        judgment = judgment_by_id.get(group.group_id)
        margin = group_margins[group.group_id]
        if include_group_audit:
            group_audit.append(
                {
                    "group_id": group.group_id,
                    "selector_margin_chosen_minus_rejected": margin,
                    "selector_uncertainty": 1.0 / (1.0 + abs(margin)),
                    "selector_predicts_official_preference": margin > 0,
                    "llm_feedback_correct": (
                        bool(judgment["correct"]) if judgment is not None else None
                    ),
                    "prompt_word_count": len(group.prompt.split()),
                    "chosen_response_word_count": len(_response_text(by_candidate["chosen"]).split()),
                    "rejected_response_word_count": len(_response_text(by_candidate["rejected"]).split()),
                }
            )
    llm_accuracy = (
        mean(float(judgment_by_id[group.group_id]["correct"]) for group in groups)
        if groups and all(group.group_id in judgment_by_id for group in groups)
        else None
    )
    return {
        "selected_groups": len(groups),
        "llm_feedback_accuracy": llm_accuracy,
        "selector_pairwise_accuracy_vs_official_label": (
            mean(float(group_margins[group.group_id] > 0) for group in groups)
            if groups
            else None
        ),
        "mean_absolute_selector_margin": (
            mean(abs(group_margins[group.group_id]) for group in groups) if groups else None
        ),
        "mean_selector_uncertainty": (
            mean(1.0 / (1.0 + abs(group_margins[group.group_id])) for group in groups)
            if groups
            else None
        ),
        "unique_prompts": len({group.prompt.strip() for group in groups}),
        "unique_prompt_rate": (
            len({group.prompt.strip() for group in groups}) / len(groups) if groups else None
        ),
        "mean_pairwise_prompt_jaccard": mean(similarities) if similarities else 0.0,
        "mean_novelty_to_seed_prompts": (
            mean(
                _group_novelty_tokens(prompt_tokens, seed_token_sets)
                for prompt_tokens in tokens
            )
            if groups
            else None
        ),
        "mean_chosen_response_words": mean(chosen_words) if chosen_words else None,
        "mean_rejected_response_words": mean(rejected_words) if rejected_words else None,
        "selected_group_audit": group_audit if include_group_audit else None,
    }


def _paired_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    average = mean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": average, "ci95_low": None, "ci95_high": None}
    half_width = T95.get(len(values), 1.96) * stdev(values) / sqrt(len(values))
    return {
        "n": len(values),
        "mean": average,
        "ci95_low": average - half_width,
        "ci95_high": average + half_width,
    }


def audit(report_path: Path, data_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_data_hash = _sha256_file(data_path)
    expected_data_hash = report.get("data_sha256")
    if expected_data_hash and actual_data_hash != expected_data_hash:
        raise ValueError(
            f"data hash mismatch: report={expected_data_hash}, input={actual_data_hash}"
        )
    configuration = report["configuration"]
    split_unit = configuration.get("split_unit", "prompt")
    split_seed = int(configuration["split_seed"])
    groups = load_hh_rlhf_groups(
        data_path, split_seed=split_seed, split_unit=split_unit
    )
    adapters_root = report_path.parent / f"{report_path.stem}-adapters"
    base_model_path = report["base_model_path"]
    training = configuration["training"]
    batch_size = int(configuration.get("eval_batch_size", 8))
    _configure_memory_limit(device(), training)

    rows_by_seed_strategy = {
        (int(row["seed"]), row["strategy"]): row for row in report.get("results", [])
    }
    seeds = sorted({seed for seed, strategy in rows_by_seed_strategy if strategy == STRATEGY})
    if not seeds:
        raise ValueError(f"report contains no {STRATEGY} result rows")

    output: dict[str, Any] = {
        "source_report": str(report_path.resolve()),
        "source_status": report.get("status"),
        "dataset": report.get("dataset"),
        "dataset_revision": report.get("dataset_revision"),
        "data_sha256": actual_data_hash,
        "split_protocol": {
            "unit": split_unit,
            "base_split_seed": split_seed,
            "split_seeds_by_run_seed": {},
        },
        "interpretation": {
            "feedback_accuracy": "LLM judgments matching the dataset's binary chosen/rejected label among selected groups.",
            "selector_margin": "Qwen seed-trained chosen-minus-rejected logit margin; model-relative, not an oracle preference-strength score.",
            "oracle_preference_strength": "Unavailable: HH-RLHF supplies binary pairwise labels, not a continuous strength or annotator-agreement score.",
            "inference_scope": "Paired variation across runs on the report's fixed split; intervals do not include split-sampling uncertainty.",
        },
        "by_seed": {},
    }

    for seed in seeds:
        aladdin_row = rows_by_seed_strategy[(seed, STRATEGY)]
        random_row = rows_by_seed_strategy.get((seed, BASELINE))
        if random_row is None:
            raise ValueError(f"report has no random result for seed={seed}")
        run_split_seed = int(aladdin_row.get("split_seed", split_seed))
        output["split_protocol"]["split_seeds_by_run_seed"][str(seed)] = run_split_seed
        run_groups = _groups_for_split_seed(groups, run_split_seed, split_unit)
        audit_counts = split_audit(run_groups)
        if configuration.get("fail_on_split_overlap") and audit_counts["prompts_with_multiple_splits"]:
            raise ValueError(
                f"prompt overlap detected while reconstructing seed={seed}: {audit_counts}"
            )
        seed_groups = _sample_groups(
            [group for group in run_groups if group.split == "seed"],
            int(configuration["seed_group_limit"]),
            run_split_seed + 1103,
        )
        pool_groups = _sample_groups(
            [group for group in run_groups if group.split == "held_in"],
            int(configuration["pool_group_limit"]),
            run_split_seed + 2207,
        )
        pool_by_id = {group.group_id: group for group in pool_groups}
        chosen_ids = {
            STRATEGY: list(aladdin_row["selected_group_ids"]),
            BASELINE: list(random_row["selected_group_ids"]),
        }
        if any(group_id not in pool_by_id for ids in chosen_ids.values() for group_id in ids):
            raise ValueError(f"selection references a group outside the reconstructed pool for seed={seed}")
        if len(chosen_ids[STRATEGY]) != len(chosen_ids[BASELINE]):
            raise ValueError(f"selection budgets differ for seed={seed}")
        for strategy, ids in chosen_ids.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate selected group IDs for seed={seed}, strategy={strategy}")
        expected_budget = int(configuration["candidate_budget"])
        if len(chosen_ids[STRATEGY]) != expected_budget:
            raise ValueError(
                f"selection budget mismatch for seed={seed}: "
                f"expected={expected_budget}, observed={len(chosen_ids[STRATEGY])}"
            )

        adapter_path = adapters_root / f"{STRATEGY}-selector-seed{seed}"
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"selector adapter not found: {adapter_path}")
        selector_hash = hash_directory(adapter_path)
        expected_selector_hash = (
            aladdin_row.get("diagnostics", {}).get("selector_checkpoint", {}).get("adapter_hash")
        )
        if expected_selector_hash and selector_hash != expected_selector_hash:
            raise ValueError(
                f"selector adapter hash mismatch for seed={seed}: "
                f"report={expected_selector_hash}, adapter={selector_hash}"
            )
        tokenizer = load_tokenizer(str(adapter_path))
        checkpoint = {
            "base_model_path": base_model_path,
            "adapter_path": str(adapter_path),
            "label_ids": label_ids(tokenizer),
            "max_length": int(training["max_length"]),
        }
        del tokenizer
        judgment_ids = {
            strategy: {item["group_id"] for item in row.get("judgments", [])}
            for strategy, row in ((STRATEGY, aladdin_row), (BASELINE, random_row))
        }
        for strategy in (STRATEGY, BASELINE):
            if judgment_ids[strategy] != set(chosen_ids[strategy]):
                raise ValueError(
                    f"judgments do not match selected IDs for seed={seed}, strategy={strategy}"
                )
        selected_union_ids = set(chosen_ids[STRATEGY]) | set(chosen_ids[BASELINE])
        selected_union = [pool_by_id[group_id] for group_id in sorted(selected_union_ids)]
        pool_records = [record for group in selected_union for record in group.hidden_records]
        print(
            f"auditing seed={seed}: scoring {len(selected_union)} unique selected groups "
            f"({len(pool_records)} response records) with the saved selector adapter",
            flush=True,
        )
        record_margins = _record_preference_margins(
            checkpoint, pool_records, batch_size=batch_size
        )
        group_margins: dict[str, float] = {}
        for group in selected_union:
            candidate_records = {
                str(record.record_id).rsplit("-", 1)[-1]: record
                for record in group.hidden_records
            }
            group_margins[group.group_id] = (
                record_margins[candidate_records["chosen"].record_id]
                - record_margins[candidate_records["rejected"].record_id]
            )

        seed_prompts = [group.prompt for group in seed_groups]
        seed_report_rows = {}
        for strategy, row in ((STRATEGY, aladdin_row), (BASELINE, random_row)):
            selected = [pool_by_id[group_id] for group_id in chosen_ids[strategy]]
            seed_report_rows[strategy] = _selected_metrics(
                selected, row.get("judgments", []), seed_prompts, group_margins
            )
        output["by_seed"][str(seed)] = {
            "selector_adapter_hash": selector_hash,
            "reconstructed_split_audit": audit_counts,
            "selection_overlap_groups": len(set(chosen_ids[STRATEGY]) & set(chosen_ids[BASELINE])),
            "selected": seed_report_rows,
            "existing_downstream_pairwise_accuracy": {
                strategy: {
                    channel: row.get("metrics", {}).get(channel, {}).get("pairwise_accuracy")
                    for channel in ("llm_feedback", "oracle_feedback")
                }
                for strategy, row in ((STRATEGY, aladdin_row), (BASELINE, random_row))
            },
            "delta_aladdin_qwen_minus_random": {
                metric: (
                    seed_report_rows[STRATEGY][metric] - seed_report_rows[BASELINE][metric]
                    if seed_report_rows[STRATEGY][metric] is not None
                    and seed_report_rows[BASELINE][metric] is not None
                    else None
                )
                for metric in METRICS
            },
            "existing_downstream_delta_aladdin_qwen_minus_random": {
                channel: (
                    aladdin_row.get("metrics", {}).get(channel, {}).get("pairwise_accuracy")
                    - random_row.get("metrics", {}).get(channel, {}).get("pairwise_accuracy")
                    if aladdin_row.get("metrics", {}).get(channel, {}).get("pairwise_accuracy") is not None
                    and random_row.get("metrics", {}).get(channel, {}).get("pairwise_accuracy") is not None
                    else None
                )
                for channel in ("llm_feedback", "oracle_feedback")
            },
        }
        print(
            f"seed={seed} overlap={output['by_seed'][str(seed)]['selection_overlap_groups']} "
            f"accuracy={seed_report_rows[STRATEGY]['llm_feedback_accuracy']:.3f}/"
            f"{seed_report_rows[BASELINE]['llm_feedback_accuracy']:.3f} "
            f"|margin={seed_report_rows[STRATEGY]['mean_absolute_selector_margin']:.3f}/"
            f"{seed_report_rows[BASELINE]['mean_absolute_selector_margin']:.3f} "
            f"|unique_prompt_rate={seed_report_rows[STRATEGY]['unique_prompt_rate']:.3f}/"
            f"{seed_report_rows[BASELINE]['unique_prompt_rate']:.3f}",
            flush=True,
        )

    paired: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            float(seed["delta_aladdin_qwen_minus_random"][metric])
            for seed in output["by_seed"].values()
            if seed["delta_aladdin_qwen_minus_random"][metric] is not None
        ]
        paired[metric] = _paired_summary(values)
    output["paired_differences_aladdin_qwen_minus_random"] = paired
    output["status"] = (
        "completed"
        if report.get("status") == "completed"
        else "partial_source_report"
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output_path)
        print(f"wrote selection audit: {output_path}", flush=True)
    print("\nPaired differences (aladdin-qwen minus random; percentage points where applicable):")
    for metric, summary in paired.items():
        if summary["mean"] is None:
            print(f"{metric}: no paired data")
            continue
        if summary["ci95_low"] is None:
            print(f"{metric}: n={summary['n']} mean={summary['mean']:.4f}")
            continue
        scale = (
            100.0
            if metric in {
                "llm_feedback_accuracy",
                "selector_pairwise_accuracy_vs_official_label",
            }
            else 1.0
        )
        unit = " pp" if scale == 100.0 else ""
        print(
            f"{metric}: n={summary['n']} mean={summary['mean'] * scale:+.3f}{unit} "
            f"95% CI=[{summary['ci95_low'] * scale:+.3f}, "
            f"{summary['ci95_high'] * scale:+.3f}]{unit}"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.report, args.data, args.output)


if __name__ == "__main__":
    main()
