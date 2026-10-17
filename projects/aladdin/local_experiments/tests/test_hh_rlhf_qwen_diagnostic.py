from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from core_system.mle_core.models import TaskRecord
from local_experiments.aladdin.analyze_hh_rlhf_qwen_lora_label_diagnostic import (
    _validate_qwen_diagnostic_report,
)
from local_experiments.aladdin.finalize_hh_rlhf_qwen_audit import audit
from local_experiments.aladdin import hh_rlhf_qwen_lora_label_diagnostic as diagnostic
from local_experiments.aladdin.hh_rlhf_feedback import HHRLHFGroup
from local_experiments.aladdin.hh_rlhf_qwen_lora_label_diagnostic import (
    QWEN_ALADDIN_RELIABLE,
    _select_qwen_feedback_groups,
    _mark_failed,
    _training_steps,
)
from local_experiments.aladdin.hh_rlhf_llm_smoke import LLMSmokeError, _parse_choice


def make_group(index: int, split: str = "held_in") -> HHRLHFGroup:
    group_id = f"group-{index:03d}"
    prompt = f"Human: question {index}"
    hidden = tuple(
        TaskRecord(
            record_id=f"{group_id}-{candidate}",
            task_id="hh-rlhf-response-preference",
            split=split,
            input=f"{prompt}\n\nResponse: {candidate} answer",
            target=None,
            source_group_id=group_id,
            metadata={"feedback_group_id": group_id, "feedback_released": False},
        )
        for candidate in ("chosen", "rejected")
    )
    released = tuple(
        TaskRecord(
            record_id=f"{group_id}-{candidate}",
            task_id="hh-rlhf-response-preference",
            split=split,
            input=f"{prompt}\n\nResponse: {candidate} answer",
            target="preferred" if candidate == "chosen" else "rejected",
            source_group_id=group_id,
            metadata={"feedback_group_id": group_id, "feedback_released": True},
        )
        for candidate in ("chosen", "rejected")
    )
    return HHRLHFGroup(
        group_id=group_id,
        prompt=prompt,
        split=split,
        batch_id="batch",
        hidden_records=hidden,
        released_records=released,
        labels=("preferred", "rejected"),
    )


class QwenDiagnosticTests(unittest.TestCase):
    def test_reliable_selector_uses_hidden_records_and_unique_budget(self) -> None:
        groups = [make_group(index) for index in range(12)]
        seed_records = [make_group(100, "seed").hidden_records[0]]

        def fake_uncertainty(checkpoint, candidates, *, batch_size):
            self.assertIs(checkpoint, checkpoint_marker)
            self.assertEqual(batch_size, 4)
            self.assertTrue(all(record.target is None for group in candidates for record in group.hidden_records))
            return {group.group_id: float(index) for index, group in enumerate(candidates)}

        checkpoint_marker = {"adapter_hash": "hash", "training_steps": 3, "seed": 17}
        with patch(
            "local_experiments.aladdin.hh_rlhf_qwen_lora_label_diagnostic._pairwise_uncertainty",
            side_effect=fake_uncertainty,
        ):
            selected, diagnostics = _select_qwen_feedback_groups(
                groups,
                budget=5,
                seed_records=seed_records,
                selector_checkpoint=checkpoint_marker,
                eval_batch_size=4,
                strategy=QWEN_ALADDIN_RELIABLE,
            )

        self.assertEqual(len(selected), 5)
        self.assertEqual(len({group.group_id for group in selected}), 5)
        self.assertEqual(diagnostics["target_uncertainty_rank"], 0.35)
        self.assertEqual(diagnostics["selected_group_ids"], [group.group_id for group in selected])
        self.assertTrue(all(record.target is None for group in selected for record in group.hidden_records))

    def test_selector_budget_cannot_exceed_candidate_pool(self) -> None:
        groups = [make_group(index) for index in range(3)]
        with patch(
            "local_experiments.aladdin.hh_rlhf_qwen_lora_label_diagnostic._pairwise_uncertainty",
            return_value={group.group_id: 0.5 for group in groups},
        ):
            selected, _ = _select_qwen_feedback_groups(
                groups,
                budget=10,
                seed_records=[],
                selector_checkpoint={"adapter_hash": "h", "training_steps": 1, "seed": 1},
                eval_batch_size=2,
                strategy=QWEN_ALADDIN_RELIABLE,
            )
        self.assertEqual(len(selected), 3)

    def test_equal_epochs_scales_steps_by_record_count(self) -> None:
        args = Namespace(
            max_steps=100,
            seed_only_max_steps=None,
            feedback_max_steps=None,
            training_budget_mode="equal-epochs",
        )
        self.assertEqual(
            _training_steps(args, "random", record_count=96, reference_record_count=64),
            150,
        )
        self.assertEqual(
            _training_steps(args, "seed-only", record_count=64, reference_record_count=64),
            100,
        )
        args.training_budget_mode = "fixed-steps"
        self.assertEqual(
            _training_steps(args, "random", record_count=96, reference_record_count=64),
            100,
        )

    def test_duplicate_seed_or_strategy_is_rejected_before_training(self) -> None:
        args = Namespace(
            strategies=["random", "random"],
            seeds=[17],
            candidate_budget=1,
            max_steps=1,
            seed_only_max_steps=None,
            feedback_max_steps=None,
            eval_batch_size=1,
            batch_size=1,
            max_length=8,
            lora_r=1,
            lora_alpha=1,
            max_gpu_memory_gib=1.0,
            learning_rate=1e-4,
            lora_dropout=0.0,
            protocol_version="custom-diagnostic-v1",
        )
        with self.assertRaisesRegex(ValueError, "strategies must not contain duplicates"):
            diagnostic.run(args)

    def test_choice_parser_rejects_non_structured_explanations(self) -> None:
        self.assertEqual(_parse_choice('{"choice":"a"}'), "A")
        self.assertEqual(_parse_choice("```json\n{\"choice\": \"B\"}\n```"), "B")
        for content in ("", '{"choice":"C"}', "I choose A because..."):
            with self.assertRaises(LLMSmokeError):
                _parse_choice(content)

    def test_failed_run_is_recorded_without_overwriting_completed_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"status": "running", "results": [1]}), encoding="utf-8")
            _mark_failed(path, RuntimeError("mock failure"))
            failed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("mock failure", failed["error"])

            path.write_text(json.dumps({"status": "completed", "results": [1]}), encoding="utf-8")
            _mark_failed(path, RuntimeError("must not replace"))
            completed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertNotIn("error", completed)

    def test_completed_reliable_report_requires_full_matrix(self) -> None:
        strategies = ("seed-only", "random", "aladdin-qwen-reliable")
        seeds = tuple(range(17, 27))
        rows = []
        for strategy in strategies:
            for seed in seeds:
                selected = [] if strategy == "seed-only" else [f"group-{i:03d}" for i in range(32)]
                rows.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "selected_groups": len(selected),
                        "selected_group_ids": selected,
                        "judgments": [{"group_id": group_id} for group_id in selected],
                        "training_records_seen": {"llm_feedback": 128, "oracle_feedback": 128},
                        "metrics": {
                            "llm_feedback": {"pairwise_accuracy": 0.5},
                            "oracle_feedback": {"pairwise_accuracy": 0.5},
                        },
                    }
                )
        report = {
            "status": "completed",
            "base_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "configuration": {
                "llm_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "strategies": list(strategies),
                "seeds": list(seeds),
                "split_unit": "prompt",
                "fail_on_split_overlap": True,
                "candidate_budget": 32,
                "seed_group_limit": 64,
                "pool_group_limit": 256,
                "eval_group_limit": 512,
                "vary_split_by_seed": False,
                "training_schedule": {"budget_mode": "equal-epochs"},
                "split_audit_by_seed": {
                    str(seed): {"prompts_with_multiple_splits": 0} for seed in seeds
                },
            },
            "results": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            _validate_qwen_diagnostic_report(report, path)
            report["results"].pop()
            with self.assertRaises(ValueError):
                _validate_qwen_diagnostic_report(report, path)

    def test_mocked_run_writes_complete_report_incrementally(self) -> None:
        groups = [make_group(index, "seed") for index in range(64)]
        groups.extend(make_group(index, "held_in") for index in range(64, 320))
        groups.extend(make_group(index, "held_out") for index in range(320, 832))
        args = Namespace(
            protocol_version="custom-diagnostic-v1",
            strategies=["seed-only", "random", "aladdin-qwen-reliable"],
            seeds=list(range(17, 27)),
            candidate_budget=32,
            max_steps=2,
            seed_only_max_steps=None,
            feedback_max_steps=None,
            training_budget_mode="equal-epochs",
            data=None,
            output=None,
            base_model="/models/a09a35458c702b33eeacc393d103063234e8bc28",
            learning_rate=1e-4,
            batch_size=2,
            max_length=64,
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            max_gpu_memory_gib=8.0,
            split_seed=23,
            split_unit="prompt",
            fail_on_split_overlap=True,
            vary_split_by_seed=False,
            seed_group_limit=64,
            pool_group_limit=256,
            eval_group_limit=512,
            endpoint="http://mock/v1/chat/completions",
            model="mock-qwen",
            model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
            eval_batch_size=4,
            timeout_seconds=5.0,
        )

        def fake_train(records, *, config, seed, output_dir):
            return (
                {
                    "adapter_hash": f"{seed}-{len(records)}-{config['max_steps']}",
                    "training_steps": config["max_steps"],
                    "seed": seed,
                },
                {
                    "training_steps": config["max_steps"],
                    "training_records_seen": len(records) * config["max_steps"],
                    "training_seconds": 0.01,
                },
            )

        def fake_judgment(*, group, **kwargs):
            return {
                "group_id": group.group_id,
                "predicted_label": "chosen",
                "correct": True,
                "usage": {"total_tokens": 5},
            }

        def fake_qwen_selection(groups, *, budget, **kwargs):
            selected = list(groups[:budget])
            return selected, {
                "strategy": QWEN_ALADDIN_RELIABLE,
                "selected_group_ids": [group.group_id for group in selected],
                "selection_definition": "mock selector",
            }

        saved_snapshots = []
        real_save = diagnostic._save

        def capture_save(path, report):
            saved_snapshots.append((report["status"], len(report["results"])))
            real_save(path, report)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "input.jsonl"
            data_path.write_text("synthetic fixture\n", encoding="utf-8")
            output_path = root / "report.json"
            args.data = data_path
            args.output = output_path
            with (
                patch.object(diagnostic, "load_hh_rlhf_groups", return_value=groups),
                patch.object(diagnostic, "_groups_for_split_seed", side_effect=lambda source, *_: source),
                patch.object(diagnostic, "train_checkpoint", side_effect=fake_train),
                patch.object(diagnostic, "_pairwise_accuracy", return_value={"pairwise_accuracy": 0.5, "pairwise_groups": 10.0}),
                patch.object(diagnostic, "_judgment", side_effect=fake_judgment),
                patch.object(diagnostic, "_select_qwen_feedback_groups", side_effect=fake_qwen_selection),
                patch.object(diagnostic, "_save", side_effect=capture_save),
            ):
                report = diagnostic.run(args)

            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["results"]), 30)
            self.assertEqual(
                [(row["strategy"], row["seed"]) for row in report["results"][:3]],
                [("seed-only", 17), ("random", 17), ("aladdin-qwen-reliable", 17)],
            )
            self.assertEqual(saved_snapshots[-1], ("completed", 30))
            self.assertTrue(any(status == "running" and count == 30 for status, count in saved_snapshots))
            self.assertTrue(all(
                row["training_records_seen"][channel] is not None
                for row in report["results"]
                for channel in ("llm_feedback", "oracle_feedback")
            ))
            _validate_qwen_diagnostic_report(report, output_path)

    def test_final_audit_requires_four_distinct_completed_splits(self) -> None:
        strategies = ("seed-only", "random", "aladdin-qwen-reliable")
        seeds = tuple(range(17, 27))

        def make_report(split_seed: int) -> dict:
            rows = []
            for strategy in strategies:
                for seed in seeds:
                    selected = [] if strategy == "seed-only" else [f"g-{i}" for i in range(32)]
                    rows.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "selected_groups": len(selected),
                            "selected_group_ids": selected,
                            "judgments": [{"group_id": group_id} for group_id in selected],
                            "training_records_seen": {"llm_feedback": 128, "oracle_feedback": 128},
                            "metrics": {
                                "llm_feedback": {"pairwise_accuracy": 0.50 + split_seed / 10000},
                                "oracle_feedback": {"pairwise_accuracy": 0.49 + split_seed / 10000},
                            },
                        }
                    )
            return {
                "status": "completed",
                "base_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "configuration": {
                    "llm_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                    "strategies": list(strategies),
                    "seeds": list(seeds),
                    "split_unit": "prompt",
                    "fail_on_split_overlap": True,
                    "candidate_budget": 32,
                    "seed_group_limit": 64,
                    "pool_group_limit": 256,
                    "eval_group_limit": 512,
                    "vary_split_by_seed": False,
                    "training_schedule": {"budget_mode": "equal-epochs"},
                    "split_audit_by_seed": {
                        str(seed): {
                            "split_seed": split_seed,
                            "prompts_with_multiple_splits": 0,
                        }
                        for seed in seeds
                    },
                },
                "results": rows,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for split_seed in (17, 23, 29, 37):
                path = root / f"split-{split_seed}.json"
                path.write_text(json.dumps(make_report(split_seed)), encoding="utf-8")
                paths.append(path)

            result = audit(paths)
            self.assertEqual(result["status"], "audited")
            self.assertEqual(result["split_seeds"], [17, 23, 29, 37])
            self.assertEqual(result["split_count"], 4)
            self.assertEqual(len(result["split_rows"]), 4)

            with self.assertRaises(ValueError):
                audit(paths[:3] + [paths[0]])


if __name__ == "__main__":
    unittest.main()
