from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_experiments.aladdin.hh_rlhf_feedback import (
    ADAPTER_VERSION,
    DATASET_REVISION,
    HHRLHFAdapterError,
    load_hh_rlhf_groups,
    run_hh_rlhf_feedback_experiment,
)


def hh_row(index: int) -> dict[str, str]:
    history = (
        f"\n\nHuman: context {index}\n\nAssistant: prior answer {index}"
        f"\n\nHuman: question {index}\n\nAssistant:"
    )
    return {
        "chosen": history + f" helpful answer {index}",
        "rejected": history + f" weak answer {index}",
    }


class HHRLHFFeedbackTests(unittest.TestCase):
    def _fixture(self, root: Path, count: int = 300) -> Path:
        path = root / "hh-rlhf-train.jsonl"
        path.write_text(
            "\n".join(json.dumps(hh_row(index)) for index in range(count)) + "\n",
            encoding="utf-8",
        )
        return path

    def test_adapter_preserves_history_and_hides_pairwise_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            groups = load_hh_rlhf_groups(self._fixture(Path(directory), count=1))
        group = groups[0]
        hidden = group.hidden_records
        released = group.release_feedback()
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].target, None)
        self.assertEqual(hidden[1].target, None)
        self.assertNotIn("feedback", hidden[0].metadata)
        self.assertEqual(released[0].target, "preferred")
        self.assertEqual(released[1].target, "rejected")
        self.assertIn("Human: question", str(hidden[0].input))
        self.assertIn("Response: helpful answer", str(hidden[0].input))
        self.assertEqual(hidden[0].metadata["adapter_version"], ADAPTER_VERSION)
        self.assertEqual(hidden[0].metadata["source_revision"], DATASET_REVISION)

    def test_stable_hash_split_is_reproducible_and_group_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(Path(directory))
            first = load_hh_rlhf_groups(path, split_seed=17)
            second = load_hh_rlhf_groups(path, split_seed=17)
        self.assertEqual(
            [(group.group_id, group.split) for group in first],
            [(group.group_id, group.split) for group in second],
        )
        self.assertTrue({group.split for group in first}.issubset({"seed", "held_in", "held_out"}))
        self.assertTrue(all(record.split == group.split for group in first for record in group.hidden_records))

    def test_prompt_split_keeps_duplicate_prompt_pairs_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-prompts.jsonl"
            history = "\n\nHuman: same question\n\nAssistant:"
            path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "chosen": history + f" answer {index}",
                            "rejected": history + f" weak answer {index}",
                        }
                    )
                    for index in range(12)
                )
                + "\n",
                encoding="utf-8",
            )
            groups = load_hh_rlhf_groups(path, split_seed=17, split_unit="prompt")
        self.assertEqual(len({group.prompt for group in groups}), 1)
        self.assertEqual(len({group.split for group in groups}), 1)

    def test_split_audit_detects_pair_level_prompt_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-prompts.jsonl"
            history = "\n\nHuman: same question\n\nAssistant:"
            path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "chosen": history + f" answer {index}",
                            "rejected": history + f" weak answer {index}",
                        }
                    )
                    for index in range(64)
                )
                + "\n",
                encoding="utf-8",
            )
            groups = load_hh_rlhf_groups(path, split_seed=17, split_unit="pair")
            from local_experiments.aladdin.hh_rlhf_feedback import split_audit

        audit = split_audit(groups)
        self.assertGreater(audit["prompts_with_multiple_splits"], 0)

    def test_final_test_path_is_recorded_but_never_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(Path(directory))
            report = run_hh_rlhf_feedback_experiment(
                path,
                seeds=(17,),
                seed_group_limit=8,
                feedback_budget=3,
                pool_group_limit=16,
                final_test_path=Path(directory) / "does-not-exist-test.jsonl",
            )
        self.assertEqual(report["final_test_path"].endswith("does-not-exist-test.jsonl"), True)
        self.assertEqual(report["split_protocol"]["final_test"].endswith("not loaded"), True)
        self.assertEqual(len(report["results"]), 5)
        for result in report["results"]:
            self.assertEqual(result["budget"]["feedback_requests"], 3)
            self.assertEqual(result["budget"]["llm_calls"], 0)

    def test_parallel_workers_keep_fixed_task_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_hh_rlhf_feedback_experiment(
                self._fixture(Path(directory)),
                strategies=("random", "aladdin"),
                seeds=(17,),
                seed_group_limit=8,
                feedback_budget=3,
                pool_group_limit=16,
                workers=2,
            )
        self.assertEqual(report["configuration"]["workers"], 2)
        self.assertEqual(
            [(item["seed"], item["strategy"]) for item in report["results"]],
            [(17, "random"), (17, "aladdin")],
        )

    def test_each_seed_uses_a_recorded_split_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_hh_rlhf_feedback_experiment(
                self._fixture(Path(directory)),
                strategies=("random",),
                seeds=(17, 18),
                split_seed=17,
                seed_group_limit=8,
                feedback_budget=3,
                pool_group_limit=16,
            )
        self.assertEqual(report["split_protocol"]["split_seed_by_seed"], {"17": 17, "18": 18})
        self.assertEqual([item["split_seed"] for item in report["results"]], [17, 18])
        self.assertNotEqual(
            report["results"][0]["diagnostics"]["selected_group_ids"],
            report["results"][1]["diagnostics"]["selected_group_ids"],
        )

    def test_chosen_and_rejected_must_share_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(
                json.dumps({
                    "chosen": "\n\nHuman: one\n\nAssistant: good",
                    "rejected": "\n\nHuman: two\n\nAssistant: bad",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(HHRLHFAdapterError):
                load_hh_rlhf_groups(path)

    def test_embedded_assistant_marker_stays_inside_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedded-marker.jsonl"
            history = "\n\nHuman: explain this\n\nAssistant:"
            path.write_text(
                json.dumps({
                    "chosen": history + " answer with\n\nAssistant: quoted marker",
                    "rejected": history + " shorter answer",
                }) + "\n",
                encoding="utf-8",
            )
            groups = load_hh_rlhf_groups(path)
        self.assertIn("quoted marker", str(groups[0].released_records[0].input))

    def test_empty_candidate_response_is_preserved_and_marked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty-response.jsonl"
            history = "\n\nHuman: explain this\n\nAssistant:"
            path.write_text(
                json.dumps({
                    "chosen": history + " useful answer",
                    "rejected": history,
                }) + "\n",
                encoding="utf-8",
            )
            groups = load_hh_rlhf_groups(path)
        group = groups[0]
        self.assertEqual(group.released_records[1].target, "rejected")
        self.assertTrue(group.released_records[1].metadata["empty_response"])
        self.assertFalse(group.released_records[0].metadata["empty_response"])
        self.assertTrue(str(group.released_records[1].input).endswith("Response: "))

    def test_turn_marker_inside_one_response_does_not_break_shared_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker-in-response.jsonl"
            history = "\n\nHuman: explain this\n\nAssistant:"
            path.write_text(
                json.dumps({
                    "chosen": history + " answer\n\nHuman: quoted text",
                    "rejected": history + " short answer",
                }) + "\n",
                encoding="utf-8",
            )
            groups = load_hh_rlhf_groups(path)
        self.assertIn("quoted text", str(groups[0].released_records[0].input))
        self.assertEqual(groups[0].released_records[0].metadata["empty_response"], False)

    def test_required_pair_fields_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps({"chosen": "only chosen"}) + "\n", encoding="utf-8")
            with self.assertRaises(HHRLHFAdapterError):
                load_hh_rlhf_groups(path)


if __name__ == "__main__":
    unittest.main()
