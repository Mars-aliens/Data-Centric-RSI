from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_experiments.aladdin.helpsteer2_feedback import (
    ADAPTER_VERSION,
    HelpSteer2AdapterError,
    load_helpsteer2_groups,
    run_helpsteer2_feedback_experiment,
    select_feedback_groups,
    _pairwise_metrics,
)
from core_system.mle_core.models import TaskRecord
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
    _checkpoint_scores,
)


def row(record_id: str, group_id: str, split: str, target: str, score: float, index: int) -> dict[str, object]:
    return {
        "record_id": record_id,
        "task_id": "helpsteer2-preference",
        "split": split,
        "input": f"Prompt: prompt {group_id}\nResponse: response {record_id}",
        "target": target,
        "source_group_id": group_id,
        "metadata": {
            "dataset": "nvidia/HelpSteer2",
            "pair_index": index,
            "pair_size": 2,
            "batch_id": (
                f"stream-{int(group_id.rsplit('-', 1)[-1]):04d}"
                if group_id.rsplit("-", 1)[-1].isdigit()
                else "stream-test"
            ),
            "score_mean": score,
            "scores": {"helpfulness": int(score), "correctness": int(score), "coherence": 3, "complexity": 3, "verbosity": 3},
            "feedback": {"decision": target, "scores": {"helpfulness": int(score)}},
        },
    }


class HelpSteer2FeedbackTests(unittest.TestCase):
    def test_pairwise_metrics_ignore_ties_and_score_preferred_candidate(self) -> None:
        records = [
            TaskRecord("a", "task", "held_out", "Prompt: x\nResponse: good", "preferred", "group-a"),
            TaskRecord("b", "task", "held_out", "Prompt: x\nResponse: bad", "rejected", "group-a"),
            TaskRecord("c", "task", "held_out", "Prompt: y\nResponse: tie-a", "tie", "group-b"),
            TaskRecord("d", "task", "held_out", "Prompt: y\nResponse: tie-b", "tie", "group-b"),
        ]
        checkpoint, _ = TfidfLogisticTrainer().train(
            train_records=records[:2],
            initial_checkpoint=None,
            config={"ngram_range": (1, 2), "max_iter": 200},
            seed=17,
            store=None,
        )
        metrics = _pairwise_metrics(checkpoint, records)
        self.assertEqual(metrics["pairwise_groups"], 1.0)
        self.assertEqual(metrics["pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["tie_groups"], 1.0)

    def test_evaluator_uses_the_same_tfidf_space_as_training(self) -> None:
        records = [
            TaskRecord(
                record_id="a",
                task_id="task",
                split="seed",
                input="Prompt: red\nResponse: apple",
                target="preferred",
            ),
            TaskRecord(
                record_id="b",
                task_id="task",
                split="seed",
                input="Prompt: blue\nResponse: sky",
                target="rejected",
            ),
        ]
        checkpoint, _ = TfidfLogisticTrainer().train(
            train_records=records,
            initial_checkpoint=None,
            config={"ngram_range": (1, 2), "max_iter": 200},
            seed=17,
            store=None,
        )
        expected = max(_checkpoint_scores(checkpoint, records[0].input).items(), key=lambda item: item[1])[0]
        metrics, _ = TfidfLogisticEvaluator().evaluate(checkpoint=checkpoint, records=records, config={})
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(expected, "preferred")

    def _fixture(self, root: Path) -> Path:
        path = root / "helpsteer2.jsonl"
        rows = []
        for index in range(20):
            group = f"group-{index:02d}"
            split = "seed" if index < 4 else "held_in" if index < 14 else "held_out"
            rows.extend([
                row(f"{group}-a", group, split, "preferred", 4.0, 0),
                row(f"{group}-b", group, split, "rejected", 2.0, 1),
            ])
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
        return path

    def test_adapter_hides_feedback_until_group_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            groups = load_helpsteer2_groups(self._fixture(Path(directory)))
        hidden = groups[0].hidden_records[0]
        released = groups[0].release_feedback()[0]
        self.assertEqual(groups[0].split, "seed")
        self.assertIsNone(hidden.target)
        self.assertFalse(hidden.metadata["feedback_released"])
        self.assertNotIn("scores", hidden.metadata)
        self.assertEqual(released.target, "preferred")
        self.assertTrue(released.metadata["feedback_released"])
        self.assertEqual(released.metadata["adapter_version"], ADAPTER_VERSION)

    def test_feedback_selection_uses_equal_group_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            groups = load_helpsteer2_groups(self._fixture(Path(directory)))
            seed_groups = [group for group in groups if group.split == "seed"]
            pool = [group for group in groups if group.split == "held_in"]
            seed_records = [record for group in seed_groups for record in group.release_feedback()]
            first, _ = select_feedback_groups(pool, strategy="random", budget=3, seed=17, seed_records=seed_records)
            second, _ = select_feedback_groups(pool, strategy="uniform", budget=3, seed=17, seed_records=seed_records)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertTrue(all(record.target is None for group in pool for record in group.hidden_records))

    def test_full_local_experiment_has_no_llm_calls_or_final_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_helpsteer2_feedback_experiment(
                self._fixture(Path(directory)),
                seeds=(17,),
                seed_group_limit=2,
                feedback_budget=2,
                pool_group_limit=4,
            )
        self.assertEqual(report["configuration"]["backbone_llm"], "not used in local mechanism experiment")
        self.assertEqual(set(report["summary"]), {"random", "uniform", "disagreement", "novelty", "aladdin"})
        for result in report["results"]:
            self.assertEqual(result["budget"]["feedback_requests"], 2)
            self.assertEqual(result["budget"]["llm_calls"], 0)
            self.assertFalse(result["llm_required"])
            self.assertEqual(result["metrics"]["num_records"], 12.0)

    def test_aladdin_v2_selects_the_requested_number_with_deterministic_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            groups = load_helpsteer2_groups(self._fixture(Path(directory)))
            seed_groups = [group for group in groups if group.split == "seed"]
            pool = [group for group in groups if group.split == "held_in"]
            seed_records = [record for group in seed_groups for record in group.release_feedback()]
            selected, diagnostics = select_feedback_groups(
                pool, strategy="aladdin-v2", budget=3, seed=17, seed_records=seed_records
            )
            repeated, repeated_diagnostics = select_feedback_groups(
                pool, strategy="aladdin-v2", budget=3, seed=17, seed_records=seed_records
            )
        self.assertEqual(len(selected), 3)
        self.assertEqual(diagnostics["selected_group_ids"], repeated_diagnostics["selected_group_ids"])
        self.assertTrue(all(record.target is None for group in pool for record in group.hidden_records))

    def test_invalid_group_crossing_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(
                json.dumps(row("a", "same", "seed", "preferred", 4.0, 0)) + "\n"
                + json.dumps(row("b", "same", "held_in", "rejected", 2.0, 1)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(HelpSteer2AdapterError):
                load_helpsteer2_groups(path)


if __name__ == "__main__":
    unittest.main()
