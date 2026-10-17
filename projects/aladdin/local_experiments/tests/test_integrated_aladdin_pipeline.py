from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core_system.mle_core.models import TaskRecord
from local_experiments.aladdin.integrated_pipeline.calibration import (
    ReliabilityCalibrator,
    wilson_lower_bound,
)
from local_experiments.aladdin.integrated_pipeline.demo import run_demo
from local_experiments.aladdin.integrated_pipeline.system import (
    DataIntegrityError,
    HumanFeedback,
    QualityAssessment,
    RequirementHypothesis,
    RequirementState,
    SynthesisBudgetAllocator,
    clean_and_deduplicate,
    find_near_duplicate_pairs,
    quality_gate,
    select_feedback_queries,
    select_mmr,
    update_requirement_weights,
)


def record(
    record_id: str,
    text: str,
    *,
    split: str = "seed",
    target: str | None = None,
    group: str | None = None,
    votes: dict[str, float] | None = None,
    review_cost: float = 1.0,
) -> TaskRecord:
    return TaskRecord(
        record_id=record_id,
        task_id="task",
        split=split,
        input=text,
        target=target,
        source_group_id=group,
        metadata={
            "fixture_profile_votes": votes or {},
            "review_cost": review_cost,
        },
    )


class VoteScorer:
    def score(self, hypothesis, item):
        return float(item.metadata["fixture_profile_votes"][hypothesis.hypothesis_id])


class FixedQualityScorer:
    def __init__(self, values):
        self.values = values

    def assess(self, item):
        return self.values[item.record_id]


class IntegratedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.profiles = [
            RequirementHypothesis("a", "Prefer account access coverage."),
            RequirementHypothesis("b", "Prefer billing coverage."),
        ]

    def test_exact_dedup_and_conflict_detection(self):
        items = [
            record("one", "Reset my password", target="account", group="g1"),
            record("two", " reset   MY password ", target="account", group="g1"),
        ]
        kept, duplicates = clean_and_deduplicate(items)
        self.assertEqual([item.record_id for item in kept], ["one"])
        self.assertEqual(duplicates, 1)
        with self.assertRaises(DataIntegrityError):
            clean_and_deduplicate(
                [items[0], record("three", "reset my password", target="billing", group="g1")]
            )

    def test_data_leakage_and_final_test_are_rejected(self):
        with self.assertRaisesRegex(DataIntegrityError, "crosses splits"):
            clean_and_deduplicate(
                [
                    record("train", "same source text", split="seed", group="same"),
                    record("dev", "different text", split="development", group="same"),
                ]
            )
        with self.assertRaisesRegex(DataIntegrityError, "final_test"):
            clean_and_deduplicate(
                [record("test", "sealed example", split="final_test", target="x")]
            )

    def test_near_duplicates_are_flagged_not_silently_removed(self):
        items = [
            record("near-a", "the account portal will not open for me"),
            record("near-b", "the account portal will not open for me today"),
            record("other", "send the payment invoice for my order"),
        ]
        pairs = find_near_duplicate_pairs(items, threshold=0.75)
        self.assertTrue(any({left, right} == {"near-a", "near-b"} for left, right, _ in pairs))
        self.assertEqual(len(items), 3)

    def test_quality_gate_defers_low_confidence_and_rejects_bad_quality(self):
        items = [record("accept", "valid request"), record("defer", "unclear request"), record("reject", "bad")]
        scores = {
            "accept": QualityAssessment(0.9, 0.95),
            "defer": QualityAssessment(0.98, 0.2),
            "reject": QualityAssessment(0.1, 0.95),
        }
        decisions, _ = quality_gate(
            items,
            FixedQualityScorer(scores),
            accept_threshold=0.7,
            min_confidence=0.7,
        )
        self.assertEqual(decisions, {"accept": "accept", "defer": "defer", "reject": "reject"})

    def test_quality_calibration_requires_separate_evidence_per_action(self):
        calibrator = ReliabilityCalibrator(
            minimum_audits=5,
            minimum_agreement_lower_bound=0.5,
        )
        item = record("candidate", "candidate request")
        self.assertEqual(calibrator.route(item, "accept"), "defer")
        calibrator.add_audits("task", "accept", [True] * 5)
        self.assertEqual(calibrator.route(item, "accept"), "accept")
        self.assertEqual(calibrator.route(item, "reject"), "defer")
        calibrator.add_audits("task", "reject", [False] * 5)
        self.assertEqual(calibrator.route(item, "reject"), "reject")

    def test_quality_calibration_uses_conservative_confidence_bound(self):
        calibrator = ReliabilityCalibrator(
            minimum_audits=5,
            minimum_agreement_lower_bound=0.8,
        )
        calibrator.add_audits("task", "accept", [True] * 5)
        self.assertFalse(calibrator.is_certified("task", "accept"))
        self.assertLess(wilson_lower_bound(5, 5), 0.8)

    def test_training_feedback_does_not_count_as_quality_calibration_audit(self):
        calibrator = ReliabilityCalibrator(
            minimum_audits=1,
            minimum_agreement_lower_bound=0.1,
        )
        item = record(
            "candidate",
            "candidate request",
            votes={"a": 1, "b": 0},
            target="account",
        )
        before = calibrator.summary()
        update_requirement_weights(
            self.profiles,
            {"a": 0.5, "b": 0.5},
            item,
            HumanFeedback(True, "account"),
            VoteScorer(),
        )
        self.assertEqual(calibrator.summary(), before)

    def test_query_score_uses_downstream_impact_and_human_effort(self):
        q_account = record("q-account", "account issue", votes={"a": 1, "b": 0})
        q_billing = record("q-billing", "billing issue", votes={"a": 0, "b": 1}, review_cost=2)
        q_consensus = record("q-consensus", "general issue", votes={"a": 1, "b": 1})
        pool = [
            record("d-account", "account access", votes={"a": 1, "b": 0}),
            record("d-billing", "payment invoice", votes={"a": 0, "b": 1}),
        ]
        queries = select_feedback_queries(
            [q_account, q_billing, q_consensus],
            pool,
            self.profiles,
            {"a": 0.5, "b": 0.5},
            VoteScorer(),
            max_queries=3,
            downstream_selection_limit=1,
            random_fraction=0,
            seed=9,
            cost_budget=4.0,
        )
        scores = {item.record_id: item.score for item in queries}
        self.assertGreater(scores["q-account"], scores["q-billing"])
        self.assertEqual(scores["q-consensus"], 0)
        self.assertGreater(queries[0].expected_selection_change, 0)

    def test_random_exploration_reserve_is_marked(self):
        pool = [
            record(f"q{i}", f"candidate {i}", votes={"a": i % 2, "b": 1 - i % 2})
            for i in range(4)
        ]
        queries = select_feedback_queries(
            pool,
            pool,
            self.profiles,
            {"a": 0.5, "b": 0.5},
            VoteScorer(),
            max_queries=2,
            downstream_selection_limit=1,
            random_fraction=0.5,
            seed=1,
        )
        self.assertEqual(len(queries), 2)
        self.assertEqual(sum(item.random_exploration for item in queries), 1)

    def test_feedback_selection_respects_human_time_budget(self):
        items = [
            record("short", "account request", votes={"a": 1, "b": 0}, review_cost=1),
            record("long", "billing request", votes={"a": 0, "b": 1}, review_cost=2),
            record("general", "general request", votes={"a": 1, "b": 1}, review_cost=1),
        ]
        queries = select_feedback_queries(
            items,
            items,
            self.profiles,
            {"a": 0.5, "b": 0.5},
            VoteScorer(),
            max_queries=3,
            downstream_selection_limit=1,
            random_fraction=0,
            seed=4,
            cost_budget=2.0,
        )
        self.assertLessEqual(sum(item.review_cost for item in queries), 2.0)

    def test_feedback_updates_requirement_posterior(self):
        item = record("q", "account issue", votes={"a": 1, "b": 0})
        updated = update_requirement_weights(
            self.profiles,
            {"a": 0.5, "b": 0.5},
            item,
            HumanFeedback(True),
            VoteScorer(),
        )
        self.assertGreater(updated["a"], updated["b"])

    def test_mmr_prefers_relevance_then_diversity(self):
        items = [
            record("a", "account portal access issue"),
            record("b", "account portal access issue today"),
            record("z", "invoice payment receipt"),
        ]
        chosen = select_mmr(
            items,
            {item.record_id: 1.0 for item in items},
            limit=2,
            relevance_weight=0.5,
        )
        self.assertEqual(chosen[0].record_id, "a")
        self.assertEqual(chosen[1].record_id, "z")

    def test_ucb_allocator_explores_then_records_gain_per_cost(self):
        allocator = SynthesisBudgetAllocator(("a", "b"), exploration=0)
        self.assertEqual(allocator.choose(), "a")
        allocator.update("a", 0.2, 2)
        self.assertEqual(allocator.choose(), "b")
        allocator.update("b", 0.1, 2)
        self.assertEqual(allocator.choose(), "a")

    def test_end_to_end_smoke_logs_artifacts_and_never_reads_final_test(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = run_demo(output)
            self.assertEqual(report["generated_count"], 5)
            self.assertEqual(report["quality_counts"]["reject"], 1)
            self.assertEqual(report["feedback_received"], 4)
            self.assertIn("human-edit-0001", report["requirement_state_after_round"]["weights"])
            self.assertEqual(report["allocator_after_round"]["counts"]["coverage"], 1)
            snapshot = output / "runs" / "integrated-pipeline-smoke.json"
            self.assertTrue(snapshot.exists())
            saved = __import__("json").loads(snapshot.read_text())
            self.assertFalse(saved["final_test_accessed"])
            self.assertEqual(saved["estimated_feedback_cost"], 32.0)
            self.assertEqual(saved["human_review_budget_overrun"], 0.0)
            self.assertEqual(saved["automatic_quality_decisions"], 4)
            self.assertAlmostEqual(saved["automatic_quality_coverage"], 0.8)
            self.assertIsNotNone(saved["quality_calibration_before"])
            self.assertTrue(any("candidate-pool" in name for name in saved["artifacts"]))
            self.assertTrue((output / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
