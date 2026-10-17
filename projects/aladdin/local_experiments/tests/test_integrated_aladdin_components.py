from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core_system.mle_core.models import TaskRecord
from local_experiments.aladdin.integrated_pipeline.calibration import ReliabilityCalibrator
from local_experiments.aladdin.integrated_pipeline.evaluation import (
    EnsembleQualityScorer,
    QualityEvaluatorState,
)
from local_experiments.aladdin.integrated_pipeline.feedback_protocol import (
    pairwise_feedback_to_records,
    select_pairwise_feedback_queries,
    update_requirement_weights_from_pairwise,
)
from local_experiments.aladdin.integrated_pipeline.recursive import (
    load_recursive_state,
    run_recursive,
)
from local_experiments.aladdin.integrated_pipeline.strategies import strategy_for
from local_experiments.aladdin.integrated_pipeline.system import (
    HumanFeedback,
    QualityAssessment,
    RequirementHypothesis,
    RequirementState,
    SynthesisBudgetAllocator,
)
from local_experiments.aladdin.integrated_pipeline.demo import (
    FixtureGenerator,
    FixtureQualityAuditor,
    FixtureQualityScorer,
    FixtureRequirementScorer,
    FixtureReviewer,
    _candidate,
)
from core_system.mle_core.local_data import build_local_text_registry


def item(record_id: str, value: float) -> TaskRecord:
    return TaskRecord(
        record_id=record_id,
        task_id="task",
        split="seed",
        input=record_id,
        target="yes",
        metadata={"fixture_profile_votes": {"a": value, "b": 1 - value}},
    )


class FixedScorer:
    def __init__(self, score: float, confidence: float):
        self.score = score
        self.confidence = confidence

    def assess(self, record):
        return QualityAssessment(self.score, self.confidence, record.record_id)


class IntegratedComponentTests(unittest.TestCase):
    def test_strategy_definitions_are_substantively_distinct(self):
        self.assertNotEqual(strategy_for("coverage").objective, strategy_for("revision").objective)
        self.assertNotEqual(strategy_for("revision").required_evidence, strategy_for("contrastive").required_evidence)

    def test_ensemble_confidence_falls_when_prompts_disagree(self):
        result = EnsembleQualityScorer(
            [FixedScorer(0.9, 0.9), FixedScorer(0.2, 0.9)]
        ).assess(item("x", 1.0))
        self.assertAlmostEqual(result.score, 0.55)
        self.assertLess(result.confidence, 0.3)

    def test_evaluator_state_records_corrections_for_next_prompt(self):
        state = QualityEvaluatorState()
        state.record_audit(item("x", 1.0), "accept", False)
        self.assertEqual(state.audit_count, 1)
        self.assertEqual(state.disagreement_count, 1)
        self.assertGreater(state.prompt_version, 1)
        self.assertIn("automatic=accept", state.prompt_context())

    def test_pairwise_feedback_selection_and_expansion(self):
        requirements = [
            RequirementHypothesis("a", "A"),
            RequirementHypothesis("b", "B"),
        ]
        records = [item("left", 1.0), item("right", 0.0)]

        class Scorer:
            def score(self, hypothesis, record):
                return record.metadata["fixture_profile_votes"][hypothesis.hypothesis_id]

        queries = select_pairwise_feedback_queries(
            records,
            requirements,
            {"a": 0.5, "b": 0.5},
            Scorer(),
            max_queries=1,
            random_fraction=0,
            seed=3,
        )
        self.assertEqual(len(queries), 1)
        feedback = HumanFeedback(
            True,
            review_seconds=4,
            feedback_type="pairwise",
            preferred_record_id="left",
        )
        expanded = pairwise_feedback_to_records(queries[0], feedback)
        self.assertTrue(expanded["left"].accept_for_training)
        self.assertFalse(expanded["right"].accept_for_training)
        updated = update_requirement_weights_from_pairwise(
            requirements,
            {"a": 0.5, "b": 0.5},
            records[0],
            records[1],
            "left",
            Scorer(),
        )
        self.assertGreater(updated["a"], updated["b"])

    def test_recursive_runner_persists_state_and_advances_allocator(self):
        registry = build_local_text_registry()
        seed = registry.records_for("local-intent", split="seed")
        development = registry.records_for("local-intent", split="development")
        requirements = [
            RequirementHypothesis("account", "account"),
            RequirementHypothesis("billing", "billing"),
        ]
        state = RequirementState.uniform(requirements)
        candidates = [
            _candidate(
                "candidate-account",
                "account request",
                "seed-001",
                {"account": 0.9, "billing": 0.1},
            ),
            _candidate(
                "candidate-billing",
                "billing request",
                "seed-002",
                {"account": 0.1, "billing": 0.9},
            ),
        ]
        reviewer = FixtureReviewer(
            {
                "candidate-account": HumanFeedback(True, "account", review_seconds=1),
                "candidate-billing": HumanFeedback(True, "billing", review_seconds=1),
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            results = run_recursive(
                run_id="recursive-test",
                rounds=3,
                seed_records=seed,
                development_records=development,
                requirement_state=state,
                requirement_scorer=FixtureRequirementScorer(),
                quality_scorer=FixtureQualityScorer(),
                quality_calibrator=ReliabilityCalibrator(
                    minimum_audits=1, minimum_agreement_lower_bound=0.1
                ),
                quality_auditor=FixtureQualityAuditor(),
                generator=FixtureGenerator(candidates),
                reviewer=reviewer,
                allocator=SynthesisBudgetAllocator(("coverage", "revision", "contrastive")),
                output_dir=Path(temp),
                candidate_count=2,
                feedback_budget=2,
                feedback_cost_budget=10,
                generation_cost_budget=5,
                selected_data_limit=2,
            )
            self.assertEqual(len(results), 3)
            self.assertEqual([result.generator_strategy for result in results], ["coverage", "revision", "contrastive"])
            self.assertGreater(
                results[1].training_seed_count, results[0].training_seed_count
            )
            self.assertTrue((Path(temp) / "recursive-state.json").exists())
            self.assertFalse(results[-1].run_id.endswith("final-test"))
            restored = load_recursive_state(Path(temp) / "recursive-state.json")
            self.assertEqual(restored["completed_rounds"], 3)
            self.assertEqual(restored["allocator"].counts["contrastive"], 1)
            self.assertEqual(
                len(restored["current_seed_records"]),
                results[-1].training_seed_count + len(results[-1].selected_records),
            )
            self.assertEqual(
                restored["requirement_state"].confirmed_edits,
                state.confirmed_edits,
            )


if __name__ == "__main__":
    unittest.main()
