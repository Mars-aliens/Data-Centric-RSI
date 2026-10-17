from __future__ import annotations

import unittest

from local_experiments.aladdin.active_feedback_impact import (
    Candidate,
    Requirement,
    _query_diagnostics,
    make_synthetic_benchmark,
    run_matrix,
    select_queries,
    update_posterior,
)


class ActiveFeedbackImpactTests(unittest.TestCase):
    def test_counterfactual_impact_counts_selection_changes_without_true_label(self) -> None:
        requirements = {
            "a": Requirement("a", (1.0, 0.0), 0.0),
            "b": Requirement("b", (-1.0, 0.0), 0.0),
        }
        candidates = [
            Candidate("c1", (1.0, 0.0), true_label=1, query_cost=1.0),
            Candidate("c2", (0.5, 0.0), true_label=1, query_cost=1.0),
            Candidate("c3", (-0.5, 0.0), true_label=0, query_cost=1.0),
        ]
        diagnostics = _query_diagnostics(
            [candidates[0]],
            requirements,
            {"a": 0.5, "b": 0.5},
            candidates[0],
            selection_budget=1,
            selection_candidates=candidates[1:],
        )
        self.assertEqual(diagnostics["disagreement"], 1.0)
        self.assertEqual(diagnostics["expected_selection_change"], 1.0)
        self.assertEqual(diagnostics["affected_selection_count"], 2)

    def test_impact_score_responds_to_downstream_selection_sensitivity(self) -> None:
        requirements = {
            "a": Requirement("a", (1.0, 0.0), 0.0),
            "b": Requirement("b", (-1.0, 0.0), 0.0),
        }
        query = Candidate("query", (1.0, 0.0), true_label=1, query_cost=1.0)
        posterior = {"a": 0.5, "b": 0.5}
        sensitive_pool = [
            Candidate("positive", (1.0, 0.0), true_label=1, query_cost=1.0),
            Candidate("negative", (-1.0, 0.0), true_label=0, query_cost=1.0),
        ]
        insensitive_pool = [
            Candidate("neutral-a", (0.0, 0.0), true_label=1, query_cost=1.0),
            Candidate("neutral-b", (0.0, 0.0), true_label=0, query_cost=1.0),
        ]

        sensitive = _query_diagnostics(
            [query], requirements, posterior, query,
            selection_budget=1, selection_candidates=sensitive_pool,
        )
        insensitive = _query_diagnostics(
            [query], requirements, posterior, query,
            selection_budget=1, selection_candidates=insensitive_pool,
        )

        self.assertEqual(sensitive["disagreement"], insensitive["disagreement"])
        self.assertGreater(sensitive["expected_selection_change"], 0.0)
        self.assertEqual(insensitive["expected_selection_change"], 0.0)
        self.assertGreater(sensitive["impact_value"], insensitive["impact_value"])

    def test_selector_does_not_depend_on_hidden_true_labels(self) -> None:
        requirements, seed, query_pool, data_pool, heldout = make_synthetic_benchmark(seed=17, total_candidates=90)
        all_ids = [item.candidate_id for item in [*seed, *query_pool, *data_pool, *heldout]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        posterior = {name: 1.0 / len(requirements) for name in requirements}
        altered = [
            Candidate(item.candidate_id, item.features, 1 - item.true_label, item.query_cost)
            for item in query_pool
        ]
        first, _, _ = select_queries(
            query_pool,
            requirements,
            posterior,
            strategy="impact-aware-random",
            cost_budget=8.0,
            selection_budget=8,
            seed=17,
        )
        second, _, _ = select_queries(
            altered,
            requirements,
            posterior,
            strategy="impact-aware-random",
            cost_budget=8.0,
            selection_budget=8,
            seed=17,
        )
        self.assertEqual(
            [item.candidate_id for item in first],
            [item.candidate_id for item in second],
        )
        self.assertGreater(len(seed), 0)
        self.assertGreater(len(data_pool), 0)
        self.assertGreater(len(heldout), 0)

    def test_all_strategies_respect_equal_cost_budget(self) -> None:
        requirements, _, pool, _, _ = make_synthetic_benchmark(seed=17, total_candidates=90)
        posterior = {name: 1.0 / len(requirements) for name in requirements}
        for strategy in ("random", "disagreement-only", "impact-aware", "impact-aware-random"):
            selected, diagnostics, spent = select_queries(
                pool,
                requirements,
                posterior,
                strategy=strategy,
                cost_budget=8.0,
                selection_budget=8,
                seed=17,
            )
            self.assertLessEqual(spent, 8.0 + 1e-9)
            self.assertEqual(len(selected), len({item.candidate_id for item in selected}))
            self.assertEqual(len(diagnostics), len(selected))

    def test_feedback_updates_requirement_posterior(self) -> None:
        requirements = {
            "a": Requirement("a", (1.0,), 0.0),
            "b": Requirement("b", (-1.0,), 0.0),
        }
        candidate = Candidate("c", (1.0,), true_label=1, query_cost=1.0)
        posterior = update_posterior(
            requirements,
            {"a": 0.5, "b": 0.5},
            candidate,
            observed_label=1,
        )
        self.assertGreater(posterior["a"], posterior["b"])

    def test_multiround_matrix_has_fixed_protocol_and_auditable_results(self) -> None:
        report = run_matrix(seeds=[17, 18], rounds=2)
        self.assertEqual(report["protocol_version"], "active-feedback-impact-local-v2")
        self.assertEqual(len(report["results"]), 8)
        self.assertFalse(report["configuration"]["selection_uses_hidden_labels"])
        self.assertTrue(report["configuration"]["feedback_updates_requirement_posterior"])
        self.assertIn("paired_delta_vs_random", report)
        self.assertIn("mean_cost_utilization", report["summary"]["random"])
        for result in report["results"]:
            self.assertLessEqual(result["total_feedback_cost"], 24.0 + 1e-9)
            ids = [candidate_id for row in result["rounds"] for candidate_id in row["queried_ids"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(result["final_selected_data_count"], 12)
            self.assertTrue(all(count == 0 for count in result["pool_overlap_counts"].values()))
            query_ids = set(result["pool_ids"]["query"])
            selection_ids = set(result["pool_ids"]["selection"])
            heldout_ids = set(result["pool_ids"]["heldout"])
            self.assertTrue(query_ids.isdisjoint(selection_ids))
            self.assertTrue(query_ids.isdisjoint(heldout_ids))
            self.assertTrue(selection_ids.isdisjoint(heldout_ids))


if __name__ == "__main__":
    unittest.main()
