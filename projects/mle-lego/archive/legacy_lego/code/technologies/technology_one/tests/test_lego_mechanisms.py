from __future__ import annotations

import tempfile
import unittest
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

from .adaptive import run_all_strategies, run_strategy
from .data import lego_stream_batches
from .harness import (
    HarnessEdit,
    HarnessVersionChain,
    apply_edits,
    dependency_closure,
    initial_harness,
)
from .harness_evolution import (
    HarnessEvolutionController,
    HarnessProposalError,
    build_harness_edit_prompt,
    parse_edit_proposal,
)
from .helpsteer2 import normalize_helpsteer2_rows
from .prism import normalize_prism_rows
from .real_data_runner import run_fixed_real_baseline
from .data_validation import DatasetValidationError, validate_records
from .protocol import (
    DataProtocol,
    DelayedFeedback,
    DelayedFeedbackQueue,
    JsonlTaskAdapter,
)
from .reproducibility import compare_run_projections
from .server_preflight import _normalized_path, _walk_placeholders
from .server_adapters import (
    CommandHarnessEditor,
    JsonCommand,
    ServerCommandError,
    _expected_versions,
    budget_from_value,
    run_json_command,
)
from .server_baseline_matrix import (
    BASELINE_METHOD_ORDER,
    METHOD_SEMANTICS,
    build_baseline_manifests,
)
from .server_technical_one_baseline import _audit_training_records
from .decision_protocol import (
    ReflectionRecord,
    ReflectionStore,
    build_policy_prompt,
)
from core_system.mle_core.models import (
    Budget,
    Observation,
    RunConfig,
    RunStatus,
    TaskSpec,
)
from core_system.mle_core.store import ArtifactStore


class HarnessMechanismTests(unittest.TestCase):
    def test_budget_parser_accepts_training_record_diagnostic_but_rejects_unknown_fields(self) -> None:
        budget = budget_from_value(
            {
                "training_steps": 100,
                "training_seconds": 28.5,
                "training_records_seen": 400,
            }
        )
        self.assertEqual(budget.training_steps, 100)
        self.assertEqual(budget.training_seconds, 28.5)
        with self.assertRaisesRegex(ValueError, "Unknown budget fields"):
            budget_from_value({"training_step": 100})

    def _proposal(self, manifest, component="generator", patch=None):
        patch = patch or {"routes": ["account", "technical"]}
        return parse_edit_proposal(
            {
                "base_harness_version": manifest.version,
                "edits": [{
                    "component": component,
                    "patch": patch,
                    "rationale": "Expand coverage for observed task categories.",
                }],
                "rationale": "Use prior held-in evidence to expand relevant coverage.",
            },
            current=manifest,
        )

    def test_harness_edit_proposal_enforces_component_allowlist_and_dependencies(self) -> None:
        manifest = initial_harness()
        with self.assertRaisesRegex(HarnessProposalError, "not editable"):
            parse_edit_proposal(
                {
                    "base_harness_version": manifest.version,
                    "edits": [{"component": "generator", "patch": {"python": "raise SystemExit()"}, "rationale": "bad"}],
                },
                current=manifest,
            )
        with self.assertRaisesRegex(HarnessProposalError, "dependency closure"):
            parse_edit_proposal(
                {
                    "base_harness_version": manifest.version,
                    "edits": [{"component": "selector", "patch": {"min_quality": 0.7}, "rationale": "lower threshold"}],
                },
                current=manifest,
            )

    def test_harness_edit_rejects_stale_versions_and_invalid_parameter_values(self) -> None:
        manifest = initial_harness()
        raw = {
            "base_harness_version": "stale",
            "edits": [{"component": "generator", "patch": {"routes": ["account"]}, "rationale": "valid"}],
        }
        with self.assertRaisesRegex(HarnessProposalError, "stale"):
            parse_edit_proposal(raw, current=manifest)
        raw["base_harness_version"] = manifest.version
        raw["edits"][0]["patch"] = {"routes": ["final_test"]}
        with self.assertRaisesRegex(HarnessProposalError, "route values"):
            parse_edit_proposal(raw, current=manifest)

    def test_harness_editor_prompt_rejects_held_out_inputs(self) -> None:
        with self.assertRaisesRegex(HarnessProposalError, "outside the allowlist"):
            build_harness_edit_prompt(
                current=initial_harness(),
                observation={"held_out_metrics": {"macro_f1": 0.9}},
                prior_feedback={},
            )
        with self.assertRaisesRegex(HarnessProposalError, "cannot contain"):
            build_harness_edit_prompt(
                current=initial_harness(),
                observation={"held_in_metrics": {"macro_f1": 0.5}},
                prior_feedback={"summary": {"final_test_score": 1.0}},
            )

    def test_command_harness_editor_uses_json_bridge_and_preserves_llm_cost(self) -> None:
        manifest = initial_harness()
        response = {
            "proposal": {
                "base_harness_version": manifest.version,
                "edits": [{
                    "component": "generator",
                    "patch": {"routes": ["account", "technical"]},
                    "rationale": "Expand coverage for observed task categories.",
                }],
                "rationale": "Held-in errors show missing technical coverage.",
            },
            "cost": {"llm_calls": 1, "llm_tokens": 137},
        }
        script = "import json,sys; json.load(sys.stdin); json.dump(" + repr(response) + ",sys.stdout)"
        editor = CommandHarnessEditor(
            JsonCommand.from_value([sys.executable, "-c", script]),
            model_id="test-decision-model",
            model_revision="test-revision",
        )
        proposal = editor.propose(
            current=manifest,
            observation={"task_id": "helpsteer2", "held_in_metrics": {"macro_f1": 0.4}},
            prior_feedback={"lesson": "review routes", "held_in_metrics": {"macro_f1": 0.4}},
        )
        self.assertEqual(proposal.base_version, manifest.version)
        self.assertEqual(proposal.cost.llm_calls, 1)
        self.assertEqual(proposal.cost.llm_tokens, 137)
        self.assertEqual(proposal.edits[0].component, "generator")
        self.assertEqual(proposal.model_id, "test-decision-model")
        self.assertEqual(len(proposal.system_prompt_hash), 64)
        self.assertEqual(len(proposal.user_prompt_hash), 64)

    def test_harness_controller_acceptance_uses_held_in_only_and_persists_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            manifest = initial_harness()
            controller = HarnessEvolutionController(manifest, store=store, run_id="evo-1")
            proposal = self._proposal(manifest)
            accepted = controller.consider(
                proposal,
                active_held_in=0.60,
                candidate_metrics={"held_in_macro_f1": 0.61, "held_out_macro_f1": 0.10},
                held_in_metric="held_in_macro_f1",
                held_out_metric="held_out_macro_f1",
                batch_id="batch-1",
            )
            self.assertEqual(accepted["status"], "accepted")
            self.assertNotEqual(controller.current.version, manifest.version)
            self.assertFalse(accepted["held_out_used_for_acceptance"])

            rejected = controller.consider(
                self._proposal(controller.current),
                active_held_in=0.61,
                candidate_metrics={"held_in_macro_f1": 0.59, "held_out_macro_f1": 1.0},
                held_in_metric="held_in_macro_f1",
                held_out_metric="held_out_macro_f1",
                batch_id="batch-2",
            )
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(controller.current.version, accepted["active_version"])
            self.assertTrue(any(event["event_type"] == "harness_version_accepted" for event in store.read_events()))

    def test_harness_evolve_batch_connects_editor_evaluator_and_versioned_next_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = initial_harness()
            controller = HarnessEvolutionController(
                manifest, store=ArtifactStore(directory), run_id="evo-loop"
            )
            proposal = self._proposal(manifest)

            class FakeEditor:
                model_id = "fake-model"
                model_revision = "r1"

                def propose(self, *, current, observation, prior_feedback):
                    self.asserted_version = current.version
                    return replace(proposal, model_id=self.model_id, model_revision=self.model_revision)

            editor = FakeEditor()
            evaluated = []

            def evaluate_candidate(candidate):
                evaluated.append(candidate.version)
                return (
                    {"held_in_macro_f1": 0.62, "held_out_macro_f1": 0.2},
                    Budget(training_steps=1, evaluator_calls=2),
                )

            result = controller.evolve_batch(
                editor=editor,
                observation={"task_id": "helpsteer2", "held_in_metrics": {"macro_f1": 0.6}},
                prior_feedback={"lesson": "expand relevant routes"},
                active_held_in=0.60,
                evaluate_candidate=evaluate_candidate,
                held_in_metric="held_in_macro_f1",
                held_out_metric="held_out_macro_f1",
                batch_id="batch-1",
            )
            self.assertEqual(editor.asserted_version, manifest.version)
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["policy_model"]["model_id"], "fake-model")
            self.assertEqual(result["candidate_evaluation_cost"].training_steps, 1)
            self.assertEqual(controller.current.version, evaluated[0])

            next_proposal = self._proposal(controller.current)
            self.assertEqual(next_proposal.base_version, result["active_version"])

    def test_unknown_policy_versions_are_replaced_by_current_guard(self) -> None:
        observation = SimpleNamespace(
            config=SimpleNamespace(
                harness_version="harness-v1",
                data_version="data-v1",
                task_id="task-v1",
            )
        )
        self.assertEqual(
            _expected_versions({"selected_seed_records": "missing"}, observation),
            {
                "harness": "harness-v1",
                "data": "data-v1",
                "task": "task-v1",
            },
        )
        self.assertEqual(
            _expected_versions({"harness": "stale", "unknown": "ignored"}, observation),
            {"harness": "stale"},
        )

    def test_prism_adapter_preserves_session_provenance_without_hard_target(self) -> None:
        conversations = [
            {
                "conversation_id": "c1",
                "user_id": "u1",
                "generated_datetime": "2023-01-01 00:00:00",
            },
            {
                "conversation_id": "c2",
                "user_id": "u1",
                "generated_datetime": "2023-01-01 00:01:00",
            },
        ]
        utterances = [
            {
                "utterance_id": "ut1",
                "interaction_id": "i1",
                "conversation_id": "c1",
                "user_id": "u1",
                "turn": 0,
                "within_turn_id": 0,
                "conversation_type": "unguided",
                "user_prompt": "question",
                "model_response": "answer",
                "model_name": "model-a",
                "model_provider": "provider-a",
                "score": 4,
                "if_chosen": True,
            },
            {
                "utterance_id": "ut2",
                "interaction_id": "i2",
                "conversation_id": "c2",
                "user_id": "u1",
                "turn": 1,
                "within_turn_id": 0,
                "conversation_type": "unguided",
                "user_prompt": "follow-up",
                "model_response": "answer 2",
                "model_name": "model-a",
                "model_provider": "provider-a",
                "score": 2,
                "if_chosen": False,
            },
        ]
        records = normalize_prism_rows(conversations, utterances, batch_size=1, feedback_delay=2)
        self.assertEqual([record.record_id for record in records], ["prism-ut1", "prism-ut2"])
        self.assertTrue(all(record.target is None for record in records))
        self.assertEqual({record.split for record in records}, {records[0].split})
        self.assertEqual(records[0].source_group_id, "conversation-c1")
        self.assertEqual(records[1].metadata["turn"], 1)
        self.assertEqual(records[1].metadata["feedback"]["release_batch"], 3)

    def test_prism_adapter_rejects_cross_user_conversation_rows(self) -> None:
        conversation = {
            "conversation_id": "c1",
            "user_id": "u1",
            "generated_datetime": "2023-01-01 00:00:00",
        }
        utterance = {
            "utterance_id": "ut1",
            "interaction_id": "i1",
            "conversation_id": "c1",
            "user_id": "u2",
            "turn": 0,
            "within_turn_id": 0,
        }
        with self.assertRaises(ValueError):
            normalize_prism_rows([conversation], [utterance])

    def test_server_preflight_helpers_are_portable_and_detect_placeholders(self) -> None:
        self.assertEqual(_normalized_path(r"data\records.jsonl"), "data/records.jsonl")
        self.assertEqual(_walk_placeholders({"model": "SET_MODEL", "seed": 17}), [".model"])

    def test_server_json_command_bridge_accepts_one_json_object(self) -> None:
        command = JsonCommand.from_value(
            [sys.executable, "-c", "import json,sys; json.dump({'ok': json.load(sys.stdin)['value'] + 1}, sys.stdout)"]
        )
        self.assertEqual(run_json_command(command, {"value": 2}), {"ok": 3})

    def test_server_json_command_bridge_rejects_nonzero_process(self) -> None:
        command = JsonCommand.from_value([sys.executable, "-c", "raise SystemExit(3)"])
        with self.assertRaises(ServerCommandError):
            run_json_command(command, {})

    def test_reproducibility_projection_ignores_run_id_but_detects_behavior_changes(self) -> None:
        base = {
            "config": {"run_id": "a", "seed": 17},
            "status": "completed",
            "metrics_by_split": {"held_out": {"accuracy": 1.0}},
            "final_metrics": {"accuracy": 1.0},
            "artifacts": [{"kind": "checkpoint", "content_hash": "same"}],
            "costs": [{"training_steps": 1}],
        }
        same = {**base, "config": {"run_id": "b", "seed": 17}}
        changed = {**same, "final_metrics": {"accuracy": 0.5}}
        self.assertEqual(compare_run_projections(base, same), [])
        self.assertEqual(compare_run_projections(base, changed), ["final_metrics"])

    def test_edits_are_atomic_and_parent_versioned(self) -> None:
        original = initial_harness()
        updated = apply_edits(
            original,
            [HarnessEdit("generator", {"routes": ["account", "billing"]}, "test")],
        )
        self.assertNotEqual(original.version, updated.version)
        self.assertIsNone(original.parent_version)
        self.assertEqual(updated.parent_version, original.version)
        self.assertEqual(original.components["generator"].config["routes"], ["account"])

    def test_dependency_closure_includes_upstream_generator(self) -> None:
        manifest = initial_harness()
        self.assertEqual(dependency_closure(manifest, ("selector",)), ("generator", "selector"))

    def test_version_chain_can_restore_previous_harness(self) -> None:
        chain = HarnessVersionChain(initial_harness())
        original = chain.current
        candidate = chain.propose(
            [HarnessEdit("generator", {"routes": ["account", "billing"]}, "test")]
        )
        chain.accept(candidate)
        self.assertEqual(chain.current.version, candidate.version)
        self.assertEqual(chain.rollback().version, original.version)

    def test_dependency_aware_strategy_accepts_joint_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_strategy("dependency-aware", Path(directory))
        self.assertTrue(result.accepted_versions)
        self.assertEqual(result.final_harness_version, result.accepted_versions[-1])
        self.assertEqual(result.batches[1].selected_count, 0)
        experience = result.experiences.experiences[0]
        self.assertIn("held_out_regression_report_only", experience.validation)
        self.assertNotIn("held_out_unchanged_or_better", experience.validation)

    def test_independent_edit_is_rejected_and_current_version_survives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_strategy("independent-edit", Path(directory))
        self.assertFalse(result.accepted_versions)
        self.assertTrue(result.rejected_versions)
        self.assertEqual(result.final_harness_version, result.initial_harness_version)
        self.assertGreaterEqual(result.rollbacks, 1)

    def test_all_comparison_strategies_have_same_stream_and_budget_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = run_all_strategies(Path(directory))
        self.assertEqual(set(results), {"fixed", "reflexion", "independent-edit", "random-joint-edit", "dependency-aware"})
        for result in results.values():
            self.assertEqual([batch.batch_id for batch in result.batches], ["batch-0001", "batch-0002", "batch-0003"])
            self.assertGreater(result.used_budget.training_steps, 0)

    def test_jsonl_protocol_adapter_validates_and_groups_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                '{"record_id":"r1","split":"seed","input":"a","target":"x","batch_id":"b0"}\n'
                '{"record_id":"r2","split":"held_out","input":"b","target":"y","batch_id":"b1"}\n',
                encoding="utf-8",
            )
            records = JsonlTaskAdapter(DataProtocol("task")).load(path)
        self.assertEqual(list(JsonlTaskAdapter(DataProtocol("task")).group_batches(records)), ["b0", "b1"])

    def test_jsonl_reader_does_not_split_unicode_line_separator_in_json_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.jsonl"
            path.write_text(
                '{"record_id":"r1","split":"seed","input":"first\u2028second","target":"x"}\n',
                encoding="utf-8",
            )
            records = JsonlTaskAdapter(DataProtocol("task")).load(path)
        self.assertEqual(records[0].input, "first\u2028second")

    def test_delayed_feedback_is_released_once_at_its_declared_batch(self) -> None:
        queue = DelayedFeedbackQueue(
            [DelayedFeedback("f1", "r1", source_batch=1, release_batch=3, value="keep")]
        )
        self.assertEqual(queue.release_for_batch(2), [])
        released = queue.release_for_batch(3)
        self.assertEqual([item.feedback_id for item in released], ["f1"])
        self.assertEqual(queue.release_for_batch(4), [])

    def test_dataset_validation_checks_groups_parents_and_delayed_feedback(self) -> None:
        protocol = DataProtocol("task")
        from core_system.mle_core.models import TaskRecord

        valid = [
            TaskRecord(
                "r1",
                "task",
                "seed",
                "seed input",
                "x",
                source_group_id="g1",
                metadata={"batch_id": "b0"},
            ),
            TaskRecord(
                "r2",
                "task",
                "held_out",
                "held out input",
                "x",
                source_group_id="g2",
                metadata={
                    "batch_id": "b1",
                    "feedback": {"source_batch": 1, "release_batch": 3},
                },
            ),
        ]
        summary = validate_records(valid, protocol)
        self.assertEqual(summary.record_count, 2)
        self.assertEqual(summary.delayed_feedback_count, 1)
        invalid = valid + [
            TaskRecord(
                "r3",
                "held-out-parent-child",
                "held_out",
                "bad task",
                "x",
                metadata={"batch_id": "b1"},
                parent_record_ids=("r1",),
            )
        ]
        with self.assertRaises(DatasetValidationError):
            validate_records(invalid, protocol)

    def test_helpsteer2_adapter_keeps_pairs_together_and_preserves_scores(self) -> None:
        rows = [
            {"prompt": "p", "response": "a", "helpfulness": 4, "correctness": 4, "coherence": 4, "complexity": 2, "verbosity": 3},
            {"prompt": "p", "response": "b", "helpfulness": 2, "correctness": 3, "coherence": 3, "complexity": 2, "verbosity": 2},
        ]
        records = normalize_helpsteer2_rows(rows, feedback_delay=2)
        self.assertEqual(len(records), 2)
        self.assertEqual({record.source_group_id for record in records}, {records[0].source_group_id})
        self.assertEqual(len({record.split for record in records}), 1)
        self.assertEqual(records[0].metadata["feedback"]["release_batch"] - records[0].metadata["feedback"]["source_batch"], 2)
        self.assertIn("scores", records[0].metadata)

    def test_helpsteer2_validation_split_is_reserved_for_final_test(self) -> None:
        rows = [
            {"prompt": "validation prompt", "response": "a", "helpfulness": 4, "correctness": 4, "coherence": 4, "complexity": 2, "verbosity": 3},
            {"prompt": "validation prompt", "response": "b", "helpfulness": 2, "correctness": 3, "coherence": 3, "complexity": 2, "verbosity": 2},
        ]
        records = normalize_helpsteer2_rows(rows, source_split="validation")
        self.assertEqual({record.split for record in records}, {"final_test"})

    def test_helpsteer2_adapter_handles_a_prompt_group_larger_than_two(self) -> None:
        rows = [
            {"prompt": "p4", "response": str(index), "helpfulness": score, "correctness": score, "coherence": score, "complexity": score, "verbosity": score}
            for index, score in enumerate((4, 3, 2, 1))
        ]
        records = normalize_helpsteer2_rows(rows)
        self.assertEqual(len(records), 4)
        self.assertEqual([record.target for record in records], ["preferred", "tie", "tie", "rejected"])

    def test_real_runner_requires_explicit_final_test_data_to_unlock_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = normalize_helpsteer2_rows(
                [
                    {"prompt": "train", "response": "a", "helpfulness": 4, "correctness": 4, "coherence": 4, "complexity": 4, "verbosity": 4},
                    {"prompt": "train", "response": "b", "helpfulness": 1, "correctness": 1, "coherence": 1, "complexity": 1, "verbosity": 1},
                ]
            )
            from .helpsteer2 import write_jsonl

            dataset = root / "train.jsonl"
            write_jsonl(records, str(dataset))
            with self.assertRaisesRegex(ValueError, "requires an explicit --final-test"):
                run_fixed_real_baseline(
                    dataset,
                    root / "runs",
                    "locked-final-test",
                    report_final_test=True,
                )

    def test_reflection_store_hides_new_records_until_next_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReflectionStore(Path(directory) / "reflections.jsonl")
            record = ReflectionRecord(
                reflection_id="b0-reflection",
                source_batch="b0",
                problem="empty selection",
                evidence={"selected_count": 0},
                lesson="Check route compatibility before selecting records.",
                conditions={"task_id": "helpsteer2"},
                created_by_model="Qwen/Qwen2.5-7B-Instruct",
                model_revision="test-revision",
                prompt_hash="hash",
                token_count=18,
                eligible_from_step=1,
            )
            store.add(record)
            self.assertEqual(store.eligible(decision_step=0), [])
            self.assertEqual([item.reflection_id for item in store.eligible(decision_step=1)], [record.reflection_id])

    def test_reflexion_prompt_contains_only_eligible_text_memory(self) -> None:
        observation = Observation(
            run_id="run",
            batch_id="run-batch-0001",
            batch_index=1,
            status=RunStatus.RUNNING,
            task=TaskSpec("helpsteer2", "HelpSteer2", "text", "label"),
            config=RunConfig(
                run_id="run",
                task_id="helpsteer2",
                backbone_llm="Qwen/Qwen2.5-7B-Instruct",
                task_model="task-model",
                evaluator="helpsteer2-evaluator",
                harness_version="fixed-v0",
                data_version="data-v0",
                initial_checkpoint="checkpoint",
                max_batches=3,
                budget_limit=Budget(llm_calls=10, llm_tokens=1000),
                seed=17,
            ),
            artifacts=(),
            metrics={"held_in_mae": 1.0},
            pending_feedback=(),
            remaining_budget=Budget(llm_calls=9, llm_tokens=900),
        )
        record = ReflectionRecord(
            reflection_id="b0-reflection",
            source_batch="b0",
            problem="regression",
            evidence={"held_in_mae": 1.2},
            lesson="Use the previous selection rule.",
            conditions={"task_id": "helpsteer2"},
            created_by_model="model",
            model_revision="revision",
            prompt_hash="hash",
            token_count=10,
            eligible_from_step=1,
        )
        system, user = build_policy_prompt(
            observation, method="react-reflexion", reflections=[record]
        )
        self.assertIn("eligible_reflections", user)
        self.assertIn("b0-reflection", user)
        self.assertIn("text-only episodic memory", system)

    def test_technology_one_fixed_baseline_audit_is_deterministic(self) -> None:
        from core_system.mle_core.models import TaskRecord
        from .technical_one_local import _format_record, initial_manifest

        manifest = initial_manifest(train_group_budget=1)
        raw = [
            TaskRecord(
                "r1",
                "helpsteer2-preference",
                "seed",
                "Prompt: explain\nResponse: " + "word " * 100,
                "preferred",
                source_group_id="g1",
                metadata={"score_mean": 4.0},
            ),
            TaskRecord(
                "r2",
                "helpsteer2-preference",
                "seed",
                "Prompt: explain\nResponse: short",
                "rejected",
                source_group_id="g1",
                metadata={"score_mean": 1.0},
            ),
        ]
        formatted = [_format_record(record, 80) for record in raw]
        first = _audit_training_records(raw, formatted, ["g1"], manifest=manifest)
        second = _audit_training_records(raw, formatted, ["g1"], manifest=manifest)
        self.assertEqual(first, second)
        self.assertEqual(first["train_group_count"], 1)
        self.assertEqual(first["train_record_count"], 2)
        self.assertEqual(first["selector"]["max_response_words"], 80)
        self.assertEqual(first["selector"]["truncated_record_count"], 1)
        self.assertEqual(first["label_counts"], {"preferred": 1, "rejected": 1})

    def test_baseline_matrix_covers_required_methods_and_labels_adaptations(self) -> None:
        from .technical_one_local import initial_manifest

        current = initial_manifest(train_group_budget=1)
        edits = {
            "generator": HarnessEdit(
                "generator",
                {"allowed_buckets": list(range(10))},
                "test generator edit",
            ),
            "selector": HarnessEdit(
                "selector",
                {"max_response_words": 160},
                "test selector edit",
            ),
            "filter": HarnessEdit(
                "filter",
                {"min_group_margin": 0.5},
                "test filter edit",
            ),
        }
        manifests = build_baseline_manifests(
            current,
            edits=edits,
            suspected_components=("selector",),
            seed=17,
        )
        self.assertEqual(set(manifests), set(BASELINE_METHOD_ORDER))
        self.assertIs(manifests["fixed-harness-react"], current)
        self.assertIs(manifests["react-reflexion"], current)
        self.assertIs(manifests["same-memory-no-process-update"], current)
        self.assertFalse(METHOD_SEMANTICS["general-self-harness"]["external_reproduction"])
        self.assertFalse(METHOD_SEMANTICS["curateevo-adaptation"]["external_reproduction"])


if __name__ == "__main__":
    unittest.main()
