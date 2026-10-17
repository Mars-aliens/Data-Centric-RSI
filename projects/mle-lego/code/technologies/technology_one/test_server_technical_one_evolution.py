from __future__ import annotations

import json
import os
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from .harness import HarnessEdit, apply_edits
from .technical_one_local import initial_manifest
from .server_technical_one_evolution import (
    TechnologyOneHarnessEditor,
    TechnologyOneProposalFailure,
    _harness_config_fingerprint,
    _parse_technology_edit,
    build_technology_one_edit_prompt,
)
from .server_adapters import JsonCommand


class TechnologyOneEvolutionProtocolTests(unittest.TestCase):
    def test_editor_retries_semantically_invalid_dependency_proposal(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        with TemporaryDirectory() as directory:
            counter = Path(directory) / "counter"
            script = (
                "import json,os,pathlib\n"
                "p=pathlib.Path(os.environ['COUNTER'])\n"
                "n=int(p.read_text()) if p.exists() else 0\n"
                "p.write_text(str(n+1))\n"
                "req=json.load(__import__('sys').stdin)\n"
                "base=req['prompt']['user'].split('\\n\\nSchema correction:',1)[0]\n"
                "v=json.loads(base)['current_harness']['version']\n"
                "if n == 0:\n"
                "  edits=[{'component':'selector','patch':{'max_response_words':120},'rationale':'Adjust selection.'}]\n"
                "else:\n"
                "  edits=[{'component':'generator','patch':{'allowed_buckets':list(range(10))},'rationale':'Restore coverage.'},{'component':'selector','patch':{'max_response_words':120},'rationale':'Adjust selection.'}]\n"
                "json.dump({'proposal':{'base_harness_version':v,'edits':edits,'rationale':'Retry with closure.'},'cost':{'llm_calls':1,'llm_tokens':3}},__import__('sys').stdout)\n"
            )
            command = JsonCommand.from_value([sys.executable, "-c", script])
            old = os.environ.get("COUNTER")
            os.environ["COUNTER"] = str(counter)
            try:
                editor = TechnologyOneHarnessEditor(command, model_id="m", model_revision="r")
                proposal = editor.propose(
                    current=manifest,
                    observation={"batch_id": "b-0", "run_id": "r", "batch_index": 0, "task_id": "t", "active_harness_version": manifest.version, "held_in_metrics": {}, "available_feedback": 0, "remaining_budget": {}},
                    prior_feedback={},
                )
            finally:
                if old is None:
                    os.environ.pop("COUNTER", None)
                else:
                    os.environ["COUNTER"] = old
            self.assertEqual(proposal.cost.llm_calls, 2)
            self.assertEqual([edit.component for edit in proposal.edits], ["generator", "selector"])

    def test_editor_failure_type_retains_cost_after_two_invalid_proposals(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        with TemporaryDirectory() as directory:
            script = "import json,sys; req=json.load(sys.stdin); base=req['prompt']['user'].split('\\n\\nSchema correction:',1)[0]; v=json.loads(base)['current_harness']['version']; json.dump({'proposal':{'base_harness_version':v,'edits':[{'component':'selector','patch':{'max_response_words':120},'rationale':'Invalid closure.'}],'rationale':'Invalid.'},'cost':{'llm_calls':1,'llm_tokens':2}},sys.stdout)"
            editor = TechnologyOneHarnessEditor(
                JsonCommand.from_value([sys.executable, "-c", script]), model_id="m", model_revision="r"
            )
            with self.assertRaises(TechnologyOneProposalFailure) as context:
                editor.propose(
                    current=manifest,
                    observation={"batch_id": "b-0", "run_id": "r", "batch_index": 0, "task_id": "t", "active_harness_version": manifest.version, "held_in_metrics": {}, "available_feedback": 0, "remaining_budget": {}},
                    prior_feedback={},
                )
            self.assertEqual(context.exception.cost.llm_calls, 2)

    def test_prompt_excludes_report_only_splits_from_observation_payload(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        system, user = build_technology_one_edit_prompt(
            current=manifest,
            observation={
                "run_id": "pilot",
                "batch_id": "pilot-batch-0000",
                "batch_index": 0,
                "task_id": "helpsteer2-preference",
                "active_harness_version": manifest.version,
                "held_in_metrics": {"macro_f1": 0.2},
                "available_feedback": 0,
                "remaining_budget": {"llm_calls": 2},
            },
            prior_feedback={"history": []},
        )
        self.assertIn("held-out", system.lower())
        self.assertIn("final-test data are never decision inputs", system.lower())
        self.assertIn("held_in_metrics", user)
        self.assertNotIn('"held_out"', user)
        self.assertNotIn('"final_test"', user)

    def test_dependency_prompt_lists_exact_legal_component_sets(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        system, user = build_technology_one_edit_prompt(
            current=manifest,
            observation={
                "run_id": "pilot",
                "batch_id": "pilot-batch-0000",
                "batch_index": 0,
                "task_id": "helpsteer2-preference",
                "active_harness_version": manifest.version,
                "held_in_metrics": {"macro_f1": 0.2},
                "available_feedback": 0,
                "remaining_budget": {"llm_calls": 2},
            },
            prior_feedback={},
        )
        payload = json.loads(user)
        self.assertEqual(
            payload["legal_edit_component_sets"],
            {
                "generator": ["generator"],
                "selector": ["generator", "selector"],
                "filter": ["generator", "selector", "filter"],
            },
        )
        self.assertIn("exactly its listed component set", payload["proposal_contract"])

    def test_independent_prompt_lists_single_component_sets(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        system, user = build_technology_one_edit_prompt(
            current=manifest,
            observation={
                "run_id": "pilot",
                "batch_id": "pilot-batch-0000",
                "batch_index": 0,
                "task_id": "helpsteer2-preference",
                "active_harness_version": manifest.version,
                "held_in_metrics": {"macro_f1": 0.2},
                "available_feedback": 0,
                "remaining_budget": {"llm_calls": 2},
            },
            prior_feedback={},
            proposal_mode="independent",
        )
        payload = json.loads(user)
        self.assertEqual(
            payload["legal_edit_component_sets"],
            {
                "generator": ["generator"],
                "selector": ["selector"],
                "filter": ["filter"],
            },
        )
        self.assertIn("exactly one edit", payload["proposal_contract"])
        self.assertIn("exactly one component", system)
        self.assertNotIn("dependency-only ancestors must also appear", system)

    def test_joint_prompt_excludes_generator_only_target(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        _, user = build_technology_one_edit_prompt(
            current=manifest,
            observation={
                "run_id": "pilot",
                "batch_id": "pilot-batch-0000",
                "batch_index": 0,
                "task_id": "helpsteer2-preference",
                "active_harness_version": manifest.version,
                "held_in_metrics": {"macro_f1": 0.2},
                "available_feedback": 0,
                "remaining_budget": {"llm_calls": 2},
            },
            prior_feedback={},
            proposal_mode="dependency-aware-joint",
        )
        payload = json.loads(user)
        self.assertEqual(
            payload["legal_edit_component_sets"],
            {
                "selector": ["generator", "selector"],
                "filter": ["generator", "selector", "filter"],
            },
        )
        self.assertNotIn("generator", payload["legal_edit_component_sets"])

    def test_joint_parser_rejects_generator_only_edit(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        with self.assertRaisesRegex(ValueError, "at least two components"):
            _parse_technology_edit(
                {
                    "base_harness_version": manifest.version,
                    "edits": [
                        {
                            "component": "generator",
                            "patch": {"allowed_buckets": list(range(10))},
                            "rationale": "Expand source coverage.",
                        }
                    ],
                    "rationale": "Invalid joint proposal.",
                },
                current=manifest,
                proposal_mode="dependency-aware-joint",
            )

    def test_joint_parser_accepts_selector_dependency_closure(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        proposal = _parse_technology_edit(
            {
                "base_harness_version": manifest.version,
                "edits": [
                    {
                        "component": "generator",
                        "patch": {"allowed_buckets": list(range(8))},
                        "rationale": "Focus generation on observed buckets.",
                    },
                    {
                        "component": "selector",
                        "patch": {"max_response_words": 120},
                        "rationale": "Align selection with shorter responses.",
                    },
                ],
                "rationale": "Coordinate generation and selection.",
            },
            current=manifest,
            proposal_mode="dependency-aware-joint",
        )
        self.assertEqual(
            [edit.component for edit in proposal.edits], ["generator", "selector"]
        )

    def test_parser_requires_dependency_closure(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        with self.assertRaisesRegex(ValueError, "dependency closure"):
            _parse_technology_edit(
                {
                    "base_harness_version": manifest.version,
                    "edits": [
                        {
                            "component": "selector",
                            "patch": {"max_response_words": 120},
                            "rationale": "Increase response coverage.",
                        }
                    ],
                    "rationale": "Test selector coverage.",
                },
                current=manifest,
            )

    def test_parser_accepts_selector_with_generator_closure(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        proposal = _parse_technology_edit(
            {
                "base_harness_version": manifest.version,
                "edits": [
                    {
                        "component": "generator",
                        "patch": {"allowed_buckets": list(range(10))},
                        "rationale": "Restore observed source coverage.",
                    },
                    {
                        "component": "selector",
                        "patch": {"max_response_words": 120},
                        "rationale": "Reduce response truncation.",
                    },
                ],
                "rationale": "Coordinate dependent coverage and selection edits.",
            },
            current=manifest,
        )
        candidate = apply_edits(manifest, proposal.edits)
        self.assertEqual(candidate.components["selector"].config["max_response_words"], 120)
        self.assertEqual(proposal.base_version, manifest.version)

    def test_independent_mode_accepts_one_component_without_closure(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        proposal = _parse_technology_edit(
            {
                "base_harness_version": manifest.version,
                "edits": [
                    {
                        "component": "selector",
                        "patch": {"max_response_words": 120},
                        "rationale": "Test the selector contribution independently.",
                    }
                ],
                "rationale": "Measure one component without dependency closure.",
            },
            current=manifest,
            proposal_mode="independent",
        )
        self.assertEqual(len(proposal.edits), 1)
        self.assertEqual(proposal.edits[0].component, "selector")

    def test_independent_mode_still_rejects_joint_edits(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        with self.assertRaisesRegex(ValueError, "exactly one component"):
            _parse_technology_edit(
                {
                    "base_harness_version": manifest.version,
                    "edits": [
                        {
                            "component": "generator",
                            "patch": {"allowed_buckets": list(range(10))},
                            "rationale": "Expand source coverage.",
                        },
                        {
                            "component": "selector",
                            "patch": {"max_response_words": 120},
                            "rationale": "Test selection independently.",
                        },
                    ],
                    "rationale": "Invalid independent joint edit.",
                },
                current=manifest,
                proposal_mode="independent",
            )

    def test_parser_rejects_effective_noop(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        with self.assertRaisesRegex(ValueError, "no-op"):
            _parse_technology_edit(
                {
                    "base_harness_version": manifest.version,
                    "edits": [
                        {
                            "component": "generator",
                            "patch": {"allowed_buckets": list(range(9))},
                            "rationale": "Keep the current coverage unchanged.",
                        }
                    ],
                    "rationale": "No effective configuration change.",
                },
                current=manifest,
            )

    def test_effective_config_fingerprint_ignores_version_history(self) -> None:
        manifest = initial_manifest(train_group_budget=10)
        same_config_new_version = apply_edits(
            manifest,
            (
                HarnessEdit(
                    "generator",
                    {"allowed_buckets": list(range(9))},
                    "Record an otherwise identical configuration.",
                ),
            ),
        )
        self.assertNotEqual(manifest.version, same_config_new_version.version)
        self.assertEqual(
            _harness_config_fingerprint(manifest),
            _harness_config_fingerprint(same_config_new_version),
        )


if __name__ == "__main__":
    unittest.main()
