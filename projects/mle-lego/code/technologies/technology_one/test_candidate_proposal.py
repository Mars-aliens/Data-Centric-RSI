from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.propose_t1_candidate import propose_candidate

from .candidate_proposal import (
    CandidateProposalError,
    build_candidate_prompt,
    read_candidate_manifest,
    validate_candidate_proposal,
    write_candidate_bundle,
)


class CandidateProposalTests(unittest.TestCase):
    def _proposal(self, **edit: object) -> dict[str, object]:
        return {
            "rationale": "A bounded candidate change for the stream.",
            "edits": [
                {
                    "path": "candidate.py",
                    "operation": "write",
                    "content": "candidate = True\n",
                    **edit,
                }
            ],
        }

    def test_validation_rejects_path_escape_duplicate_and_oversized_edit(self) -> None:
        for path in ("../outside.py", "/tmp/outside.py", "a//b.py", "a/./b.py"):
            with self.subTest(path=path), self.assertRaises(CandidateProposalError):
                validate_candidate_proposal(
                    self._proposal(path=path),
                    candidate_id="t1-candidate-01",
                    base_version_id="runtime-stable",
                )

        with self.assertRaisesRegex(CandidateProposalError, "duplicate edit path"):
            validate_candidate_proposal(
                {
                    **self._proposal(),
                    "edits": [self._proposal()["edits"][0], self._proposal()["edits"][0]],
                },
                candidate_id="t1-candidate-01",
                base_version_id="runtime-stable",
            )
        with self.assertRaisesRegex(CandidateProposalError, "per-file size"):
            validate_candidate_proposal(
                {
                    "rationale": "bounded",
                    "edits": [{"path": "candidate.py", "operation": "write", "content": "x" * 300_000}],
                },
                candidate_id="t1-candidate-01",
                base_version_id="runtime-stable",
            )

    def test_candidate_bundle_filters_private_artifacts_and_records_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text("candidate = False\n", encoding="utf-8")
            (source / ".env.local").write_text("SECRET=value\n", encoding="utf-8")
            (source / "dataset.jsonl").write_text('{"target":"secret"}\n', encoding="utf-8")
            (source / "runs").mkdir()
            (source / "runs" / "old.json").write_text("{}\n", encoding="utf-8")
            (source / ".git").mkdir()
            (source / ".git" / "config").write_text("private\n", encoding="utf-8")
            destination = root / "candidate"

            manifest = write_candidate_bundle(
                base_dir=source,
                output_dir=destination,
                proposal=self._proposal(),
                candidate_id="t1-candidate-01",
                base_version_id="runtime-stable",
                decision_model={"provider": "bailian_openai_compatible", "model_id": "kimi-k3"},
                prompt_hashes={"system_sha256": "a" * 64},
                proposal_cost={"llm_calls": 1, "llm_tokens": 50},
            )

            self.assertEqual((destination / "candidate.py").read_text(encoding="utf-8"), "candidate = True\n")
            self.assertFalse((destination / ".env.local").exists())
            self.assertFalse((destination / "dataset.jsonl").exists())
            self.assertFalse((destination / "runs").exists())
            self.assertFalse((destination / ".git").exists())
            self.assertEqual(read_candidate_manifest(destination), manifest)
            self.assertEqual(manifest["candidate_id"], "t1-candidate-01")

            (destination / "candidate.py").write_text("tampered = True\n", encoding="utf-8")
            with self.assertRaisesRegex(CandidateProposalError, "do not match"):
                read_candidate_manifest(destination)

    def test_failed_patch_leaves_no_partial_candidate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "directory.txt").mkdir()
            destination = root / "candidate"
            proposal = {
                "rationale": "Attempt an invalid file deletion.",
                "edits": [{"path": "directory.txt", "operation": "delete"}],
            }

            with self.assertRaisesRegex(CandidateProposalError, "only remove files"):
                write_candidate_bundle(
                    base_dir=source,
                    output_dir=destination,
                    proposal=proposal,
                    candidate_id="t1-candidate-01",
                    base_version_id="runtime-stable",
                    decision_model={},
                    prompt_hashes={},
                    proposal_cost={},
                )

            self.assertFalse(destination.exists())
            self.assertFalse(list(root.glob(".candidate.*.tmp")))

    def test_rejects_overlapping_directories_and_source_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                write_candidate_bundle(
                    base_dir=source,
                    output_dir=source / "nested-output",
                    proposal=self._proposal(),
                    candidate_id="t1-candidate-01",
                    base_version_id="runtime-stable",
                    decision_model={},
                    prompt_hashes={},
                    proposal_cost={},
                )

            outside = root / "outside.py"
            outside.write_text("secret = True\n", encoding="utf-8")
            try:
                os.symlink(outside, source / "linked.py")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable in this Windows environment")
            with self.assertRaisesRegex(CandidateProposalError, "symlink"):
                build_candidate_prompt(
                    base_dir=source,
                    observation={"event": "regression"},
                    candidate_id="t1-candidate-01",
                    base_version_id="runtime-stable",
                )

    def test_candidate_observation_rejects_unreleased_labels_and_evaluation_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "runner.py").write_text("pass\n", encoding="utf-8")
            for observation in (
                {"current_batch": {"target": "preferred"}},
                {"metrics": {"held-out": {"accuracy": 0.9}}},
                {"unreleased_feedback": [{"target": "preferred"}]},
            ):
                with self.subTest(observation=observation), self.assertRaises(CandidateProposalError):
                    build_candidate_prompt(
                        base_dir=source,
                        observation=observation,
                        candidate_id="t1-candidate-01",
                        base_version_id="runtime-stable",
                    )

            _system, user = build_candidate_prompt(
                base_dir=source,
                observation={
                    "released_feedback": [
                        {"record_id": "r0", "target": "preferred", "scores": {"helpfulness": 5}}
                    ]
                },
                candidate_id="t1-candidate-01",
                base_version_id="runtime-stable",
            )
            self.assertIn('"released_feedback"', user)

    def test_noop_patch_does_not_create_a_candidate_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text(
                "candidate = True\n", encoding="utf-8", newline="\n"
            )
            destination = root / "candidate"
            proposal = {
                "rationale": "Write the source file without changing it.",
                "edits": [
                    {"path": "candidate.py", "operation": "write", "content": "candidate = True\n"}
                ],
            }

            with self.assertRaisesRegex(CandidateProposalError, "does not change"):
                write_candidate_bundle(
                    base_dir=source,
                    output_dir=destination,
                    proposal=proposal,
                    candidate_id="t1-candidate-01",
                    base_version_id="runtime-stable",
                    decision_model={},
                    prompt_hashes={},
                    proposal_cost={},
                )
            self.assertFalse(destination.exists())

    def test_proposal_cli_retries_schema_errors_and_records_model_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "code"
            source.mkdir()
            (source / "candidate.py").write_text("candidate = True\n", encoding="utf-8")
            output = root / "candidate-output"
            responses = [
                {"proposal": {"rationale": "invalid", "edits": []}, "cost": {"llm_calls": 1, "llm_tokens": 5}},
                {
                    "proposal": {
                        "rationale": "Change one bounded runtime rule.",
                        "edits": [
                            {"path": "candidate.py", "operation": "write", "content": "candidate = False\n"}
                        ],
                    },
                    "cost": {"llm_calls": 1, "llm_tokens": 9},
                },
            ]
            with patch("scripts.propose_t1_candidate.call_endpoint", side_effect=responses) as call:
                manifest = propose_candidate(
                    profile={
                        "decision_model": {
                            "provider": "bailian_openai_compatible",
                            "model_id": "qwen-or-kimi-model-id",
                            "revision": "provider-revision",
                            "api_url_env": "LEGO_POLICY_API_URL",
                            "api_key_env": "LEGO_POLICY_API_KEY",
                        }
                    },
                    source_dir=source,
                    output_dir=output,
                    observation={"recent_events": ["delayed feedback released"]},
                    candidate_id="t1-candidate-01",
                    base_version_id="runtime-stable",
                )

            self.assertEqual(call.call_count, 2)
            self.assertIn("Schema correction", call.call_args_list[1].args[0]["prompt"]["user"])
            self.assertEqual(manifest["decision_model"]["model_id"], "qwen-or-kimi-model-id")
            self.assertEqual(manifest["proposal_cost"], {"llm_calls": 2, "llm_tokens": 14})
            self.assertTrue((output / "candidate_proposal.json").is_file())


if __name__ == "__main__":
    unittest.main()
