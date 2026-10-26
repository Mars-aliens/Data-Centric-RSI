from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from core_system.mle_core.store import ArtifactStore

from .candidate_proposal import write_candidate_bundle
from .candidate_sandbox import SandboxLimits, SandboxResult, hash_directory
from .candidate_shadow import LongHorizonCandidateShadow


class FakeSandbox:
    backend_name = "docker"
    image = "fake@sha256:" + "a" * 64
    limits = SandboxLimits(timeout_seconds=60, memory_gib=4, cpu_cores=2)

    def __init__(
        self, *, fail: bool = False, invalid: bool = False, backend_name: str = "docker"
    ):
        self.calls: list[dict[str, Any]] = []
        self.fail = fail
        self.invalid = invalid
        self.backend_name = backend_name

    def execute(self, *, command, payload, mounts, output_dir, environment=None):
        self.calls.append(payload)
        self.assert_safe_payload(payload)
        if self.fail:
            return SandboxResult(
                status="timed_out",
                backend=self.backend_name,
                elapsed_seconds=60,
                returncode=None,
                response=None,
                response_sha256=None,
                failure_code="timeout",
                error="candidate timed out",
                stderr_tail="",
            )
        output = Path(output_dir) / "checkpoint"
        output.mkdir()
        (output / "adapter.bin").write_bytes(b"candidate adapter")
        checkpoint = {
            "backend": "qwen-lora-preference-v1",
            "task_model_version": "candidate-model-v1",
            "base_model_path": "/models/task",
            "base_model_revision": "immutable-task-model-v1",
            "adapter_path": "/candidate-output/checkpoint",
            "adapter_hash": hash_directory(output),
            "parent_checkpoint": {
                "task_model_version": payload["checkpoint"]["task_model_version"],
                "adapter_hash": payload["checkpoint"]["adapter_hash"],
            },
        }
        predictions = [
            {"record_id": item["record_id"], "prediction": "preferred"}
            for item in payload["records"]
        ]
        response = {
            "protocol": "lego-t1-candidate-step-v1",
            "candidate_id": payload["candidate_id"],
            "parent_version_id": payload["parent_version_id"],
            "predictions": predictions,
            "checkpoint": checkpoint,
            "cost": {
                "training_steps": 1,
                "training_records_seen": len(payload["seed_examples"]),
                "training_seconds": 0.5,
                "prediction_records": len(predictions),
                "prediction_seconds": 0.1,
            },
        }
        if self.invalid:
            response["predictions"][0]["prediction"] = "invalid-label"
        return SandboxResult(
            status="completed",
            backend=self.backend_name,
            elapsed_seconds=1,
            returncode=0,
            response=response,
            response_sha256="response-hash",
            failure_code=None,
            error=None,
            stderr_tail="",
        )

    @staticmethod
    def assert_safe_payload(payload: dict[str, Any]) -> None:
        assert all("target" not in row and "scores" not in row for row in payload["records"])
        assert all("scores" not in row for row in payload["released_feedback"])


def checkpoint(adapter: Path) -> dict[str, Any]:
    return {
        "backend": "qwen-lora-preference-v1",
        "task_model_version": "stable-model-v1",
        "base_model_path": str(adapter.parent / "base"),
        "base_model_revision": "immutable-task-model-v1",
        "adapter_path": str(adapter),
        "adapter_hash": hash_directory(adapter),
        "parent_checkpoint": None,
    }


class CandidateShadowTests(unittest.TestCase):
    def _setup(self, root: Path, sandbox: FakeSandbox):
        candidate_dir = root / "bundle"
        candidate_dir.mkdir(exist_ok=True)
        (candidate_dir / "candidate.py").write_text("candidate", encoding="utf-8")
        base = root / "base"
        base.mkdir(exist_ok=True)
        (base / "config.json").write_text("{}", encoding="utf-8")
        adapter = root / "stable-adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "adapter.bin").write_bytes(b"stable")
        shadow = LongHorizonCandidateShadow(
            candidate_id="t1-candidate-1",
            candidate_dir=candidate_dir,
            output_dir=root / "shadow",
            sandbox=sandbox,
            command=("python", "/candidate/candidate.py"),
            store=ArtifactStore(root / "events"),
            max_reported_training_steps=10,
        )
        return shadow, base, adapter

    def _run_batch(self, shadow, base: Path, adapter: Path, *, batch_index: int = 2):
        return shadow.run_batch(
            run_id="test-run",
            batch={
                "batch_id": f"batch-{batch_index:04d}",
                "batch_index": batch_index,
                "records": [{"record_id": "r1", "input": "prompt one"}],
            },
            released_feedback=[
                {
                    "feedback_id": "f0",
                    "record_id": "old-r",
                    "source_batch_index": 0,
                    "input": "old prompt",
                    "target": "preferred",
                    "scores": {"helpfulness": 5},
                }
            ],
            seed_examples=[
                {
                    "record_id": "seed-r",
                    "input": "seed prompt",
                    "target": "rejected",
                    "source_group_id": "seed-group",
                }
            ],
            stable_checkpoint=checkpoint(adapter),
            task_config={
                "base_model_path": str(base),
                "base_model_revision": "immutable-task-model-v1",
                "seed": 17,
                "prediction_batch_size": 3,
            },
            task_manifest_hash="task-manifest-hash",
        )

    def test_candidate_is_admitted_only_after_runtime_evidence_and_resume_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = FakeSandbox()
            shadow, base, adapter = self._setup(root, sandbox)
            stable_hash = hash_directory(adapter)
            first = self._run_batch(shadow, base, adapter)
            resumed = self._setup(root, sandbox)[0]
            repeated = self._run_batch(resumed, base, adapter)

        self.assertEqual(first["status"], "eligible_for_observation")
        self.assertEqual(sandbox.calls[0]["seed"], 20)
        self.assertEqual(sandbox.calls[0]["config"]["prediction_batch_size"], 3)
        self.assertEqual(first["gate"]["status"], "eligible_for_observation")
        self.assertEqual(first["gate"]["evidence"]["stable_parent_hash_before"], stable_hash)
        self.assertEqual(first["gate"]["evidence"]["stable_parent_hash_after"], stable_hash)
        self.assertEqual(first["predictions"], repeated["predictions"])
        self.assertEqual(len(sandbox.calls), 1)

    def test_execution_timeout_and_bad_predictions_reject_without_touching_stable(self) -> None:
        for sandbox in (FakeSandbox(fail=True), FakeSandbox(invalid=True)):
            with self.subTest(fail=sandbox.fail, invalid=sandbox.invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shadow, base, adapter = self._setup(root, sandbox)
                before = hash_directory(adapter)
                result = self._run_batch(shadow, base, adapter)
                after = hash_directory(adapter)

                self.assertEqual(result["status"], "rejected")
                self.assertIsNone(result["predictions"])
                self.assertEqual(before, after)
                self.assertEqual(result["gate"]["status"], "rejected")

    def test_successful_local_backend_is_rejected_by_safety_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shadow, base, adapter = self._setup(
                root, FakeSandbox(backend_name="local")
            )
            result = self._run_batch(shadow, base, adapter)

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["gate"]["failed_checks"], ["sandboxed"])
        self.assertEqual(result["gate"]["evidence"]["sandbox_backend"], "local")
        self.assertFalse(result["gate"]["evidence"]["container_isolation"])
        self.assertFalse(result["candidate_checkpoint_retained"])

    def test_candidate_retains_only_the_latest_active_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shadow, base, adapter = self._setup(root, FakeSandbox())
            self._run_batch(shadow, base, adapter, batch_index=2)
            first_checkpoint = json.loads(shadow.latest_path.read_text(encoding="utf-8"))["checkpoint"]
            first_adapter = Path(first_checkpoint["adapter_path"])
            self.assertTrue(first_adapter.is_dir())

            self._run_batch(shadow, base, adapter, batch_index=3)
            current_checkpoint = json.loads(shadow.latest_path.read_text(encoding="utf-8"))["checkpoint"]
            current_adapter = Path(current_checkpoint["adapter_path"])

            self.assertFalse(first_adapter.exists())
            self.assertTrue(current_adapter.is_dir())

    def test_proposal_manifest_is_checked_and_runtime_parent_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text("candidate = True\n", encoding="utf-8")
            bundle = root / "bundle"
            write_candidate_bundle(
                base_dir=source,
                output_dir=bundle,
                proposal={
                    "rationale": "Run this candidate on the current stable stream.",
                    "edits": [
                        {"path": "candidate.py", "operation": "write", "content": "candidate = False\n"}
                    ],
                },
                candidate_id="t1-candidate-1",
                base_version_id="runtime-stable",
                decision_model={"provider": "bailian_openai_compatible", "model_id": "qwen-test"},
                prompt_hashes={},
                proposal_cost={"llm_calls": 1, "llm_tokens": 10},
            )
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter.bin").write_bytes(b"stable")
            sandbox = FakeSandbox()
            shadow = LongHorizonCandidateShadow(
                candidate_id="t1-candidate-1",
                candidate_dir=bundle,
                output_dir=root / "shadow",
                sandbox=sandbox,
                command=("python", "/candidate/candidate.py"),
                store=ArtifactStore(root / "events"),
                require_proposal_manifest=True,
            )

            result = self._run_batch(shadow, base, adapter)

        self.assertEqual(result["status"], "eligible_for_observation")
        metadata = shadow.run_metadata()
        self.assertEqual(metadata["proposal_base_version_id"], "runtime-stable")
        self.assertEqual(metadata["proposal_base_version_binding"], result["parent_version_id"])
        self.assertTrue(metadata["proposal_manifest_sha256"])

    def test_proposal_manifest_rejects_wrong_declared_runtime_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text("candidate = True\n", encoding="utf-8")
            bundle = root / "bundle"
            write_candidate_bundle(
                base_dir=source,
                output_dir=bundle,
                proposal={
                    "rationale": "Try a candidate with a deliberately mismatched parent.",
                    "edits": [
                        {"path": "candidate.py", "operation": "write", "content": "candidate = False\n"}
                    ],
                },
                candidate_id="t1-candidate-1",
                base_version_id="stable-from-another-run",
                decision_model={},
                prompt_hashes={},
                proposal_cost={},
            )
            sandbox = FakeSandbox()
            shadow = LongHorizonCandidateShadow(
                candidate_id="t1-candidate-1",
                candidate_dir=bundle,
                output_dir=root / "shadow",
                sandbox=sandbox,
                command=("python", "/candidate/candidate.py"),
                store=ArtifactStore(root / "events"),
                require_proposal_manifest=True,
            )
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter.bin").write_bytes(b"stable")

            first = self._run_batch(shadow, base, adapter)
            later = self._run_batch(shadow, base, adapter, batch_index=3)

        self.assertEqual(first["status"], "rejected")
        self.assertEqual(first["failure_code"], "proposal_parent_mismatch")
        self.assertEqual(later["status"], "not_run_terminal")
        self.assertEqual(len(sandbox.calls), 0)


if __name__ == "__main__":
    unittest.main()
