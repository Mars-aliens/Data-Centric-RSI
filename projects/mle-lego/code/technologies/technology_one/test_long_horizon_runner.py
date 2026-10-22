from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .long_horizon_runner import LongHorizonQwenRunner
from .long_horizon_task import (
    LongHorizonTaskConfig,
    prepare_helpsteer2_long_horizon_task,
)
from .server_adapters import JsonCommand


def _rows(prefix: str, count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(count):
        for response, score in (("better", 4), ("worse", 0)):
            result.append(
                {
                    "prompt": f"{prefix} prompt {index}",
                    "response": f"{response} response",
                    "helpfulness": score,
                    "correctness": score,
                    "coherence": score,
                    "complexity": score,
                    "verbosity": score,
                }
            )
    return result


def _task():
    return prepare_helpsteer2_long_horizon_task(
        _rows("train", 800),
        _rows("validation", 30),
        config=LongHorizonTaskConfig(
            stream_seed=23,
            groups_per_batch=100,
            feedback_delay_batches=2,
        ),
    )


class FakeQwenCommand:
    def __init__(self, *, fail_on_prediction: int | None = None):
        self.calls: list[dict[str, Any]] = []
        self.train_count = 0
        self.prediction_count = 0
        self.fail_on_prediction = fail_on_prediction

    def __call__(self, _command: JsonCommand, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        if request["operation"] == "train":
            self.train_count += 1
            version = f"model-v{self.train_count}"
            parent = request["initial_checkpoint"]
            return {
                "checkpoint": {
                    "task_model_version": version,
                    "adapter_hash": f"hash-v{self.train_count}",
                    "parent_checkpoint": (
                        {
                            "task_model_version": parent["task_model_version"],
                            "adapter_hash": parent["adapter_hash"],
                        }
                        if parent is not None
                        else None
                    ),
                },
                "cost": {
                    "training_steps": 2,
                    "training_records_seen": len(request["records"]),
                    "training_seconds": 0.1,
                },
            }
        self.prediction_count += 1
        if self.prediction_count == self.fail_on_prediction:
            raise RuntimeError("injected prediction interruption")
        return {
            "predictions": [
                {"record_id": item["record_id"], "prediction": "preferred"}
                for item in request["records"]
            ],
            "cost": {
                "prediction_records": len(request["records"]),
                "prediction_seconds": 0.05,
            },
        }


class FakeCandidateShadow:
    candidate_id = "candidate-test"

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def run_signature(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "fixture": "fake-v1", "fail": self.fail}

    def run_metadata(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "calls": self.calls}

    def run_batch(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected candidate failure")
        records = kwargs["batch"]["records"]
        return {
            "candidate_id": self.candidate_id,
            "candidate_version_id": "candidate-version-test",
            "status": "eligible_for_observation",
            "predictions": [
                {"record_id": item["record_id"], "prediction": "preferred"}
                for item in records
            ],
        }


class LongHorizonQwenRunnerTests(unittest.TestCase):
    def _runner(self, root: Path, fake: FakeQwenCommand) -> LongHorizonQwenRunner:
        return LongHorizonQwenRunner(
            _task(),
            command=JsonCommand.from_value(("fake-qwen",)),
            config={"max_steps": 2, "base_model_path": "local-test-model"},
            output_dir=root,
            run_id="long-horizon-test",
            seed=23,
            command_runner=fake,
        )

    def test_stream_predicts_before_release_and_trains_only_on_available_feedback(self) -> None:
        with TemporaryDirectory() as directory:
            fake = FakeQwenCommand()
            runner = self._runner(Path(directory) / "run", fake)
            task = runner.task
            summary = runner.run()

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed_batches"], task.scheduler.batch_count)
            self.assertGreaterEqual(task.scheduler.batch_count, 4)
            history = summary["batch_history"]
            self.assertEqual(history[0]["released_feedback_ids"], [])
            self.assertEqual(history[1]["released_feedback_ids"], [])
            self.assertEqual(history[2]["released_feedback_ids"], [])
            self.assertTrue(history[3]["released_feedback_ids"])
            self.assertEqual(
                [item["model_version_id"] for item in history[:4]],
                ["model-v1", "model-v1", "model-v1", "model-v1"],
            )

            operations = [item["operation"] for item in fake.calls]
            self.assertEqual(
                operations[:6],
                ["train", "predict", "predict", "predict", "predict", "train"],
            )
            first_update = fake.calls[5]
            expected_ids = {
                item.record_id for item in task.agent_view.seed_examples
            } | {item.record_id for item in task.scheduler.batches[0].records}
            self.assertEqual(
                {item["record_id"] for item in first_update["records"]},
                expected_ids,
            )
            self.assertEqual(
                first_update["initial_checkpoint"]["task_model_version"], "model-v1"
            )
            self.assertEqual(
                fake.calls[4]["checkpoint"]["task_model_version"], "model-v1"
            )
            self.assertEqual(history[3]["active_model_version_after_batch"], "model-v2")
            for request in fake.calls:
                if request["operation"] == "predict":
                    self.assertTrue(
                        all("target" not in item and "scores" not in item for item in request["records"])
                    )
            self.assertFalse(summary["held_out_used"])
            self.assertFalse(summary["final_test_used"])
            self.assertEqual(
                summary["command_attempt_audit"]["counts"]["completed"],
                len(fake.calls),
            )
            self.assertIsNotNone(summary["prequential_metrics"])
            self.assertEqual(
                summary["prequential_metrics"]["num_records"],
                sum(item["num_records"] for item in summary["terminal_metrics"])
                + sum(
                    metric["num_records"]
                    for batch in history
                    for metric in batch["released_metrics"]
                ),
            )
            for batch in history:
                artifact = Path(directory) / "run" / batch["prediction_artifact"]
                self.assertTrue(artifact.is_file())
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    batch["prediction_sha256"],
                )

            call_count = len(fake.calls)
            resumed = self._runner(Path(directory) / "run", fake)
            resumed_summary = resumed.run()
            self.assertEqual(resumed_summary["status"], "complete")
            self.assertEqual(len(fake.calls), call_count)

    def test_interrupted_batch_restores_last_committed_state_and_resumes(self) -> None:
        with TemporaryDirectory() as directory:
            fake = FakeQwenCommand(fail_on_prediction=2)
            runner = self._runner(Path(directory) / "run", fake)
            with self.assertRaisesRegex(RuntimeError, "injected prediction interruption"):
                runner.run()
            self.assertEqual(runner.state["completed_batches"], 1)
            self.assertEqual(
                runner.state["task_state"]["scheduler"]["next_batch_index"], 1
            )
            attempts = [
                json.loads(line)
                for line in (Path(directory) / "run" / "command_attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(any(item["status"] == "failed" for item in attempts))
            self.assertTrue(all(item.get("request_sha256") for item in attempts if item["status"] == "started"))

            fake.fail_on_prediction = None
            summary = runner.run()
            self.assertEqual(summary["status"], "complete")
            self.assertGreater(summary["completed_batches"], 1)
            self.assertEqual(summary["command_attempt_audit"]["counts"]["failed"], 1)
            self.assertEqual(fake.train_count, 1 + sum(
                1 for item in summary["batch_history"] if item["trained_this_batch"]
            ))

    def test_run_state_rejects_different_stream_or_training_config(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            runner = self._runner(root, FakeQwenCommand())
            runner.run()
            with self.assertRaisesRegex(ValueError, "does not match this invocation"):
                LongHorizonQwenRunner(
                    _task(),
                    command=JsonCommand.from_value(("fake-qwen",)),
                    config={"max_steps": 99},
                    output_dir=root,
                    run_id="long-horizon-test",
                    seed=23,
                    command_runner=FakeQwenCommand(),
                )

    def test_trainer_cannot_return_checkpoint_with_wrong_parent_lineage(self) -> None:
        with TemporaryDirectory() as directory:
            class WrongLineage(FakeQwenCommand):
                def __call__(self, command: JsonCommand, request: dict[str, Any]):
                    response = super().__call__(command, request)
                    if request["operation"] == "train" and request["initial_checkpoint"]:
                        response["checkpoint"]["parent_checkpoint"] = {
                            "task_model_version": "wrong-parent",
                            "adapter_hash": "wrong-hash",
                        }
                    return response

            fake = WrongLineage()
            runner = self._runner(Path(directory) / "run", fake)
            with self.assertRaisesRegex(ValueError, "parent provenance"):
                runner.run()
            self.assertEqual(runner.state["completed_batches"], 3)

    def test_candidate_shadow_metrics_are_separate_and_never_promoted(self) -> None:
        with TemporaryDirectory() as directory:
            candidate = FakeCandidateShadow()
            runner = LongHorizonQwenRunner(
                _task(),
                command=JsonCommand.from_value(("fake-qwen",)),
                config={"max_steps": 2, "base_model_path": "local-test-model"},
                output_dir=Path(directory) / "run",
                run_id="t1-shadow-test",
                seed=23,
                command_runner=FakeQwenCommand(),
                candidate_shadow=candidate,
            )
            summary = runner.run()

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(candidate.calls, summary["completed_batches"])
        self.assertIsNotNone(summary["prequential_metrics"])
        candidate_metrics = summary["candidate_prequential_metrics"][
            "candidate-version-test"
        ]
        self.assertIsNotNone(candidate_metrics)
        self.assertEqual(
            summary["prequential_metrics"]["num_records"],
            candidate_metrics["num_records"],
        )
        self.assertEqual(summary["candidate_shadow"]["candidate_id"], "candidate-test")
        self.assertTrue(all(
            item["candidate_trial"]["status"] == "eligible_for_observation"
            for item in summary["batch_history"]
        ))

    def test_candidate_failure_does_not_stop_stable_stream(self) -> None:
        with TemporaryDirectory() as directory:
            candidate = FakeCandidateShadow(fail=True)
            runner = LongHorizonQwenRunner(
                _task(),
                command=JsonCommand.from_value(("fake-qwen",)),
                config={"max_steps": 2, "base_model_path": "local-test-model"},
                output_dir=Path(directory) / "run",
                run_id="t1-failure-isolation-test",
                seed=23,
                command_runner=FakeQwenCommand(),
                candidate_shadow=candidate,
            )
            summary = runner.run()

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_batches"], summary["target_batches"])
        self.assertEqual(candidate.calls, summary["target_batches"])
        self.assertTrue(all(
            item["candidate_trial"]["status"] == "execution_error"
            for item in summary["batch_history"]
        ))


if __name__ == "__main__":
    unittest.main()
