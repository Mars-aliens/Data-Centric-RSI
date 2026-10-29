from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "t1_kubernetes_runner_test", PROJECT_ROOT / "scripts/run_long_horizon_t1_qwen.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class KubernetesPreflightRegressionTests(unittest.TestCase):
    @staticmethod
    def _configuration(root: Path):
        candidate = root / "candidate"
        base = root / "model"
        candidate.mkdir()
        (candidate / "candidate_strategy.json").write_text(
            '{"schema":"lego-t1-data-strategy-v1","data_policy":'
            '{"min_input_chars":1,"deduplicate_inputs":true,"max_feedback_examples":null}}',
            encoding="utf-8",
        )
        base.mkdir()
        task = {
            "base_model_path": str(base), "base_model_revision": "immutable-snapshot-v1",
            "base_model_id": "test-model",
        }
        config = {
            "candidate_id": "test-candidate", "candidate_dir": str(candidate),
            "require_proposal_manifest": False,
            "sandbox_backend": "kubernetes",
            "image": "registry.example/candidate@sha256:" + "a" * 64,
            "command": [
                "python3", "-m", "technologies.technology_one.long_horizon_candidate_qwen_command"
            ],
            "kubernetes": {
                "shared_root": str(root), "pvc_name": "lego-shared",
                "network_policy_name": "lego-t1-deny-egress",
                "network_policy_enforcement_verified": True,
                "gpu_count": 1,
            },
        }
        return task, config

    def _invoke(self, root, task, config, *, output=None):
        task_file, candidate_file = root / "task.json", root / "candidate.json"
        task_file.write_text(json.dumps(task), encoding="utf-8")
        candidate_file.write_text(json.dumps(config), encoding="utf-8")
        argv = [
            "run_long_horizon_t1_qwen.py", "--train-jsonl", "unused-train",
            "--validation-jsonl", "unused-validation", "--task-config-json", str(task_file),
            "--candidate-config-json", str(candidate_file), "--output-dir",
            str(output or root / "output"), "--run-id", "test-preflight",
            "--max-batches", "4", "--preflight-only",
        ]
        stream = SimpleNamespace(
            agent_view=SimpleNamespace(manifest={}),
            scheduler=SimpleNamespace(batch_count=4),
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(RUNNER.sys, "argv", argv), patch.object(
            RUNNER.shutil, "which", return_value="kubectl"
        ), patch.object(
            RUNNER.KubernetesJobCandidateSandbox, "preflight"
        ) as preflight, patch.object(
            RUNNER, "_read_jsonl", return_value=[]
        ), patch.object(
            RUNNER, "prepare_helpsteer2_long_horizon_task", return_value=stream
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                RUNNER.main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue(), preflight.call_count

    def test_k8s_regression_preflight_uses_kubernetes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            task, config = self._configuration(root)
            code, stdout, stderr, calls = self._invoke(root, task, config)
        self.assertEqual(code, 0, stderr)
        report = json.loads(stdout)
        self.assertEqual(report["status"], "preflight_ok")
        self.assertEqual(report["sandbox_backend"], "kubernetes")
        self.assertEqual(report["sandbox_configuration"]["gpu_count"], 1)
        self.assertEqual(report["fault_injection"], {})
        self.assertEqual(calls, 1)

    def test_k8s_regression_preflight_rejects_ignored_options_and_outside_paths(self):
        cases = {
            "fault": "fault_injection is not supported",
            "pids": "does not implement per-Pod pids_limit",
            "gpu": "physical gpu_devices",
            "unused_mount_path": "unsupported Kubernetes settings",
            "candidate_path": "inside shared_root",
            "model_path": "inside shared_root",
            "output_path": "inside shared_root",
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                shared = root / "shared"
                shared.mkdir()
                task, config = self._configuration(shared)
                output = shared / "output"
                if case == "fault":
                    config["fault_injection"] = {"1": "timeout"}
                elif case == "pids":
                    config["limits"] = {"pids_limit": 64}
                elif case == "gpu":
                    config["limits"] = {"gpu_devices": "0"}
                elif case == "unused_mount_path":
                    config["kubernetes"]["pvc_mount_path"] = "/ignored"
                elif case == "output_path":
                    output = root / "outside-output"
                else:
                    outside = root / "outside"
                    outside.mkdir()
                    if case == "candidate_path":
                        config["candidate_dir"] = str(outside)
                    else:
                        task["base_model_path"] = str(outside)
                code, _stdout, stderr, calls = self._invoke(
                    root, task, config, output=output
                )
                self.assertEqual(code, 2)
                self.assertIn(expected, stderr)
                self.assertNotIn("Traceback", stderr)
                self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
