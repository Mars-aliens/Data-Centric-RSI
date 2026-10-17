from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from .server_technical_one_evolution import run_server_technology_one_evolution
from .server_adapters import ServerCommandError


class TechnologyOneEvolutionIntegrationTests(unittest.TestCase):
    def _configure_test_dataset(
        self, profile: dict, temp: Path, repo_root: Path
    ) -> None:
        """Use a tiny temporary dataset so integration tests need no real data."""
        data_root = temp / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        train_records = [
            {
                "record_id": f"seed-{index}-{choice}",
                "split": "seed",
                "input": f"Prompt: seed {index} Response: answer {choice}",
                "target": choice,
                "source_group_id": f"group-seed-{index}",
                "batch_id": f"seed-{index}",
            }
            for index in range(60)
            for choice in range(2)
        ] + [
            {
                "record_id": f"held-in-0-{choice}",
                "split": "held_in",
                "input": f"Prompt: held in Response: answer {choice}",
                "target": choice,
                "source_group_id": "group-held-in-0",
                "batch_id": "held-in-0",
            }
            for choice in range(2)
        ] + [
            {
                "record_id": f"held-out-0-{choice}",
                "split": "held_out",
                "input": f"Prompt: held out Response: answer {choice}",
                "target": choice,
                "source_group_id": "group-held-out-0",
                "batch_id": "held-out-0",
            }
            for choice in range(2)
        ]
        final_records = [
            {
                "record_id": f"final-0-{choice}",
                "split": "final_test",
                "input": f"Prompt: final Response: answer {choice}",
                "target": choice,
                "source_group_id": "group-final-0",
            }
            for choice in range(2)
        ]
        train_path = data_root / "train.jsonl"
        final_path = data_root / "final.jsonl"
        train_path.write_text(
            "\n".join(json.dumps(item) for item in train_records) + "\n",
            encoding="utf-8",
        )
        final_path.write_text(
            "\n".join(json.dumps(item) for item in final_records) + "\n",
            encoding="utf-8",
        )
        freeze_path = data_root / "freeze.json"
        freeze_path.write_text(json.dumps({"files": []}), encoding="utf-8")
        profile["dataset"].update(
            {
                "train_jsonl": str(train_path),
                "final_test_jsonl": str(final_path),
                "freeze_manifest": str(freeze_path),
                "protocol": str(
                    repo_root / "code/technologies/technology_one/data_protocol.json"
                ),
            }
        )

    def _fake_profile(
        self, temp: Path, repo_root: Path, decision_source: str
    ) -> Path:
        decision = temp / "decision.py"
        trainer = temp / "trainer.py"
        evaluator = temp / "evaluator.py"
        decision.write_text(decision_source, encoding="utf-8")
        trainer.write_text(
            "import json,sys\n"
            "request=json.load(sys.stdin)\n"
            "json.dump({'checkpoint':{'record_count':len(request['records']),'seed':request['seed']},'cost':{'training_steps':1}},sys.stdout)\n",
            encoding="utf-8",
        )
        evaluator.write_text(
            "import json,sys\n"
            "request=json.load(sys.stdin)\n"
            "json.dump({'metrics':{'macro_f1':0.25,'num_records':float(len(request['records']))},'cost':{'evaluator_calls':1}},sys.stdout)\n",
            encoding="utf-8",
        )
        source_profile = repo_root / "code/technologies/technology_one/server_profile.evolution.qwen.example.json"
        profile = json.loads(source_profile.read_text(encoding="utf-8"))
        profile["commands"] = {
            "policy": [sys.executable, str(decision)],
            "trainer": [sys.executable, str(trainer)],
            "evaluator": [sys.executable, str(evaluator)],
        }
        profile["action_system"]["command_cwd"] = str(repo_root)
        self._configure_test_dataset(profile, temp, repo_root)
        profile["task_model"]["training_config"]["max_steps"] = 1
        profile["stream"]["max_batches"] = 3
        profile["budget"].update(
            {"training_steps": 20, "evaluator_calls": 20, "llm_calls": 20, "llm_tokens": 100}
        )
        profile_path = temp / "profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        return profile_path

    def test_three_batch_fake_server_records_llm_reflections_and_rollback(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        source_profile = repo_root / "code/technologies/technology_one/server_profile.evolution.qwen.example.json"
        profile = json.loads(source_profile.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            decision = temp / "decision.py"
            trainer = temp / "trainer.py"
            evaluator = temp / "evaluator.py"
            decision.write_text(
                "import json,sys\n"
                "request=json.load(sys.stdin)\n"
                "payload=json.loads(request['prompt']['user'])\n"
                "current=payload['current_harness']['version']\n"
                "response={'proposal':{'base_harness_version':current,'edits':[{'component':'generator','patch':{'allowed_buckets':list(range(10))},'rationale':'Expand the observed source coverage.'}],'rationale':'Run a bounded coverage edit.'},'reflection':{'problem':'No observed gain yet.','evidence':{'batch_index':0},'lesson':'Expand coverage only when held-in improves.','conditions':{'task_id':'helpsteer2-preference'},'source_batch':'pilot-batch-0000'},'cost':{'llm_calls':1,'llm_tokens':7}}\n"
                "json.dump(response,sys.stdout)\n",
                encoding="utf-8",
            )
            trainer.write_text(
                "import json,sys\n"
                "request=json.load(sys.stdin)\n"
                "json.dump({'checkpoint':{'record_count':len(request['records']),'seed':request['seed']},'cost':{'training_steps':1}},sys.stdout)\n",
                encoding="utf-8",
            )
            evaluator.write_text(
                "import json,sys\n"
                "request=json.load(sys.stdin)\n"
                "json.dump({'metrics':{'macro_f1':0.25,'num_records':float(len(request['records']))},'cost':{'evaluator_calls':1}},sys.stdout)\n",
                encoding="utf-8",
            )
            argv = [sys.executable]
            profile["commands"] = {
                "policy": argv + [str(decision)],
                "trainer": argv + [str(trainer)],
                "evaluator": argv + [str(evaluator)],
            }
            profile["action_system"]["command_cwd"] = str(repo_root)
            self._configure_test_dataset(profile, temp, repo_root)
            profile["task_model"]["training_config"]["max_steps"] = 1
            profile["stream"]["max_batches"] = 3
            profile["budget"].update(
                {"training_steps": 5, "evaluator_calls": 10, "llm_calls": 4, "llm_tokens": 100}
            )
            profile_path = temp / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            result = run_server_technology_one_evolution(
                profile_path,
                root=repo_root,
                output_dir=temp / "runs",
                run_id="fake-evolution",
                seed=17,
                batches=3,
                max_train_groups=3000,
                max_eval_groups=1000,
                train_group_budget=20,
            )

            self.assertTrue(result["decision_llm_used"])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["run_status"], "complete")
            self.assertEqual(result["batch_count"], 3)
            self.assertEqual(result["budget"]["llm_calls"], 2)
            self.assertEqual(len(result["reflections"]), 2)
            self.assertEqual(result["rejected_candidate_count"], 2)
            self.assertEqual(result["rollback_count"], 2)
            self.assertFalse(result["final_test_used"])

    def test_resume_after_policy_outage_keeps_completed_batch(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            decision_source = (
                "import json,os,pathlib,sys\n"
                "counter=pathlib.Path(os.environ['POLICY_COUNTER'])\n"
                "calls=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(calls+1))\n"
                "request=json.load(sys.stdin)\n"
                "if calls == 0: sys.exit(1)\n"
                "payload=json.loads(request['prompt']['user'])\n"
                "version=payload['current_harness']['version']\n"
                "json.dump({'proposal':{'base_harness_version':version,'edits':[{'component':'generator','patch':{'allowed_buckets':list(range(10))},'rationale':'Expand the current data coverage policy.'}],'rationale':'Try a bounded coverage update.'},'cost':{'llm_calls':1,'llm_tokens':5}},sys.stdout)\n"
            )
            profile_path = self._fake_profile(temp, repo_root, decision_source)
            counter_path = temp / "policy-counter"
            previous = __import__("os").environ.get("POLICY_COUNTER")
            __import__("os").environ["POLICY_COUNTER"] = str(counter_path)
            try:
                with self.assertRaises(ServerCommandError):
                    run_server_technology_one_evolution(
                        profile_path,
                        root=repo_root,
                        output_dir=temp / "runs",
                        run_id="resume-test",
                        seed=17,
                        batches=3,
                        train_group_budget=20,
                    )

                run_root = temp / "runs" / "resume-test"
                state = json.loads((run_root / "endurance_state.json").read_text())
                self.assertEqual(state["phase"], "proposal_pending")
                self.assertEqual(state["next_batch_index"], 1)
                self.assertEqual(len(state["batches"]), 1)

                result = run_server_technology_one_evolution(
                    profile_path,
                    root=repo_root,
                    output_dir=temp / "runs",
                    run_id="resume-test",
                    seed=17,
                    batches=3,
                    train_group_budget=20,
                    resume=True,
                )
                self.assertEqual(result["batch_count"], 3)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["batches"]), 3)
                self.assertEqual(result["reliability"]["resume_count"], 1)
                self.assertFalse(result["final_test_used"])
                summary_path = run_root / "summary.json"
                self.assertTrue(summary_path.exists())
                persisted = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertEqual(persisted["status"], "complete")
            finally:
                if previous is None:
                    __import__("os").environ.pop("POLICY_COUNTER", None)
                else:
                    __import__("os").environ["POLICY_COUNTER"] = previous


if __name__ == "__main__":
    unittest.main()
