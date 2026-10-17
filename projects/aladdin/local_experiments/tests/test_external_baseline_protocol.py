from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_experiments.aladdin.external_baselines import (
    ExternalBaselineConfig,
    build_adapter,
    load_configs,
    load_dataset_registry,
    load_specs,
)


ROOT = Path(__file__).parents[1] / "aladdin" / "external_baselines"


class ExternalBaselineProtocolTests(unittest.TestCase):
    def test_all_declared_external_baselines_have_specs_and_dry_run_plans(self) -> None:
        specs = load_specs(ROOT / "baseline-specs.json")
        configs = load_configs(ROOT / "server-configs.local.json")
        self.assertEqual(set(specs), {"ditto", "dataenvgym", "dataevolver", "gepa"})
        self.assertEqual({config.baseline_key for config in configs}, set(specs))
        for config in configs:
            plan = build_adapter(specs[config.baseline_key]).dry_run(config)
            self.assertEqual(plan["execution_mode"], "validated-dry-run")
            self.assertFalse(plan["official_method_execution"])
            self.assertIn("artifact_provenance", plan["required_audit_fields"])

    def test_final_test_cannot_be_unlocked_in_preregistered_config(self) -> None:
        specs = load_specs(ROOT / "baseline-specs.json")
        config = load_configs(ROOT / "server-configs.local.json")[0]
        value = json.loads(json.dumps(config.__dict__))
        value["final_test_unlocked"] = True
        with self.assertRaises(ValueError):
            ExternalBaselineConfig.from_dict(value).validate(specs[config.baseline_key])

    def test_server_template_is_not_accepted_as_a_completed_run(self) -> None:
        specs = load_specs(ROOT / "baseline-specs.json")
        config = load_configs(ROOT / "server-configs.json")[0]
        with self.assertRaises(ValueError):
            config.validate(specs[config.baseline_key])

    def test_dataset_registry_has_pinned_or_explicitly_unpinned_sources(self) -> None:
        entries = load_dataset_registry(ROOT / "dataset-registry.json")
        by_key = {entry["dataset_key"]: entry for entry in entries}
        self.assertGreaterEqual(len(entries), 15)
        self.assertEqual(
            by_key["ultrafeedback"]["revision"],
            "40b436560ca83a8dba36114c22ab3c66e43f6d5e",
        )
        self.assertEqual(by_key["coauthor"]["revision"], None)
        self.assertEqual(by_key["ultrafeedback"]["status"], "not-human-feedback")
        self.assertTrue(by_key["prism"]["supports_user_identity"])


if __name__ == "__main__":
    unittest.main()
