from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from .technical_one_protocol import (
    COMPONENT_MATRIX_METHODS,
    audit_complete_technology_one,
)


class CompleteTechnologyOneAuditTests(unittest.TestCase):
    seeds = (17, 23)

    def _write(self, root: Path, name: str, seed: int, **updates: object) -> None:
        run_root = root / f"{name}-seed-{seed}"
        run_root.mkdir(parents=True)
        payload = {
            "status": "complete",
            "seed": seed,
            "final_test_used": False,
            "batch_count": 3,
            "requested_train_group_budget": 1000,
            "budget_limit": {"training_steps": 70000, "evaluator_calls": 1400},
        }
        payload.update(updates)
        (run_root / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        paths = {name: root / name for name in ("component", "independent", "dependency", "fixed", "reflexion")}
        for seed in self.seeds:
            self._write(paths["component"], "component", seed, methods=list(COMPONENT_MATRIX_METHODS))
            self._write(paths["independent"], "independent", seed, proposal_mode="independent")
            self._write(
                paths["dependency"], "dependency", seed,
                proposal_mode="dependency-aware-joint",
                transitions=[{"edits": [{"component": "generator"}, {"component": "selector"}]}],
            )
            self._write(paths["fixed"], "fixed", seed, method="fixed-harness-react", decision_llm_used=True)
            self._write(paths["reflexion"], "reflexion", seed, method="react-reflexion", decision_llm_used=True)
        return paths

    def _audit(self, paths: dict[str, Path]) -> dict[str, object]:
        return audit_complete_technology_one(
            component_root=paths["component"],
            independent_root=paths["independent"],
            dependency_root=paths["dependency"],
            fixed_root=paths["fixed"],
            reflexion_root=paths["reflexion"],
            expected_seeds=self.seeds,
        )

    def test_matching_complete_evidence_passes(self) -> None:
        with TemporaryDirectory() as directory:
            result = self._audit(self._fixture(Path(directory)))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["final_test"], "LOCKED")

    def test_final_test_use_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._write(
                paths["dependency"], "newer-dependency", 17,
                proposal_mode="dependency-aware-joint",
                final_test_used=True,
                transitions=[{"edits": [{"component": "generator"}, {"component": "selector"}]}],
            )
            with self.assertRaisesRegex(ValueError, "final_test_used"):
                self._audit(paths)

    def test_mode_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._write(paths["independent"], "newer-independent", 17, proposal_mode="dependency-aware")
            with self.assertRaisesRegex(ValueError, "proposal_mode"):
                self._audit(paths)

    def test_declared_budget_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._write(
                paths["dependency"], "newer-dependency", 17,
                proposal_mode="dependency-aware-joint",
                budget_limit={"training_steps": 1, "evaluator_calls": 1400},
                transitions=[{"edits": [{"component": "generator"}, {"component": "selector"}]}],
            )
            with self.assertRaisesRegex(ValueError, "budget mismatch"):
                self._audit(paths)

    def test_missing_decision_baseline_seed_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            (paths["fixed"] / "fixed-seed-23" / "summary.json").unlink()
            with self.assertRaisesRegex(ValueError, r"missing=\[23\]"):
                self._audit(paths)

    def test_joint_evidence_without_joint_proposal_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            self._write(
                paths["dependency"], "newer-dependency", 17,
                proposal_mode="dependency-aware-joint",
                transitions=[{"edits": [{"component": "generator"}]}],
            )
            with self.assertRaisesRegex(ValueError, "no joint proposal"):
                self._audit(paths)


if __name__ == "__main__":
    unittest.main()
