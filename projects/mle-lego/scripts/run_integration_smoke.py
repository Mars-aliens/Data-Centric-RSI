from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core_system.mle_core.evolution import CandidateVersion, DataSystemVersion
from core_system.mle_core.store import ArtifactStore
from integration.experiment_matrix import all_feature_configs, feature_config_from_name
from integration.orchestrator import (
    BatchResult,
    IntegrationOrchestrator,
    IntegrationRunConfig,
    write_result,
)
from integration.runtime import DataSystemEvolutionRuntime
from technologies.technology_one.candidate_gate import SafetyChecklist
from technologies.technology_three.version_selection import (
    PairedMeanRetentionPolicy,
    WindowedVersionSelector,
)


def _version(version_id: str, parent: str | None = None) -> DataSystemVersion:
    return DataSystemVersion(
        version_id, f"data-{version_id}", f"model-{version_id}",
        f"harness-{version_id}", f"memory-{version_id}",
        f"skill-{version_id}", "evaluator-v1", parent_version_id=parent,
    )


def _executor(fault_batch: int | None):
    def execute(batch_index: int, active_version: str, features) -> BatchResult:
        failed = features.candidate_isolation and batch_index == fault_batch
        candidate_score = None if failed else 0.64 + 0.01 * batch_index
        stable_score = 0.60
        active_score = 0.60 if active_version == "stable-v1" else (0.42 if batch_index == 3 else 0.63)
        return BatchResult(
            batch_index=batch_index,
            stable_score=stable_score,
            candidate_score=candidate_score,
            candidate_failed=failed,
            active_score=active_score,
            feedback_version_id=active_version,
            metadata={"fault_injected": failed},
        )
    return execute


def run_one(name: str, output: Path, batches: int, fault_batch: int | None):
    features = feature_config_from_name(name)
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(directory)
        stable = _version("stable-v1")
        runtime = DataSystemEvolutionRuntime(
            stable_version=stable,
            store=store,
            selector=WindowedVersionSelector(
                store,
                policy=PairedMeanRetentionPolicy(min_batches=3, min_mean_gain=0.01, max_regression=0.05),
            ),
        )
        candidate = CandidateVersion("candidate-1", _version("candidate-v2", "stable-v1"), "smoke", "batch-0")
        orchestrator = IntegrationOrchestrator(
            runtime=runtime,
            features=features,
            config=IntegrationRunConfig(name, batches),
            candidate=candidate,
            checklist=SafetyChecklist(True, True, True, True, True, True),
        )
        result = orchestrator.run(_executor(fault_batch))
    write_result(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic cross-technology integration smoke.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", choices=[item.name for item in all_feature_configs()])
    parser.add_argument("--all-configs", action="store_true")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--fault-batch", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir)
    names = [item.name for item in all_feature_configs()] if args.all_configs else [args.config or all_feature_configs()[-1].name]
    for name in names:
        result = run_one(name, output / name / "summary.json", args.batches, args.fault_batch)
        print(f"{name}: {result.status} {result.completed_batches}/{args.batches} active={result.active_version_id}")


if __name__ == "__main__":
    main()
