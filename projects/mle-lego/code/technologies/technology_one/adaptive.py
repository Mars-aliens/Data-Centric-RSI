from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core_system.mle_core.models import Budget, TaskRecord, to_jsonable
from core_system.mle_core.sklearn_components import (
    TfidfLogisticEvaluator,
    TfidfLogisticTrainer,
)
from core_system.mle_core.store import ArtifactStore

from .data import StreamBatch, build_lego_scenario_registry, lego_stream_batches, record_to_dict
from .diagnostics import FailureDiagnostic, diagnose_batch
from .experience import Experience, ExperienceStore, Skill
from .harness import (
    HarnessEdit,
    HarnessManifest,
    HarnessVersionChain,
    apply_edits,
    dependency_closure,
    initial_harness,
    random_joint_edit,
)


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    harness_version: str
    source_count: int
    selected_count: int
    selected_record_ids: tuple[str, ...]
    held_in: dict[str, float]
    held_out: dict[str, float]
    diagnostic: FailureDiagnostic | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "harness_version": self.harness_version,
            "source_count": self.source_count,
            "selected_count": self.selected_count,
            "selected_record_ids": list(self.selected_record_ids),
            "held_in": self.held_in,
            "held_out": self.held_out,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
        }


@dataclass
class TrialResult:
    strategy: str
    initial_harness_version: str
    final_harness_version: str
    batches: list[BatchResult] = field(default_factory=list)
    experiences: ExperienceStore = field(default_factory=ExperienceStore)
    accepted_versions: list[str] = field(default_factory=list)
    rejected_versions: list[str] = field(default_factory=list)
    rollbacks: int = 0
    budget_limit: Budget = field(
        default_factory=lambda: Budget(training_steps=8, evaluator_calls=16)
    )
    used_budget: Budget = field(default_factory=Budget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "initial_harness_version": self.initial_harness_version,
            "final_harness_version": self.final_harness_version,
            "batches": [batch.to_dict() for batch in self.batches],
            "experiences": self.experiences.to_dict(),
            "accepted_versions": self.accepted_versions,
            "rejected_versions": self.rejected_versions,
            "rollbacks": self.rollbacks,
            "budget_limit": to_jsonable(self.budget_limit),
            "used_budget": to_jsonable(self.used_budget),
        }


def _records_for_training(seed_records: list[TaskRecord], selected: list[TaskRecord]) -> list[TaskRecord]:
    return seed_records + selected


class HarnessExecutor:
    """Small local executor using shared-core trainers, evaluators and artifacts."""

    def __init__(self, root: Path):
        self.root = root
        self.registry = build_lego_scenario_registry()
        self.trainer = TfidfLogisticTrainer()
        self.evaluator = TfidfLogisticEvaluator()
        self.budget = Budget()

    def _select(self, manifest: HarnessManifest, batch: StreamBatch) -> list[TaskRecord]:
        generator = manifest.components["generator"].config
        selector = manifest.components["selector"].config
        filter_config = manifest.components["filter"].config
        routes = set(generator.get("routes", []))
        minimum = float(selector.get("min_quality", 0.0))
        return [
            record
            for record in batch.records
            if record.metadata.get("route") in routes
            and float(record.metadata.get("quality", 0.0)) >= minimum
            and (
                not filter_config.get("require_accepted", False)
                or bool(record.metadata.get("accepted", False))
            )
        ]

    def execute_batch(self, manifest: HarnessManifest, batch: StreamBatch) -> BatchResult:
        store = ArtifactStore(self.root / manifest.version / batch.batch_id)
        harness_artifact = store.put_json(
            kind="harness_manifest",
            value=manifest.to_dict(),
            version=manifest.version,
        )
        selected = self._select(manifest, batch)
        seed_records = self.registry.records_for("lego-local-intent", split="seed")
        dataset = _records_for_training(seed_records, selected)
        dataset_artifact = store.put_json(
            kind="dataset",
            value=[record_to_dict(record) for record in dataset],
            version=manifest.version,
        )
        checkpoint, train_cost = self.trainer.train(
            train_records=dataset,
            initial_checkpoint=None,
            config={},
            seed=17,
            store=store,
        )
        checkpoint_artifact = store.put_json(
            kind="checkpoint",
            value=checkpoint,
            version=f"{self.trainer.name}-{self.trainer.version}",
            parents=[dataset_artifact.artifact_id],
        )
        held_in, eval_cost = self.evaluator.evaluate(
            checkpoint=checkpoint,
            records=self.registry.records_for("lego-local-intent", split="held_in"),
            config={},
        )
        held_out, held_out_cost = self.evaluator.evaluate(
            checkpoint=checkpoint,
            records=self.registry.records_for("lego-local-intent", split="held_out"),
            config={},
        )
        store.append_event(
            "lego_batch_completed",
            {
                "batch_id": batch.batch_id,
                "harness_version": manifest.version,
                "harness_artifact_id": harness_artifact.artifact_id,
                "dataset_artifact_id": dataset_artifact.artifact_id,
                "checkpoint_artifact_id": checkpoint_artifact.artifact_id,
                "selected_record_ids": [record.record_id for record in selected],
                "held_in": held_in,
                "held_out": held_out,
            },
        )
        self.budget = self.budget.add(train_cost).add(eval_cost).add(held_out_cost)
        diagnostic = diagnose_batch(
            batch_id=batch.batch_id,
            manifest=manifest,
            source_records=[record_to_dict(record) for record in batch.records],
            selected_records=[record_to_dict(record) for record in selected],
        )
        return BatchResult(
            batch_id=batch.batch_id,
            harness_version=manifest.version,
            source_count=len(batch.records),
            selected_count=len(selected),
            selected_record_ids=tuple(record.record_id for record in selected),
            held_in=held_in,
            held_out=held_out,
            diagnostic=diagnostic,
        )


def _objective(result: BatchResult) -> tuple[float, float, float]:
    return (
        float(result.held_out.get("accuracy", 0.0)),
        float(result.held_in.get("accuracy", 0.0)),
        float(result.selected_count),
    )


def _validate_candidate(
    executor: HarnessExecutor,
    current: HarnessManifest,
    candidate: HarnessManifest,
    batch: StreamBatch,
) -> tuple[bool, dict[str, Any]]:
    current_result = executor.execute_batch(current, batch)
    candidate_result = executor.execute_batch(candidate, batch)
    validation = {
        "batch_id": batch.batch_id,
        "current": current_result.to_dict(),
        "candidate": candidate_result.to_dict(),
        "held_in_unchanged_or_better": _objective(candidate_result)[1] >= _objective(current_result)[1],
        "held_out_regression_report_only": {
            "current_accuracy": current_result.held_out.get("accuracy", 0.0),
            "candidate_accuracy": candidate_result.held_out.get("accuracy", 0.0),
        },
        "selected_count_change": candidate_result.selected_count - current_result.selected_count,
    }
    accepted = (
        validation["held_in_unchanged_or_better"]
        and candidate_result.selected_count >= current_result.selected_count
        and candidate_result.selected_count > 0
    )
    validation["accepted"] = accepted
    return accepted, validation


def _reflection(diagnostic: FailureDiagnostic) -> str:
    return (
        f"Batch {diagnostic.batch_id} selected no records. "
        f"Observed missing routes={diagnostic.evidence.get('missing_routes', [])}; "
        "check generator and selector compatibility before changing the Harness."
    )


def run_strategy(
    strategy: str,
    output_dir: Path,
    *,
    seed: int = 17,
) -> TrialResult:
    if strategy not in {"fixed", "reflexion", "independent-edit", "random-joint-edit", "dependency-aware"}:
        raise ValueError(f"Unknown LEGO strategy: {strategy}")
    root = output_dir / strategy
    root.mkdir(parents=True, exist_ok=True)
    executor = HarnessExecutor(root)
    chain = HarnessVersionChain(initial_harness())
    harness = chain.current
    result = TrialResult(
        strategy=strategy,
        initial_harness_version=harness.version,
        final_harness_version=harness.version,
    )
    batches = lego_stream_batches()
    for index, batch in enumerate(batches):
        batch_result = executor.execute_batch(harness, batch)
        result.batches.append(batch_result)
        diagnostic = batch_result.diagnostic
        if diagnostic is None or index == len(batches) - 1:
            continue
        reflection = _reflection(diagnostic)
        edits = diagnostic.candidate_edits
        selected_edits: tuple[HarnessEdit, ...] = ()
        if strategy == "reflexion":
            selected_edits = ()
        elif strategy == "independent-edit":
            selected_edits = (edits[next(iter(edits))],)
        elif strategy == "random-joint-edit":
            selected_edits = random_joint_edit(edits, seed=seed + index, size=min(2, len(edits)))
        elif strategy == "dependency-aware":
            names = dependency_closure(harness, diagnostic.suspected_components)
            selected_edits = tuple(edits[name] for name in names if name in edits)
        if selected_edits:
            candidate = apply_edits(harness, selected_edits)
            accepted, validation = _validate_candidate(executor, harness, candidate, batch)
            experience = Experience(
                experience_id=f"{strategy}-{batch.batch_id}",
                problem=diagnostic.problem,
                conditions={"harness_version": harness.version, "batch_id": batch.batch_id},
                reflection=reflection,
                edits=tuple(
                    {"component": edit.component, "patch": edit.patch}
                    for edit in selected_edits
                ),
                validation=validation,
                status="accepted" if accepted else "rejected",
            )
            result.experiences.add(experience)
            if accepted:
                chain.accept(candidate)
                harness = chain.current
                result.accepted_versions.append(candidate.version)
                result.experiences.add_skill(
                    Skill(
                        skill_id=f"skill-{candidate.version}",
                        name="recover-empty-selection",
                        trigger="a stream batch selects no records",
                        procedure=("diagnose routes and threshold", "validate on held-in and held-out", "adopt only if non-regressive"),
                        source_experience_ids=(experience.experience_id,),
                    )
                )
            else:
                result.rejected_versions.append(candidate.version)
                result.rollbacks += 1
        else:
            result.experiences.add(
                Experience(
                    experience_id=f"{strategy}-{batch.batch_id}",
                    problem=diagnostic.problem,
                    conditions={"harness_version": harness.version, "batch_id": batch.batch_id},
                    reflection=reflection,
                    edits=(),
                    validation={"accepted": False, "reason": "text-only reflection"},
                    status="reflected",
                )
            )
    result.final_harness_version = harness.version
    result.used_budget = executor.budget
    if result.used_budget.exceeds(result.budget_limit):
        raise RuntimeError(
            f"Strategy {strategy} exceeded its shared budget limit: "
            f"{result.used_budget} > {result.budget_limit}"
        )
    (root / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def run_all_strategies(output_dir: Path, *, seed: int = 17) -> dict[str, TrialResult]:
    strategies = ("fixed", "reflexion", "independent-edit", "random-joint-edit", "dependency-aware")
    return {name: run_strategy(name, output_dir, seed=seed) for name in strategies}

