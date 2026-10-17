from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from core_system.mle_core.models import TaskRecord, to_jsonable

from .calibration import ReliabilityCalibrator
from .evaluation import QualityEvaluatorState
from .system import (
    CandidateGenerator,
    FeedbackReviewer,
    QualityAuditor,
    QualityScorer,
    RequirementScorer,
    RequirementHypothesis,
    RequirementState,
    RoundResult,
    SynthesisBudgetAllocator,
    run_round,
)


def load_recursive_state(path: str | Path) -> dict[str, object]:
    """Load and validate the persisted state for a resumable controller."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"completed_rounds", "requirement_state", "allocator"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"recursive state missing keys: {sorted(missing)}")
    requirement = value["requirement_state"]
    hypotheses = requirement.get("hypotheses", [])
    state = RequirementState(
        [
            RequirementHypothesis(str(row["hypothesis_id"]), str(row["text"]))
            for row in hypotheses
        ],
        {str(key): float(item) for key, item in requirement["weights"].items()},
        [str(item) for item in requirement.get("confirmed_edits", [])],
    )
    calibrator = (
        ReliabilityCalibrator.from_dict(value["quality_calibration"])
        if value.get("quality_calibration")
        else None
    )
    allocator = SynthesisBudgetAllocator.from_dict(value["allocator"])
    evaluator_state = (
        QualityEvaluatorState(**value["evaluator_state"])
        if value.get("evaluator_state")
        else None
    )
    current_seed_records = [
        TaskRecord(
            record_id=str(row["record_id"]),
            task_id=str(row["task_id"]),
            split=str(row["split"]),
            input=row["input"],
            target=row.get("target"),
            source_group_id=row.get("source_group_id"),
            metadata=row.get("metadata", {}),
            parent_record_ids=tuple(row.get("parent_record_ids", [])),
        )
        for row in value.get("current_seed_records", [])
    ]
    return {
        "completed_rounds": int(value["completed_rounds"]),
        "requirement_state": state,
        "quality_calibrator": calibrator,
        "allocator": allocator,
        "evaluator_state": evaluator_state,
        "current_seed_records": current_seed_records,
    }


def _state_payload(
    *,
    round_index: int,
    requirements: RequirementState,
    calibrator: ReliabilityCalibrator | None,
    allocator: SynthesisBudgetAllocator,
    evaluator_state: QualityEvaluatorState | None,
    current_seed_records: Sequence[TaskRecord],
) -> dict[str, Any]:
    return {
        "completed_rounds": round_index + 1,
        "requirement_state": {
            "hypotheses": [asdict(item) for item in requirements.hypotheses],
            "weights": requirements.weights,
            "confirmed_edits": requirements.confirmed_edits,
        },
        "quality_calibration": calibrator.to_dict() if calibrator else None,
        "allocator": allocator.to_dict(),
        "evaluator_state": evaluator_state.to_dict() if evaluator_state else None,
        "current_seed_records": to_jsonable(current_seed_records),
    }


def run_recursive(
    *,
    run_id: str,
    rounds: int,
    seed_records: Sequence[TaskRecord],
    development_records: Sequence[TaskRecord],
    requirement_state: RequirementState,
    requirement_scorer: RequirementScorer,
    quality_scorer: QualityScorer,
    generator: CandidateGenerator,
    reviewer: FeedbackReviewer,
    allocator: SynthesisBudgetAllocator,
    output_dir: str | Path,
    quality_calibrator: ReliabilityCalibrator | None = None,
    quality_auditor: QualityAuditor | None = None,
    evaluator_state: QualityEvaluatorState | None = None,
    round_run_id_suffix: bool = True,
    start_round: int = 0,
    **round_kwargs: Any,
) -> tuple[RoundResult, ...]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if start_round < 0:
        raise ValueError("start_round must be non-negative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    current_seed_records = list(seed_records)
    for offset in range(rounds):
        round_index = start_round + offset
        result = run_round(
            run_id=(
                f"{run_id}-round-{round_index}"
                if round_run_id_suffix or round_index > 0
                else run_id
            ),
            round_index=round_index,
            seed_records=current_seed_records,
            development_records=development_records,
            requirement_state=requirement_state,
            requirement_scorer=requirement_scorer,
            quality_scorer=quality_scorer,
            generator=generator,
            reviewer=reviewer,
            allocator=allocator,
            output_dir=output,
            quality_calibrator=quality_calibrator,
            quality_auditor=quality_auditor,
            quality_evaluator_state=evaluator_state,
            **round_kwargs,
        )
        results.append(result)
        existing_ids = {item.record_id for item in current_seed_records}
        current_seed_records.extend(
            item
            for item in result.selected_records
            if item.record_id not in existing_ids
        )
        state_path = output / "recursive-state.json"
        state_path.write_text(
            json.dumps(
                _state_payload(
                    round_index=round_index,
                    requirements=requirement_state,
                    calibrator=quality_calibrator,
                    allocator=allocator,
                    evaluator_state=evaluator_state,
                    current_seed_records=current_seed_records,
                ),
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return tuple(results)


def write_recursive_manifest(output_dir: str | Path, results: Sequence[RoundResult]) -> None:
    path = Path(output_dir) / "recursive-manifest.json"
    existing = []
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        existing = value.get("rounds", [])
    by_run_id = {row["run_id"]: row for row in existing}
    by_run_id.update({result.run_id: asdict(result) for result in results})
    path.write_text(
        json.dumps(
            {
                "rounds": list(by_run_id.values()),
                "final_test_accessed": False,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
