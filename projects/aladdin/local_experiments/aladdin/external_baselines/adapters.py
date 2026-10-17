from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from .protocol import ExternalBaselineConfig, ExternalBaselineSpec


class ExternalBaselineAdapter(ABC):
    """Validated server handoff, not an implementation of a paper method."""

    def __init__(self, spec: ExternalBaselineSpec):
        self.spec = spec

    @abstractmethod
    def method_settings(self) -> dict[str, Any]:
        raise NotImplementedError

    def dry_run(self, config: ExternalBaselineConfig) -> dict[str, Any]:
        config.validate(self.spec)
        return {
            "baseline": asdict(self.spec),
            "config": asdict(config),
            "execution_mode": "validated-dry-run",
            "official_method_execution": False,
            "actions": [
                "ingest_dataset_manifest",
                "select_seed_records",
                "run_method_specific_data_or_prompt_step",
                "train_or_adapt_task_model",
                "evaluate_development",
                "evaluate_held_out",
                "seal_final_test_until_freeze",
            ],
            "method_settings": self.method_settings(),
            "required_audit_fields": [
                "official_commit",
                "initial_checkpoint",
                "dataset_manifest_checksum",
                "budget_used",
                "artifact_provenance",
                "feedback_provenance",
            ],
        }


class DittoAdapter(ExternalBaselineAdapter):
    def method_settings(self) -> dict[str, Any]:
        return {
            "adaptation": "LoRA+DPO",
            "feedback_unit": "demonstration_preferred_over_model_output",
            "negative_sampling": "online_and_replay_checkpoints",
            "reference_policy": "pinned_initial_policy",
        }


class DataEnvGymAdapter(ExternalBaselineAdapter):
    def method_settings(self) -> dict[str, Any]:
        return {
            "teacher_student": True,
            "environments": ["open-ended", "skill-list", "skill-tree"],
            "state_variants": ["with-state", "no-state"],
            "feedback_unit": "student_evaluation_state",
        }


class DataEvolverAdapter(ExternalBaselineAdapter):
    def method_settings(self) -> dict[str, Any]:
        return {
            "adaptation": "SFT_on_prepared_data",
            "seed_examples": 20,
            "comparisons": ["vanilla-sft", "dataflow-sft", "dataevolver-sft"],
            "ablations": ["without-operator-level-evolution", "without-pipeline-level-evolution"],
        }


class GepaAdapter(ExternalBaselineAdapter):
    def method_settings(self) -> dict[str, Any]:
        return {
            "adaptation": "prompt-evolution",
            "comparisons": ["base-program", "miprov2", "grpo"],
            "primary_aladdin_role": "supplemental-prompt-level-control",
        }


def build_adapter(spec: ExternalBaselineSpec) -> ExternalBaselineAdapter:
    adapters = {
        "ditto": DittoAdapter,
        "dataenvgym": DataEnvGymAdapter,
        "dataevolver": DataEvolverAdapter,
        "gepa": GepaAdapter,
    }
    try:
        return adapters[spec.key](spec)
    except KeyError as exc:
        raise ValueError(f"No local adapter registered for {spec.key}") from exc
