import json
from argparse import Namespace

import pytest

from experiments.evaluation.qwen_sft import (
    build_synthesis_training_records,
    build_training_schedule,
    format_prompt,
    pairwise_preference_metrics,
)
from experiments import experiment_one_synthesis
from experiments.experiment_one_synthesis import _load_development, select_seed_groups


def test_seed_group_selection_is_deterministic_and_disjoint():
    rows = [
        {"record_id": "a", "input": "a", "output": "1", "source_group_id": "g1"},
        {"record_id": "b", "input": "b", "output": "2", "source_group_id": "g2"},
        {"record_id": "c", "input": "c", "output": "3", "source_group_id": "g3"},
    ]
    left = select_seed_groups(rows, 2, 17)
    right = select_seed_groups(rows, 2, 17)
    assert left == right
    assert len({row["source_group_id"] for row in left}) == 2


def test_training_schedule_equalizes_count_without_mutating_records():
    rows = [{"record_id": "a"}, {"record_id": "b"}]
    selected = build_training_schedule(rows, 5, seed=17)
    assert len(selected) == 5
    assert rows == [{"record_id": "a"}, {"record_id": "b"}]


def test_synthesis_training_records_keep_fixed_seed_and_synthetic_slots():
    seeds = [{"record_id": "s1"}, {"record_id": "s2"}]
    candidates = [{"record_id": "c1"}, {"record_id": "c2"}]
    rows = build_synthesis_training_records(
        seeds,
        candidates,
        strategy="fixed-prompt",
        train_records=6,
        synthetic_train_records=2,
        seed=17,
    )
    assert len(rows) == 6
    assert sum(row["record_id"].startswith("c") for row in rows) == 2
    assert sum(row["record_id"].startswith("s") for row in rows) == 4


def test_group_balanced_schedule_and_pairwise_metrics():
    rows = [
        {"record_id": "a1", "source_group_id": "a"},
        {"record_id": "a2", "source_group_id": "a"},
        {"record_id": "b1", "source_group_id": "b"},
    ]
    schedule = build_training_schedule(rows, 4, seed=17)
    assert sum(row["source_group_id"] == "a" for row in schedule) == 2
    assert sum(row["source_group_id"] == "b" for row in schedule) == 2
    scored = [
        ({"source_group_id": "u", "conversation_id": "c", "interaction_id": "i", "if_chosen": True}, -1.0),
        ({"source_group_id": "u", "conversation_id": "c", "interaction_id": "i", "if_chosen": False}, -2.0),
        ({"source_group_id": "u", "conversation_id": "c", "interaction_id": "j", "if_chosen": True}, -1.0),
        ({"source_group_id": "u", "conversation_id": "c", "interaction_id": "j", "if_chosen": False}, -1.0),
    ]
    metrics = pairwise_preference_metrics(scored)
    assert metrics == {"pairwise_comparisons": 2, "pairwise_accuracy": 0.75}


def test_prompt_includes_context_and_input():
    prompt = format_prompt({"context": [{"role": "user", "content": "Earlier"}], "input": "Now"})
    assert "Earlier" in prompt
    assert "User: Now" in prompt
    assert prompt.endswith("Assistant:")


def test_qwen_chat_template_disables_thinking():
    class FakeTokenizer:
        chat_template = "qwen3-template"

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "formatted prompt"

    tokenizer = FakeTokenizer()
    prompt = format_prompt({"input": "Now"}, tokenizer)

    assert prompt == "formatted prompt"
    assert tokenizer.kwargs["enable_thinking"] is False


def test_development_loader_rejects_final_test(tmp_path):
    path = tmp_path / "final_test.jsonl"
    path.write_text(json.dumps({"record_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="final_test"):
        _load_development(path, None)


def test_development_loader_attaches_labels(tmp_path):
    data = tmp_path / "development.jsonl"
    labels = tmp_path / "development-labels.jsonl"
    data.write_text(
        json.dumps({"record_id": "x", "input": "q", "output": "a", "split": "development"}) + "\n",
        encoding="utf-8",
    )
    labels.write_text(json.dumps({"record_id": "x", "if_chosen": True, "score": 90}) + "\n", encoding="utf-8")
    rows = _load_development(data, labels)
    assert rows[0]["if_chosen"] is True
    assert rows[0]["score"] == 90


def test_generation_stops_after_shared_token_budget_is_reached(tmp_path, monkeypatch):
    class FakeGenerator:
        def __init__(self, *args, **kwargs):
            self.usage = Namespace(
                requests=0,
                reported_requests=0,
                complete_usage_requests=0,
                reported=False,
                prompt_tokens=0,
                completion_tokens=0,
            )

        def generate(self, prompt, *, max_tokens, temperature):
            self.usage.requests += 1
            self.usage.reported_requests += 1
            self.usage.complete_usage_requests += 1
            self.usage.reported = True
            self.usage.prompt_tokens += 3
            self.usage.completion_tokens += 2
            return json.dumps({"input": f"new {self.usage.requests}", "output": "response"})

    monkeypatch.setattr(experiment_one_synthesis, "OpenAICompatibleGenerator", FakeGenerator)
    args = Namespace(
        endpoint="http://localhost/v1/chat/completions",
        model="test-model",
        api_key_env="OPENAI_API_KEY",
        timeout_seconds=10,
        generation_token_budget=5,
        max_candidates_per_strategy=10,
        max_tokens=20,
        temperature=0.0,
    )
    seeds = [{"record_id": "s", "input": "seed", "output": "answer", "source_group_id": "g"}]
    rows, manifest = experiment_one_synthesis._run_synthesis(seeds, "fixed-prompt", args, tmp_path / "rows.jsonl", 17)
    assert len(rows) == 1
    assert manifest["generation_tokens_used"] == 5
    assert manifest["generation_budget_overshoot_tokens"] == 0


def test_runner_uses_seed_and_development_only_and_writes_summary(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.jsonl"
    dev_path = tmp_path / "development.jsonl"
    labels_path = tmp_path / "development-labels.jsonl"
    seed_path.write_text(
        json.dumps({"record_id": "s", "input": "seed", "output": "answer", "source_group_id": "g", "split": "seed"}) + "\n",
        encoding="utf-8",
    )
    dev_path.write_text(
        json.dumps({"record_id": "d", "input": "dev", "output": "target", "source_group_id": "d-group", "split": "development"}) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(json.dumps({"record_id": "d", "if_chosen": True}) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "prepared",
                "dataset": "HannahRoseKirk/prism-alignment",
                "revision": "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6",
                "split_seed": 17,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiment_one_synthesis,
        "train_and_evaluate",
        lambda train, dev, config: {
            "status": "completed",
            "training_examples": len(train),
            "pairwise_accuracy": 1.0,
            "pairwise_comparisons": 1,
            "mean_target_nll": 1.0,
            "training_seconds": 0.1,
        },
    )
    monkeypatch.setattr(
        experiment_one_synthesis,
        "_run_synthesis",
        lambda seeds, strategy, args, output, run_seed: (
            experiment_one_synthesis.synthesize(seeds, strategy="seed-only", count=0),
            {"candidate_count": len(seeds), "generation_tokens_used": 0},
        ) if strategy == "seed-only" else (
            [{**seeds[0], "record_id": f"c-{run_seed}", "synthesis_strategy": strategy}],
            {"candidate_count": 1, "generation_tokens_used": 5},
        ),
    )
    args = experiment_one_synthesis.build_parser().parse_args(
        [
            "--seed-data", str(seed_path),
            "--development-data", str(dev_path),
            "--development-labels", str(labels_path),
            "--output", str(tmp_path / "out"),
            "--base-model", experiment_one_synthesis.MODEL_ID,
            "--strategies", "seed-only", "fixed-prompt",
            "--seeds", "17", "23",
            "--seed-group-limit", "1",
            "--train-records", "2",
            "--synthetic-train-records", "1",
            "--generation-token-budget", "10",
            "--development-group-limit", "1",
        ]
    )
    result = experiment_one_synthesis.run(args)
    assert result["data"]["final_test_accessed"] is False
    assert result["feedback_boundary"]["new_human_feedback_collected"] is False
    assert result["feedback_boundary"]["human_feedback_used_for_generation_or_selection"] is False
    assert result["feedback_boundary"]["development_labels_role"] == "offline_outcome_evaluation_only"
    assert len(result["results"]) == 4
    assert all(row["training"]["training_examples"] == 2 for row in result["results"])
    assert (tmp_path / "out" / "result.json").is_file()


def test_development_sampling_keeps_whole_source_groups():
    rows = [
        {"source_group_id": "a", "record_id": "a1"},
        {"source_group_id": "a", "record_id": "a2"},
        {"source_group_id": "b", "record_id": "b1"},
    ]
    sampled = experiment_one_synthesis.select_development_groups(rows, 1, 101)
    assert len({row["source_group_id"] for row in sampled}) == 1
    assert len(sampled) in (1, 2)


def test_formal_model_id_and_revision_are_frozen():
    args = experiment_one_synthesis.build_parser().parse_args(
        [
            "--seed-data", "seed.jsonl",
            "--development-data", "development.jsonl",
            "--development-labels", "labels.jsonl",
            "--output", "run",
            "--base-model", experiment_one_synthesis.MODEL_ID,
        ]
    )
    experiment_one_synthesis.validate_args(args)
    assert args.model == experiment_one_synthesis.MODEL_ID
    assert args.model_revision == experiment_one_synthesis.MODEL_REVISION

    args.model = "some-other-model"
    with pytest.raises(ValueError, match="generation model is frozen"):
        experiment_one_synthesis.validate_args(args)

    args.model = experiment_one_synthesis.MODEL_ID
    args.base_model = "some-other-model"
    with pytest.raises(ValueError, match="frozen Hugging Face ID"):
        experiment_one_synthesis.validate_args(args)

    args.base_model = f"/models/snapshots/{experiment_one_synthesis.MODEL_REVISION}"
    experiment_one_synthesis.validate_args(args)
