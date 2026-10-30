import json
from types import SimpleNamespace

from experiments.rsi.runner import MODEL_REVISION, PRISM_REPOSITORY, PRISM_REVISION, run_feedback_rsi


class FakeGenerator:
    def __init__(self):
        self.calls = 0
        self.usage = SimpleNamespace(
            requests=0,
            complete_usage_requests=0,
            reported=True,
            prompt_tokens=0,
            completion_tokens=0,
        )

    def generate(self, prompt, *, max_tokens, temperature):
        self.calls += 1
        self.usage.requests += 1
        self.usage.complete_usage_requests += 1
        self.usage.prompt_tokens += 5
        self.usage.completion_tokens += 5
        if "propose one new task input" in prompt:
            return f"A generated task {self.calls}"
        return f"A generated response {self.calls}"


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_inputs(tmp_path):
    processed = tmp_path / "processed"
    seed_path = processed / "seed.jsonl"
    development_path = processed / "development.jsonl"
    labels_path = processed / "labels" / "development-labels.jsonl"
    seeds = [
        {
            "record_id": f"s{i}", "input": f"Seed task {i}", "output": f"Seed response {i}",
            "source_group_id": f"seed-user-{i}", "split": "seed", "context": [],
        }
        for i in range(4)
    ]
    development = []
    labels = []
    for user in range(20):
        for chosen in (False, True):
            record_id = f"d{user}-{int(chosen)}"
            development.append(
                {
                    "record_id": record_id, "input": f"Development question {user}",
                    "output": f"Response {user} {chosen}", "source_group_id": f"dev-user-{user}",
                    "interaction_id": f"interaction-{user}", "conversation_id": f"conversation-{user}",
                    "split": "development", "context": [],
                }
            )
            labels.append({"record_id": record_id, "if_chosen": chosen})
    _write_jsonl(seed_path, seeds)
    _write_jsonl(development_path, development)
    _write_jsonl(labels_path, labels)
    (processed / "manifest.json").write_text(
        json.dumps({"status": "prepared", "dataset": PRISM_REPOSITORY, "revision": PRISM_REVISION}),
        encoding="utf-8",
    )

    config = {
        "comparison": "overall", "seed_data": str(seed_path),
        "development_data": str(development_path), "development_labels": str(labels_path),
        "output": str(tmp_path / "runs" / "pilot"), "endpoint": "http://localhost/v1/chat/completions",
        "model_revision": MODEL_REVISION, "rounds": 2, "seeds": [17], "seed_group_limit": 2,
        "candidate_count_per_arm_round": 2, "query_count_per_arm_round": 1,
        "feedback_budget_seconds_per_arm_round": 30, "estimated_review_seconds": 5,
        "calibration_fraction": 0.5, "generation_token_budget_per_arm_round": 1000,
        "max_tokens": 32, "train_records": 4, "train_token_budget": 100,
        "max_steps": 1, "allow_cpu_training": True, "save_adapters": False,
    }
    return config


def _collector(queries, *, annotator_id, budget_seconds_per_arm, output_path):
    events = []
    for query in queries:
        events.append(
            {
                "feedback_source": "live_human", "query_id": query["query_id"],
                "record_id": query["record_id"], "arm_id": query["arm_id"],
                "source_group_id": query["source_group_id"],
                "round": query["round"],
                "selection_method": query["selection_method"],
                "annotator_id": annotator_id,
                "judgment": "usable", "issue_code": "none",
                "requirement_clarity": "clear", "actionable": True,
                "new_requirement": "Keep generated examples concise.", "duration_seconds": 2.0,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return events


def _trainer(calls):
    def trainer(train_rows, eval_rows, config):
        calls.append((len(train_rows), len(eval_rows), config.seed))
        return {
            "pairwise_comparisons": 2,
            "pairwise_accuracy": 0.5 + 0.01 * len(calls),
            "training_seconds": 0.1,
        }

    return trainer


def test_feedback_rsi_runs_all_phases_without_opening_final_test(tmp_path):
    config = _make_inputs(tmp_path)
    calls = []
    result = run_feedback_rsi(
        config,
        generator_factory=lambda _config: FakeGenerator(),
        trainer=_trainer(calls),
        feedback_collector=_collector,
        annotator_id="test-reviewer",
    )

    assert result["status"] == "completed"
    assert result["data"]["final_test_accessed"] is False
    assert result["feedback_boundary"]["phase_one_human_feedback"] is False
    assert result["feedback_boundary"]["phase_two_human_feedback"] is False
    assert len(result["arms"]["recursive_policy"]["rounds"]) == 2
    assert result["arms"]["fixed_policy"]["rounds"][0]["policy_update"]["status"] == "no_op"
    assert result["arms"]["recursive_policy"]["rounds"][0]["policy_update"]["status"] == "updated"
    assert result["arms"]["recursive_policy"]["rounds"][1]["training_feedback_examples_cumulative"] >= 2
    assert len(calls) == 4
    assert (tmp_path / "runs" / "pilot" / "result.json").is_file()


def test_interrupted_feedback_rsi_resumes_from_round_checkpoints(tmp_path):
    config = _make_inputs(tmp_path)
    trainer_calls = []
    generator_factories = []
    collector_calls = []

    def fail_once_trainer(train_rows, eval_rows, training_config):
        trainer_calls.append((len(train_rows), len(eval_rows), training_config.seed))
        if len(trainer_calls) == 3:
            raise RuntimeError("simulated interruption after feedback collection")
        return {
            "pairwise_comparisons": 2,
            "pairwise_accuracy": 0.5,
            "training_seconds": 0.1,
        }

    def counted_collector(*args, **kwargs):
        collector_calls.append(1)
        return _collector(*args, **kwargs)

    def generator_factory(_config):
        generator_factories.append(1)
        return FakeGenerator()

    import pytest

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_feedback_rsi(
            config,
            generator_factory=generator_factory,
            trainer=fail_once_trainer,
            feedback_collector=counted_collector,
            annotator_id="test-reviewer",
        )

    result = run_feedback_rsi(
        config,
        resume=True,
        generator_factory=generator_factory,
        trainer=_trainer(trainer_calls),
        feedback_collector=counted_collector,
        annotator_id="test-reviewer",
    )

    assert result["status"] == "completed"
    assert len(result["arms"]["recursive_policy"]["rounds"]) == 2
    assert len(trainer_calls) == 5
    assert len(generator_factories) == 3
    assert len(collector_calls) == 2


def test_all_arms_share_round_zero_candidate_pool(tmp_path):
    config = _make_inputs(tmp_path)
    config.update({"comparison": "all", "rounds": 1})
    generator_factories = []

    def generator_factory(_config):
        generator_factories.append(1)
        return FakeGenerator()

    result = run_feedback_rsi(
        config,
        generator_factory=generator_factory,
        trainer=lambda _train, _eval, _config: {
            "pairwise_comparisons": 2, "pairwise_accuracy": 0.5,
        },
        feedback_collector=_collector,
        annotator_id="test-reviewer",
    )
    run_root = tmp_path / "runs" / "pilot" / "seed-17"
    candidate_sets = []
    for arm in result["arms"]:
        rows = [
            json.loads(line)
            for line in (run_root / arm / "round-0" / "generated.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        candidate_sets.append(
            [(row["input"], row["output"], row["source_group_id"]) for row in rows]
        )
    shared_usage = json.loads(
        (run_root / "recursive_policy" / "round-0" / "generation.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "completed"
    assert candidate_sets[0] == candidate_sets[1] == candidate_sets[2]
    assert shared_usage["requests"] == 0
    assert shared_usage["generation_cost_owner"] == "fixed_policy"
    assert len(generator_factories) == 1


def test_model_feedback_runs_each_arm_with_equal_query_count(tmp_path, monkeypatch):
    config = _make_inputs(tmp_path)
    config.update({"feedback_mode": "model", "rounds": 2})

    class ModelJudge(FakeGenerator):
        def generate(self, prompt, *, max_tokens, temperature):
            self.calls += 1
            self.usage.requests += 1
            self.usage.complete_usage_requests += 1
            self.usage.prompt_tokens += 10
            self.usage.completion_tokens += 10
            return json.dumps({"judgment": "usable", "issue_code": "none",
                               "requirement_clarity": "clear", "corrected_output": None,
                               "new_requirement": "Keep responses concise."})

    monkeypatch.setattr("experiments.rsi.runner.OpenAICompatibleGenerator", lambda *args, **kwargs: ModelJudge())
    result = run_feedback_rsi(config, generator_factory=lambda _config: FakeGenerator(), trainer=_trainer([]))
    assert result["status"] == "completed"
    assert result["feedback_boundary"]["phase_three_model_feedback"] is True
    assert result["feedback_boundary"]["phase_three_live_feedback"] is False
    for arm in result["arms"].values():
        assert [row["model_feedback_requests"] for row in arm["rounds"]] == [1, 1]
        assert all(row["feedback_seconds"] is None for row in arm["rounds"])
    assert result["arms"]["fixed_policy"]["rounds"][0]["policy_update"]["status"] == "no_op"
    assert result["arms"]["recursive_policy"]["rounds"][0]["policy_update"]["status"] == "updated"
