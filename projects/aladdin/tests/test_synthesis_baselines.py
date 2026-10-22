import json

import pytest

from experiments.synthesis.generator import OpenAICompatibleGenerator
from experiments.synthesis.runner import build_parser, run_experiment
from experiments.synthesis.strategies import STRATEGIES, synthesize


class FakeGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, max_tokens, temperature):
        self.prompts.append((prompt, max_tokens, temperature))
        if "Return only JSON" in prompt:
            return json.dumps({"input": "new task", "output": "new answer"})
        if "Return only the task input" in prompt:
            return "new task"
        if "Return only the revised" in prompt:
            return "evolved task"
        return "new answer"


@pytest.fixture
def seeds():
    return [{"id": "s1", "prompt": "seed task", "response": "seed answer", "source_group_id": "g1"}]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_strategy_has_common_schema_and_lineage(seeds, strategy):
    result = synthesize(
        seeds,
        strategy=strategy,
        count=1,
        generator=FakeGenerator() if strategy != "seed-only" else None,
    )

    assert len(result) == 1
    assert result[0]["input"]
    assert result[0]["output"]
    assert result[0]["source_group_id"] == "g1"
    assert result[0]["synthesis_strategy"] == strategy
    if strategy == "seed-only":
        assert result[0]["record_id"] == "s1"
    else:
        assert result[0]["parent_record_id"] == "s1"


def test_generation_call_counts_are_explicit(seeds):
    fixed = synthesize(seeds, strategy="fixed-prompt", count=2, generator=FakeGenerator())
    self_instruct = synthesize(seeds, strategy="self-instruct", count=2, generator=FakeGenerator())
    evol = synthesize(seeds, strategy="evol-instruct", count=2, generator=FakeGenerator())

    assert sum(row["generation_calls"] for row in fixed) == 2
    assert sum(row["generation_calls"] for row in self_instruct) == 4
    assert sum(row["generation_calls"] for row in evol) == 4
    assert evol[0]["evolution_operation"] != evol[1]["evolution_operation"]


def test_context_is_preserved_and_given_to_all_adapters(seeds):
    seeds[0]["context"] = [{"role": "user", "content": "earlier request"}]
    for strategy in ("fixed-prompt", "self-instruct", "evol-instruct"):
        generator = FakeGenerator()
        rows = synthesize(seeds, strategy=strategy, count=1, generator=generator)
        assert rows[0]["context"] == seeds[0]["context"]
        assert any("earlier request" in prompt for prompt, _, _ in generator.prompts)


def test_bad_structured_generation_fails_closed(seeds):
    with pytest.raises(ValueError, match="JSON object"):
        synthesize(
            seeds,
            strategy="fixed-prompt",
            count=1,
            generator=lambda *args, **kwargs: "not json",
        )


def test_seed_only_needs_no_generator(seeds):
    rows = synthesize(seeds, strategy="seed-only", count=0)
    assert len(rows) == 1
    assert rows[0]["generation_calls"] == 0


def test_cli_writes_candidates_and_manifest(tmp_path):
    data = tmp_path / "seed.jsonl"
    output = tmp_path / "run" / "candidates.jsonl"
    data.write_text(json.dumps({"input": "seed", "output": "answer"}) + "\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["--data", str(data), "--output", str(output), "--strategy", "seed-only"]
    )

    manifest = run_experiment(args)

    assert manifest["status"] == "completed"
    assert manifest["candidate_count"] == 1
    assert manifest["generation_calls"] == 0
    assert output.exists()
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_endpoint_client_parses_chat_completions(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            ).encode()

    def fake_urlopen(request, **kwargs):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleGenerator("http://localhost/v1/chat/completions", "test")

    assert client.generate("prompt", max_tokens=8, temperature=0) == "ok"
    assert client.usage.requests == 1
    assert client.usage.reported is True
    assert client.usage.complete_usage_requests == 1
    assert client.usage.prompt_tokens == 3
    assert client.usage.completion_tokens == 2
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_manifest_records_complete_usage_request_count(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{
                        "message": {
                            "content": json.dumps({"input": "new task", "output": "new answer"})
                        }
                    }],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    data = tmp_path / "seed.jsonl"
    output = tmp_path / "fixed.jsonl"
    data.write_text(json.dumps({"input": "seed", "output": "answer"}) + "\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--data", str(data),
            "--output", str(output),
            "--strategy", "fixed-prompt",
            "--count", "1",
            "--endpoint", "http://localhost/v1/chat/completions",
            "--model", "test-model",
        ]
    )

    manifest = run_experiment(args)

    assert manifest["generation_usage"]["requests"] == 1
    assert manifest["generation_usage"]["reported_requests"] == 1
    assert manifest["generation_usage"]["complete_usage_requests"] == 1


def test_evol_runner_requires_task_compatibility_flag(tmp_path):
    data = tmp_path / "seed.jsonl"
    data.write_text(json.dumps({"input": "seed", "output": "answer"}) + "\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--data",
            str(data),
            "--output",
            str(tmp_path / "evol.jsonl"),
            "--strategy",
            "evol-instruct",
            "--endpoint",
            "http://localhost/v1/chat/completions",
            "--model",
            "test",
        ]
    )

    with pytest.raises(ValueError, match="task compatibility"):
        run_experiment(args)
