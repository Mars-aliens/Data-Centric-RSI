import json

import pytest

from experiments.evaluation.blind_pairwise import (
    analyze_blind_pairwise_feedback,
    collect_blind_pairwise_feedback,
    prepare_blind_pairwise_queue,
    select_final_test_prompts,
)


def test_final_prompt_sampling_uses_one_prompt_per_interaction_and_no_target_output():
    rows = [
        {
            "record_id": f"r{index}", "source_group_id": "user-1", "conversation_id": "c1",
            "interaction_id": "i1", "input": "Question", "context": [],
            "output": f"secret target {index}", "split": "final_test",
        }
        for index in range(2)
    ]
    selected = select_final_test_prompts(rows, max_items=1, seed=17)
    assert len(selected) == 1
    assert "output" not in selected[0]
    assert selected[0]["split"] == "final_test"


def test_blind_queue_hides_system_names_and_key_restores_mapping(tmp_path):
    final_test = tmp_path / "final_test.jsonl"
    rows = [
        {
            "record_id": f"r{i}", "source_group_id": f"u{i}", "conversation_id": f"c{i}",
            "interaction_id": f"i{i}", "input": f"Question {i}", "context": [],
            "output": f"hidden target {i}", "split": "final_test",
        }
        for i in range(2)
    ]
    final_test.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def generator(prompts, *, adapter_path, **_kwargs):
        prefix = "fixed" if str(adapter_path) == "adapter-fixed" else "recursive"
        return {row["record_id"]: f"{prefix} answer" for row in prompts}

    output = tmp_path / "blind"
    manifest = prepare_blind_pairwise_queue(
        final_test_path=final_test,
        adapter_a="adapter-fixed",
        adapter_b="adapter-recursive",
        system_a="fixed_policy",
        system_b="recursive_policy",
        output_dir=output,
        max_items=2,
        seed=5,
        base_model="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        generator=generator,
    )

    queue = [json.loads(line) for line in (output / "blind-queue.jsonl").read_text(encoding="utf-8").splitlines()]
    key = [json.loads(line) for line in (output / "blind-key.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["labels_read"] is False
    assert len(queue) == len(key) == 2
    assert all("system_a" not in row and "system_b" not in row for row in queue)
    assert all("hidden target" not in json.dumps(row) for row in queue)

    events = [
        {
            "evaluation_source": "final_test_blind_human", "query_id": item["query_id"],
            "source_group_id": item["source_group_id"], "annotator_id": "r1",
            "preference": "A", "duration_seconds": 3.0,
        }
        for item in queue
    ]
    report = analyze_blind_pairwise_feedback(events, key)
    assert report["final_test_accessed"] is True
    assert report["decisive_comparisons"] == 2
    assert sum(row["wins"] for row in report["systems"].values()) == 2


def test_blind_collection_records_timed_pairwise_vote(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({
            "query_id": "q1", "source_group_id": "u1", "input": "Question",
            "context": [], "candidate_a": "Answer A", "candidate_b": "Answer B",
        }) + "\n",
        encoding="utf-8",
    )
    answers = iter(["3", "Both are acceptable."])
    ticks = iter([10.0, 12.5])
    events = collect_blind_pairwise_feedback(
        queue,
        output_path=tmp_path / "ratings.jsonl",
        annotator_id="blind-01",
        input_fn=lambda _prompt: next(answers),
        clock=lambda: next(ticks),
    )
    assert events[0]["preference"] == "both_acceptable"
    assert events[0]["duration_seconds"] == 2.5


def test_blind_collection_resumes_without_repeating_completed_queries(tmp_path):
    queue = tmp_path / "queue.jsonl"
    rows = [
        {
            "query_id": f"q{i}", "source_group_id": f"u{i}", "input": f"Question {i}",
            "context": [], "candidate_a": "Answer A", "candidate_b": "Answer B",
        }
        for i in range(2)
    ]
    queue.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "ratings.jsonl"
    first_answers = iter(["1", "A preferred.", "2"])

    def interrupt_on_second_query(_prompt):
        answer = next(first_answers)
        if answer == "2":
            raise RuntimeError("simulated interruption")
        return answer

    with pytest.raises(RuntimeError, match="simulated interruption"):
        collect_blind_pairwise_feedback(
            queue,
            output_path=output,
            annotator_id="blind-01",
            input_fn=interrupt_on_second_query,
            clock=iter([0.0, 2.0, 3.0]).__next__,
        )

    resumed_answers = iter(["2", "B preferred."])
    resumed = collect_blind_pairwise_feedback(
        queue,
        output_path=output,
        annotator_id="blind-01",
        input_fn=lambda _prompt: next(resumed_answers),
        clock=iter([10.0, 12.0]).__next__,
    )
    assert [row["query_id"] for row in resumed] == ["q0", "q1"]
    assert len({row["query_id"] for row in resumed}) == 2


def test_blind_analysis_rejects_unknown_query_or_non_blind_rows():
    with pytest.raises(ValueError, match="non-final-test"):
        analyze_blind_pairwise_feedback(
            [{"evaluation_source": "live_human", "query_id": "q1"}],
            [{"query_id": "q1", "system_a": "a", "system_b": "b"}],
        )


def test_blind_analysis_reports_interannotator_agreement_and_excludes_cannot_judge():
    key = [
        {"query_id": "q1", "source_group_id": "u1", "system_a": "a", "system_b": "b"},
        {"query_id": "q2", "source_group_id": "u2", "system_a": "a", "system_b": "b"},
        {"query_id": "q3", "source_group_id": "u3", "system_a": "a", "system_b": "b"},
    ]
    feedback = [
        {"evaluation_source": "final_test_blind_human", "query_id": query, "source_group_id": group,
         "annotator_id": annotator, "preference": preference, "duration_seconds": 1.0}
        for query, group, values in (
            ("q1", "u1", ("A", "A")),
            ("q2", "u2", ("B", "B")),
            ("q3", "u3", ("cannot_judge", "A")),
        )
        for annotator, preference in zip(("r1", "r2"), values)
    ]
    report = analyze_blind_pairwise_feedback(feedback, key)
    agreement = report["inter_annotator_agreement"]
    assert agreement["queries_with_multiple_ratings"] == 2
    assert agreement["pairwise_rating_count"] == 2
    assert agreement["pairwise_raw_agreement"] == 1.0
    assert agreement["krippendorff_alpha_nominal"] == 1.0
