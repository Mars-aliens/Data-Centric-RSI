import gzip
import hashlib
import json

import pytest

from scripts.prepare_helpsteer2 import prepare_helpsteer2


def _write_gzip_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(prompt, response, helpfulness, other=2):
    return {
        "prompt": prompt,
        "response": response,
        "helpfulness": helpfulness,
        "correctness": other,
        "coherence": other,
        "complexity": other,
        "verbosity": other,
    }


def test_prepare_helpsteer2_groups_prompts_seals_validation_and_separates_labels(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "processed"
    train = []
    for index in range(80):
        train.extend(
            [
                _row(f"train prompt {index}", f"weak answer {index}", 1),
                _row(f"train prompt {index}", f"strong answer {index}", 3),
                _row(f"train prompt {index}", f"tied answer {index}", 3, other=4),
            ]
        )
    validation = [
        _row("validation prompt", "validation answer A", 2),
        _row("validation prompt", "validation answer B", 3),
    ]
    _write_gzip_jsonl(raw / "train.jsonl.gz", train)
    _write_gzip_jsonl(raw / "validation.jsonl.gz", validation)
    files = {}
    for name in ("train.jsonl.gz", "validation.jsonl.gz"):
        content = (raw / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(content).hexdigest()}
    (raw / "download-manifest.json").write_text(
        json.dumps(
            {
                "repo_id": "nvidia/HelpSteer2",
                "revision": "990b2711a36180dd19d9c94b8627844866f8982a",
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_helpsteer2(raw, output, split_seed=17)

    assert manifest["status"] == "prepared"
    assert manifest["official_train_validation_prompt_overlap"] == 0
    assert all(count == 0 for count in manifest["prompt_group_overlap_counts"].values())
    final_rows = [json.loads(line) for line in (output / "final_test.jsonl").read_text().splitlines()]
    assert len(final_rows) == 2
    assert {row["input"] for row in final_rows} == {"validation prompt"}
    assert all("ratings" not in row for row in final_rows)
    seed_rows = [json.loads(line) for line in (output / "seed.jsonl").read_text().splitlines()]
    seed_labels = [json.loads(line) for line in (output / "labels" / "seed-labels.jsonl").read_text().splitlines()]
    assert len(seed_rows) == len(seed_labels)
    assert all(row["output"].startswith(("strong answer", "tied answer")) for row in seed_rows)
    assert all(label["helpfulness_tie_for_best"] for label in seed_labels)
    assert all(set(label["ratings"]) == {"helpfulness", "correctness", "coherence", "complexity", "verbosity"} for label in seed_labels)


def test_prepare_helpsteer2_rejects_official_train_validation_overlap(tmp_path):
    raw = tmp_path / "raw"
    shared_train = [_row("  repeated   prompt ", "train response", 2)]
    shared_validation = [_row("repeated prompt", "validation response", 3)]
    _write_gzip_jsonl(raw / "train.jsonl.gz", shared_train)
    _write_gzip_jsonl(raw / "validation.jsonl.gz", shared_validation)
    files = {}
    for name in ("train.jsonl.gz", "validation.jsonl.gz"):
        files[name] = {"sha256": hashlib.sha256((raw / name).read_bytes()).hexdigest()}
    (raw / "download-manifest.json").write_text(
        json.dumps(
            {
                "repo_id": "nvidia/HelpSteer2",
                "revision": "990b2711a36180dd19d9c94b8627844866f8982a",
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="official train/validation prompt overlap"):
        prepare_helpsteer2(raw, tmp_path / "processed")
