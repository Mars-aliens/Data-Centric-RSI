import gzip
import hashlib
import json

import pytest

from scripts.prepare_lccc_association import (
    DATASET_ID,
    DATASET_REVISION,
    FILES,
    _identifier_categories,
    _normalize_utterance,
    prepare_lccc_association,
)


def _write_split(path, dialogues):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for dialogue in dialogues:
            stream.write(json.dumps({"dialog": dialogue}, ensure_ascii=False) + "\n")


def _raw_fixture(raw_dir):
    raw_dir.mkdir()
    train = [
        ["今晚吃什么", "我们去吃火锅怎么样"],
        ["你几点到", "我大概下午三点半到"],
        ["需要带什么", "带上你的电脑和充电器"],
        ["周末有什么安排", "周六我们可以去公园走走"],
        ["会议改到几点", "会议改到明天上午十点"],
        ["下班后见吗", "下班后我们在地铁站门口见"],
    ]
    valid = [
        ["你喜欢什么颜色", "我比较喜欢蓝色和绿色"],
        ["这周忙吗", "这周的安排还算比较轻松"],
    ]
    test = [
        ["午饭吃了吗", "我刚刚和同事一起吃完午饭"],
        ["今晚有空吗", "今晚我有时间可以一起散步"],
    ]
    duplicate = ["明天有时间吗", "明天下午我可以过去找你"]
    train.append(duplicate)
    test.append(duplicate)
    for split, filename in FILES.items():
        _write_split(
            raw_dir / filename,
            {"train": train, "development": valid, "final_test": test}[split],
        )
    files = {}
    for filename in FILES.values():
        path = raw_dir / filename
        files[filename] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (raw_dir / "download-manifest.json").write_text(
        json.dumps(
            {
                "dataset": "lccc",
                "repo_id": DATASET_ID,
                "revision": DATASET_REVISION,
                "files": files,
                "terms_acknowledged": True,
            }
        ),
        encoding="utf-8",
    )


def test_normalize_utterance_removes_tokenization_spaces_and_normalizes_width():
    assert _normalize_utterance(" 明 天 好 ！Ａ ") == "明天好!A"


def test_prepare_lccc_groups_dialogues_and_hides_feedback_references(tmp_path):
    raw_dir = tmp_path / "raw"
    output = tmp_path / "processed"
    _raw_fixture(raw_dir)

    manifest = prepare_lccc_association(
        raw_dir,
        output,
        split_seed=17,
        seed_dialogues=2,
        feedback_dialogues=2,
        confirm_research_terms=True,
        confirm_privacy_review=True,
    )

    assert manifest["status"] == "prepared"
    assert manifest["dialogue_counts"]["exact_duplicates_removed"] == 1
    assert all(value == 0 for value in manifest["group_overlap_counts"].values())
    assert manifest["terms_acknowledged"] is True
    assert manifest["privacy_review_confirmed_by_operator"] is True
    assert manifest["direct_identifier_filtering"]["status"] == (
        "heuristic_dialogue_exclusion_applied"
    )
    assert manifest["near_duplicate_audit"] == "not_performed"

    seed = [
        json.loads(line)
        for line in (output / "seed.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pool = [
        json.loads(line)
        for line in (output / "feedback_pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    references = [
        json.loads(line)
        for line in (output / "labels" / "feedback_pool-reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert seed and pool and references
    assert all(row["split"] == "seed" for row in seed)
    assert all(
        "output" not in row and "reference_continuation" not in row for row in pool
    )
    assert {row["record_id"] for row in pool} == {
        row["record_id"] for row in references
    }
    assert all(row["source_group_id"].startswith("lccc-") for row in seed + pool)


def test_prepare_lccc_requires_terms_and_privacy_acknowledgement(tmp_path):
    raw_dir = tmp_path / "raw"
    _raw_fixture(raw_dir)
    with pytest.raises(ValueError, match="review LCCC terms"):
        prepare_lccc_association(raw_dir, tmp_path / "no-terms")
    with pytest.raises(ValueError, match="privacy"):
        prepare_lccc_association(
            raw_dir,
            tmp_path / "no-privacy",
            confirm_research_terms=True,
        )


def test_prepare_lccc_refuses_manifest_hash_mismatch(tmp_path):
    raw_dir = tmp_path / "raw"
    _raw_fixture(raw_dir)
    with gzip.open(raw_dir / FILES["train"], "at", encoding="utf-8") as stream:
        stream.write(json.dumps({"dialog": ["新问题", "这是一个足够长的回答"]}) + "\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_lccc_association(
            raw_dir,
            tmp_path / "processed",
            seed_dialogues=1,
            feedback_dialogues=1,
            confirm_research_terms=True,
            confirm_privacy_review=True,
        )


def test_direct_identifier_screening_detects_common_patterns():
    assert _identifier_categories(["邮件 example@example.com"] ) == {"email"}
    assert _identifier_categories(["电话 13800138000"] ) == {"mainland_mobile"}
    assert _identifier_categories(["证件 11010519491231002X"] ) == {"mainland_id"}
    assert _identifier_categories(["网址 https://example.com/a"] ) == {"url"}
    assert _identifier_categories(["明天下午我们在公司楼下见面"] ) == set()
