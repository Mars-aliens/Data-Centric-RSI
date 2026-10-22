import json
import hashlib

from scripts.prepare_prism import prepare_prism


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_prism_filters_pii_splits_by_user_and_separates_labels(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()
    utterances = []
    metadata = []
    for user in range(30):
        uid = f"u{user}"
        cid = f"c{user}"
        for turn in range(2):
            iid = f"i{user}-{turn}"
            for candidate in range(2):
                utterance_id = f"ut{user}-{turn}-{candidate}"
                utterances.append(
                    {
                        "utterance_id": utterance_id,
                        "user_id": uid,
                        "conversation_id": cid,
                        "interaction_id": iid,
                        "turn": turn,
                        "user_prompt": f"prompt {turn}",
                        "model_response": f"response {turn} {candidate}",
                        "score": candidate + 1,
                        "if_chosen": candidate == 1,
                    }
                )
                metadata.append(
                    {
                        "column_id": "model_response",
                        "user_id": uid,
                        "conversation_id": cid,
                        "interaction_id": iid,
                        "utterance_id": utterance_id,
                        "pii_flag": False,
                        "pii_manual_flag": None,
                        "en_flag": True,
                        "moderation_flag": {"flagged": False},
                    }
                )
            metadata.append(
                {
                    "column_id": "user_prompt",
                    "user_id": uid,
                    "conversation_id": cid,
                    "interaction_id": iid,
                    "utterance_id": None,
                    "pii_flag": False,
                    "pii_manual_flag": None,
                    "en_flag": True,
                    "moderation_flag": {"flagged": False},
                }
            )
    metadata[-1]["pii_flag"] = True
    _write(raw / "utterances.jsonl", utterances)
    conversations = []
    for user in range(30):
        uid = f"u{user}"
        cid = f"c{user}"
        conversations.append(
            {
                "user_id": uid,
                "conversation_id": cid,
                "conversation_type": "unguided",
                "open_feedback": f"comment {user}",
            }
        )
        metadata.append(
            {
                "column_id": "open_feedback",
                "user_id": uid,
                "conversation_id": cid,
                "interaction_id": None,
                "utterance_id": None,
                "pii_flag": False,
                "pii_manual_flag": None,
                "en_flag": True,
                "moderation_flag": {"flagged": False},
            }
        )
    _write(raw / "conversations.jsonl", conversations)
    _write(raw / "metadata.jsonl", metadata)
    hashes = {
        name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
        for name in ("utterances.jsonl", "metadata.jsonl")
    }
    (raw / "download-manifest.json").write_text(
        json.dumps(
            {
                "repo_id": "HannahRoseKirk/prism-alignment",
                "revision": "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6",
                "files": {name: {"sha256": value} for name, value in hashes.items()},
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_prism(raw, out, split_seed=17)

    assert manifest["status"] == "prepared"
    assert all(value == 0 for value in manifest["user_overlap_counts"].values())
    assert manifest["excluded_reasons"]["prompt_missing_or_flagged"] == 2
    assert not (out / "survey.jsonl").exists()
    pool = [json.loads(line) for line in (out / "feedback_pool.jsonl").read_text().splitlines()]
    pool_labels = [
        json.loads(line)
        for line in (out / "labels" / "feedback_pool-labels.jsonl").read_text().splitlines()
    ]
    assert all("score" not in row and "if_chosen" not in row for row in pool)
    assert len(pool) == len(pool_labels)
    assert all(len(row["context"]) == 2 for row in pool if row["turn"] == 1)
    seed_labels = [
        json.loads(line)
        for line in (out / "labels" / "seed-labels.jsonl").read_text().splitlines()
    ]
    assert seed_labels
    assert all(row["if_chosen"] for row in seed_labels)
    conversation_pool = [
        json.loads(line)
        for line in (out / "conversation_feedback_pool.jsonl").read_text().splitlines()
    ]
    conversation_labels = [
        json.loads(line)
        for line in (out / "labels" / "conversation-feedback-labels.jsonl").read_text().splitlines()
    ]
    assert all("open_feedback" not in row for row in conversation_pool)
    assert len(conversation_pool) == len(conversation_labels)
