import gzip
import json

from scripts.audit_datasets import audit_helpsteer2


def test_helpsteer_audit_reports_prompt_overlap_without_text(tmp_path):
    rows = [
        {
            "prompt": "shared prompt",
            "response": f"answer {i}",
            "helpfulness": i,
            "correctness": i,
            "coherence": i,
            "complexity": i,
            "verbosity": i,
        }
        for i in range(2)
    ]
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows[:1]:
                handle.write(json.dumps(row) + "\n")

    report = audit_helpsteer2(tmp_path)

    assert report["splits"]["train"]["rows"] == 1
    assert report["splits"]["validation"]["malformed_rows"] == 0
    assert report["train_validation_prompt_overlap"] == 1
    assert "shared prompt" not in json.dumps(report)
