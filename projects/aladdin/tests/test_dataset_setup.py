import json
from pathlib import Path

import pytest

from scripts.download_datasets import ROOT, download, load_registry


def test_dataset_registry_pins_sources_and_excludes_prism_survey():
    registry = load_registry()
    assert registry["prism"]["revision"] == "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6"
    assert "survey.jsonl" in registry["prism"]["excluded_files"]
    assert registry["helpsteer2"]["revision"] == "990b2711a36180dd19d9c94b8627844866f8982a"


def test_prism_download_requires_explicit_terms_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match="Review configs/datasets.json"):
        download("prism", tmp_path)


def test_relative_data_root_is_project_relative():
    path = Path("data/raw")
    assert (ROOT / path).is_absolute()


def test_venv_clean_is_ignored_by_gitignore():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".venv*/" in ignore


def test_download_writes_revision_and_hash_manifest(tmp_path, monkeypatch):
    def fake_snapshot_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        for name in kwargs["allow_patterns"]:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    path = download("helpsteer2", tmp_path)
    manifest = json.loads((path / "download-manifest.json").read_text(encoding="utf-8"))

    assert manifest["revision"] == load_registry()["helpsteer2"]["revision"]
    assert set(manifest["files"]) == set(load_registry()["helpsteer2"]["files"])
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"].values())
