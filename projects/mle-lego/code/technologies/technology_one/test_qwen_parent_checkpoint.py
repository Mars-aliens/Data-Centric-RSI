from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import qwen_preference


class QwenParentCheckpointTests(unittest.TestCase):
    def _parent(self, base: Path, adapter: Path, config: dict[str, object]):
        recipe = qwen_preference._training_recipe(config)
        recipe_hash = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "backend": "qwen-lora-preference-v1",
            "task_model_version": "parent-v1",
            "base_model_path": str(base),
            "base_model_revision": config["base_model_revision"],
            "adapter_path": str(adapter),
            "adapter_hash": qwen_preference.hash_directory(adapter),
            "label_ids": {"A": 11, "B": 12, "C": 13},
            "training_recipe_hash": recipe_hash,
        }

    def test_valid_parent_adapter_is_loaded_trainably_after_integrity_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            config: dict[str, object] = {
                "base_model_revision": "immutable-rev-1",
                "max_steps": 5,
            }
            parent = self._parent(base, adapter, config)
            base_model = object()
            with (
                patch.object(qwen_preference, "load_base", return_value=base_model) as loader,
                patch.object(
                    qwen_preference.PeftModel,
                    "from_pretrained",
                    return_value="trainable-adapter",
                ) as peft_loader,
            ):
                model, parent_metadata = qwen_preference._load_trainable_model(
                    str(base),
                    qwen_preference.torch.device("cpu"),
                    config=config,
                    initial_checkpoint=parent,
                    current_label_ids={"A": 11, "B": 12, "C": 13},
                    recipe_hash=parent["training_recipe_hash"],
                )

            self.assertEqual(model, "trainable-adapter")
            self.assertEqual(parent_metadata["task_model_version"], "parent-v1")
            loader.assert_called_once_with(str(base), qwen_preference.torch.device("cpu"))
            self.assertTrue(peft_loader.call_args.kwargs["is_trainable"])

    def test_parent_revision_mismatch_is_rejected_before_loading_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter / "adapter.bin").write_bytes(b"adapter")
            config: dict[str, object] = {"base_model_revision": "revision-new"}
            parent = self._parent(base, adapter, {"base_model_revision": "revision-old"})
            with patch.object(qwen_preference, "load_base") as loader:
                with self.assertRaisesRegex(ValueError, "different base model revision"):
                    qwen_preference._load_trainable_model(
                        str(base),
                        qwen_preference.torch.device("cpu"),
                        config=config,
                        initial_checkpoint=parent,
                        current_label_ids={"A": 11, "B": 12, "C": 13},
                        recipe_hash=parent["training_recipe_hash"],
                    )
            loader.assert_not_called()

    def test_nonempty_checkpoint_output_is_rejected_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"
            output.mkdir()
            (output / "partial.bin").write_bytes(b"partial")
            with patch.object(qwen_preference, "load_tokenizer") as tokenizer_loader:
                with self.assertRaises(FileExistsError):
                    qwen_preference.train_checkpoint(
                        [{"input": "x", "target": "preferred"}],
                        config={},
                        seed=17,
                        output_dir=output,
                    )
            tokenizer_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
