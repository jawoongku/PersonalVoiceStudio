import tempfile
import unittest
from pathlib import Path

from mac_voice.config import load_config, validate_training_config


class ConfigTests(unittest.TestCase):
    def test_load_and_validate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("model_dir: /tmp/model\ntraining:\n  batch_size: 1\nlora:\n  rank: 2\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(validate_training_config(config), [])

    def test_invalid_lora_dropout(self):
        errors = validate_training_config({"model_dir": "/tmp/model", "lora": {"dropout": 1.0}})
        self.assertIn("lora.dropout must be in [0, 1)", errors)
