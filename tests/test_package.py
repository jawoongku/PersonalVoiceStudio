import json
import tempfile
import unittest
from pathlib import Path

from mac_voice.package import build_voice_package


class PackageTests(unittest.TestCase):
    def test_package_contains_adapter_and_metadata_without_base_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            (run / "adapter.pt").write_bytes(b"adapter")
            (run / "reference.wav").write_bytes(b"wav")
            (run / "reference.txt").write_text("참조 문장", encoding="utf-8")
            output = root / "voice"
            build_voice_package(run, "test_voice", output, base_model="/models/base", upstream_root=root)
            self.assertTrue((output / "adapter.pt").is_file())
            self.assertTrue((output / "voice.json").is_file())
            self.assertTrue((output / "provenance.json").is_file())
            self.assertFalse((output / "llm.pt").exists())
            self.assertEqual(json.loads((output / "voice.json").read_text())["speaker_id"], "owner")
