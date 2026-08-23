import json
import tempfile
import unittest
from pathlib import Path

from mac_voice.voice import require_adapter_inference_support, validate_voice_package


class VoiceTests(unittest.TestCase):
    def test_invalid_package_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            voice, errors = validate_voice_package(temp)
            self.assertEqual(voice, {})
            self.assertTrue(any("adapter.pt" in error for error in errors))

    def test_valid_metadata_is_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in ("adapter.pt", "provenance.json", "reference.wav", "reference.txt"):
                (root / filename).write_bytes(b"x")
            (root / "voice.json").write_text(json.dumps({
                "name": "voice", "base_model": "base", "adapter": "adapter.pt",
                "speaker_id": "owner", "language": "ko", "sample_rate": 24000,
            }), encoding="utf-8")
            voice, errors = validate_voice_package(root)
            self.assertEqual(errors, [])
            self.assertEqual(voice["language"], "ko")

    def test_adapter_inference_prerequisites_accept_connected_path(self):
        with tempfile.TemporaryDirectory() as temp:
            require_adapter_inference_support({"adapter": "adapter.pt"}, temp)
