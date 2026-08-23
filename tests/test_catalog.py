import json
import tempfile
import unittest
from pathlib import Path

from mac_voice.catalog import list_voice_packages


class CatalogTests(unittest.TestCase):
    def test_lists_valid_and_invalid_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid = root / "valid"
            valid.mkdir()
            for filename in ("adapter.pt", "provenance.json", "reference.wav", "reference.txt"):
                (valid / filename).write_bytes(b"x")
            (valid / "voice.json").write_text(json.dumps({"name": "Demo", "base_model": "base", "adapter": "adapter.pt", "speaker_id": "owner", "language": "ko", "sample_rate": 24000}), encoding="utf-8")
            invalid = root / "invalid"
            invalid.mkdir()
            rows = list_voice_packages(root)
            self.assertEqual(len(rows), 2)
            self.assertTrue(next(row for row in rows if row["name"] == "Demo")["valid"])
            self.assertFalse(next(row for row in rows if row["name"] == "invalid")["valid"])
