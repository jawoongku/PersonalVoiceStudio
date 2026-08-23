import tempfile
import unittest
from pathlib import Path

from mac_voice.project import initialize_project


class ProjectTests(unittest.TestCase):
    def test_initialize_project_creates_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "voice"
            result = initialize_project(root)
            self.assertEqual(Path(result["dataset"]) / "raw", root / "data" / "my_voice" / "raw")
            self.assertTrue((root / "data/my_voice/transcripts.csv").is_file())
            self.assertTrue((root / "training.yaml").is_file())

    def test_non_empty_project_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "voice"
            root.mkdir()
            (root / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                initialize_project(root)
