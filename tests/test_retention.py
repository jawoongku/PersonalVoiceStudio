import tempfile
import unittest
from pathlib import Path

from mac_voice.retention import retention_plan


class RetentionTests(unittest.TestCase):
    def test_plan_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a", "b", "c"):
                (root / name).mkdir()
            plan = retention_plan(root, keep=2)
            self.assertEqual(len(plan["keep"]), 2)
            self.assertEqual(len(plan["candidates"]), 1)
            self.assertTrue((root / "a").is_dir())
