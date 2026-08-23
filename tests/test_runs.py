import tempfile
import unittest
from pathlib import Path

from mac_voice.jobs import create_job
from mac_voice.runs import list_runs


class RunsTests(unittest.TestCase):
    def test_lists_job_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs"
            run = root / "demo"
            create_job(run, command="train", config="ui")
            (run / "adapter.pt").write_bytes(b"x")
            rows = list_runs(root)
            self.assertEqual(rows[0]["job_status"], "queued")
            self.assertTrue(rows[0]["checkpoint"])
