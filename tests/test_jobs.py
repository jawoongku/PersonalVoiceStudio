import tempfile
import unittest
from pathlib import Path

from mac_voice.jobs import create_job, read_job, update_job


class JobTests(unittest.TestCase):
    def test_job_lifecycle_is_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = create_job(Path(temp) / "run", command="train", config="training.yaml")
            self.assertEqual(read_job(path)["status"], "queued")
            update_job(path, "running", step=3)
            job = read_job(path)
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["step"], 3)

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = create_job(Path(temp) / "run", command="train", config="training.yaml")
            with self.assertRaises(ValueError):
                update_job(path, "unknown")
