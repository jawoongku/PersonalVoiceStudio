import tempfile
import unittest
from pathlib import Path

from mac_voice.bridge import job_snapshot, voice_catalog
from mac_voice.jobs import create_job


class BridgeTests(unittest.TestCase):
    def test_job_snapshot_reads_job_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = create_job(Path(temp) / "run", command="train", config="ui")
            self.assertEqual(job_snapshot(str(path))["status"], "queued")

    def test_voice_catalog_handles_missing_root(self):
        self.assertEqual(voice_catalog("/tmp/missing-voice-root"), [])
