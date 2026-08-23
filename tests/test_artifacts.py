import json
import tempfile
import unittest
from pathlib import Path

from mac_voice.artifacts import create_manifest, verify_manifest


class ArtifactManifestTests(unittest.TestCase):
    def test_create_and_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "adapter.pt").write_bytes(b"adapter")
            manifest = create_manifest(root)
            self.assertEqual(verify_manifest(manifest), [])
            (root / "adapter.pt").write_bytes(b"changed")
            self.assertEqual(verify_manifest(manifest), ["sha256 mismatch: adapter.pt"])
