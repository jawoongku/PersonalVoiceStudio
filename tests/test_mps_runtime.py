import unittest
from unittest.mock import patch

from mac_voice.mps_runtime import _major, render


class MpsRuntimeTests(unittest.TestCase):
    def test_major_version(self):
        self.assertEqual(_major("26.5.2"), 26)
        self.assertIsNone(_major("unknown"))

    def test_render_marks_actionable_failure(self):
        text = render({
            "architecture": "arm64", "macos": "26.5.2", "python": "3.10", "torch": "2.13.0",
            "built": True, "available": False, "tensor_probe": False,
            "status": "os-runtime-mismatch", "action": "use a compatible build", "error": "RuntimeError: mismatch",
        })
        self.assertIn("os-runtime-mismatch", text)
        self.assertIn("use a compatible build", text)
