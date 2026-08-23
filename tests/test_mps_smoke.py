import unittest

from mac_voice.mps_smoke import run_mps_smoke


class MpsSmokeTests(unittest.TestCase):
    def test_probe_reports_unavailable_or_succeeds(self):
        try:
            result = run_mps_smoke()
        except RuntimeError as exc:
            self.assertTrue("MPS" in str(exc) or "PyTorch" in str(exc))
            self.assertIn("Action:", str(exc))
        else:
            self.assertEqual(result["status"], "ok")
