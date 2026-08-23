import tempfile
import unittest
from pathlib import Path

from mac_voice.history import append_tts_history, read_tts_history


class HistoryTests(unittest.TestCase):
    def test_append_and_read_latest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.jsonl"
            append_tts_history(path, voice="demo", text="첫 문장", output="a.wav")
            append_tts_history(path, voice="demo", text="둘째 문장", output="b.wav")
            rows = read_tts_history(path, limit=1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["output"], "b.wav")
