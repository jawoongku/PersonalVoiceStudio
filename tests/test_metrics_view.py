import tempfile
import unittest
from pathlib import Path

from mac_voice.metrics_view import summarize_metrics


class MetricsViewTests(unittest.TestCase):
    def test_summarizes_recent_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text('{"step": 1, "train_loss": 2.5, "val_loss": 3.0}\n', encoding="utf-8")
            self.assertIn("step=1 train=2.5 val=3.0", summarize_metrics(path))
