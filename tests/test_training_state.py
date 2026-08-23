import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from mac_voice.training_state import MetricsLogger, load_training_state, save_training_state


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TrainingStateTests(unittest.TestCase):
    def test_optimizer_state_round_trip(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.pt"
            save_training_state(path, optimizer=optimizer, step=4, epoch=2)
            restored = torch.optim.AdamW(model.parameters(), lr=0.5)
            metadata = load_training_state(path, optimizer=restored)
            self.assertEqual(metadata["step"], 4)
            self.assertEqual(len(restored.state), len(optimizer.state))

    def test_metrics_logger_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            logger = MetricsLogger(path)
            logger.log(step=1, train_loss=2.0)
            logger.log(step=2, val_loss=1.5)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [1, 2])
