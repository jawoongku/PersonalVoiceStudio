import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from mac_voice.lora import freeze_base_parameters, inject_lora
from mac_voice.trainer_mps import TrainerConfig
from mac_voice.training_loop import fit


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TrainingLoopTests(unittest.TestCase):
    def test_fit_writes_latest_best_metrics_and_resume_state(self):
        model = nn.Sequential(nn.Linear(2, 1))
        model, _ = inject_lora(model, ["0"], rank=1, alpha=2, dropout=0.0)
        freeze_base_parameters(model)
        batches = [torch.ones(2, 2), torch.ones(2, 2)]
        forward = lambda batch: ((model(batch) - 1.0) ** 2).mean()
        with tempfile.TemporaryDirectory() as temp:
            result = fit(model, batches, batches, forward, output_dir=temp, device=torch.device("cpu"), config=TrainerConfig(device="cpu", learning_rate=0.01), max_steps=2)
            root = Path(temp)
            self.assertEqual(result["step"], 2)
            self.assertTrue((root / "checkpoints" / "adapter_latest.pt").is_file())
            self.assertTrue((root / "checkpoints" / "adapter_best.pt").is_file())
            self.assertTrue((root / "training_state.pt").is_file())
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 2)

    def test_fit_can_resume_optimizer_state(self):
        model = nn.Sequential(nn.Linear(2, 1))
        model, _ = inject_lora(model, ["0"], rank=1, alpha=2, dropout=0.0)
        freeze_base_parameters(model)
        batches = [torch.ones(2, 2)]
        forward = lambda batch: ((model(batch) - 1.0) ** 2).mean()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fit(model, batches, batches, forward, output_dir=root, device=torch.device("cpu"), config=TrainerConfig(device="cpu", learning_rate=0.01), max_steps=1)
            result = fit(model, batches, batches, forward, output_dir=root, device=torch.device("cpu"), config=TrainerConfig(device="cpu", learning_rate=0.01), max_steps=2, resume_from=root / "training_state.pt")
            self.assertEqual(result["step"], 2)
