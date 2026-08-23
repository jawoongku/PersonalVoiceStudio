import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from mac_voice.checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from mac_voice.lora import freeze_base_parameters, inject_lora


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class CheckpointTests(unittest.TestCase):
    def test_adapter_save_and_reload(self):
        model = nn.Sequential(nn.Linear(2, 1))
        model, _ = inject_lora(model, ["0"], rank=1, alpha=2, dropout=0.0)
        freeze_base_parameters(model)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter.pt"
            save_adapter_checkpoint(model, path, step=3, epoch=1, val_loss=0.5, config={"rank": 1})
            original = model[0].lora_b.detach().clone()
            with torch.no_grad():
                model[0].lora_b.add_(1)
            metadata = load_adapter_checkpoint(model, path)
            self.assertTrue(torch.equal(model[0].lora_b, original))
            self.assertEqual(metadata["step"], 3)
