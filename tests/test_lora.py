import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on selected environment
    torch = None
    nn = None

from mac_voice.lora import freeze_base_parameters, inject_lora, validate_gradients


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class LoraTests(unittest.TestCase):
    def test_inject_freeze_and_gradient_invariants(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        model, matched = inject_lora(model, ["0", "1"], rank=2, alpha=4, dropout=0.0)
        self.assertEqual(matched, ["0", "1"])
        stats = freeze_base_parameters(model)
        self.assertGreater(stats.trainable, 0)
        self.assertLess(stats.trainable, stats.total)
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        lora_ok, frozen_ok, problems = validate_gradients(model)
        self.assertTrue(lora_ok, problems)
        self.assertTrue(frozen_ok, problems)

    def test_zero_match_fails(self):
        with self.assertRaises(ValueError):
            inject_lora(nn.Linear(2, 2), ["q_proj"])
