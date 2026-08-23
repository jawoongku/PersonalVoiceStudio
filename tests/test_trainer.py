import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from mac_voice.lora import freeze_base_parameters, inject_lora, validate_gradients
from mac_voice.trainer_mps import TrainerConfig, build_optimizer, parameter_summary, train_one_step


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TrainerTests(unittest.TestCase):
    def test_one_step_updates_only_lora(self):
        model = nn.Sequential(nn.Linear(3, 1))
        model, _ = inject_lora(model, ["0"], rank=2, alpha=4, dropout=0.0)
        freeze_base_parameters(model)
        before = model[0].lora_b.detach().clone()
        config = TrainerConfig(device="cpu", learning_rate=1e-2, grad_clip=1.0)
        optimizer = build_optimizer(model, config)
        loss = train_one_step(
            model,
            torch.ones(4, 3),
            lambda batch: ((model(batch) - 1.0) ** 2).mean(),
            optimizer,
            config,
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model[0].lora_b.detach()))
        lora_ok, frozen_ok, problems = validate_gradients(model)
        self.assertTrue(lora_ok, problems)
        self.assertTrue(frozen_ok, problems)
        self.assertEqual(parameter_summary(model)["frozen"] + parameter_summary(model)["trainable"], parameter_summary(model)["total"])
