import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from mac_voice.adapter import inject_voice_adapter, inspect_lora_targets


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class AdapterTests(unittest.TestCase):
    def test_inspect_lora_targets_reports_runtime_linear_matches(self):
        class Wrapper:
            pass
        wrapper = Wrapper()
        wrapper.model = torch.nn.Module()
        wrapper.model.llm = torch.nn.Sequential(torch.nn.Linear(2, 2))
        matches = inspect_lora_targets(wrapper, ("0",))
        self.assertEqual(matches["0"], ["0"])

    def test_missing_upstream_llm_is_rejected(self):
        with self.assertRaises(RuntimeError):
            inject_voice_adapter(object(), "/tmp/missing.pt")
