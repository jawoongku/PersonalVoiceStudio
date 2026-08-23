import tempfile
from pathlib import Path
from unittest import TestCase, mock

import torch

from mac_voice.features import select_provider, validate_feature_artifacts


class FeaturesTests(TestCase):
    def test_feature_artifacts_cover_wavs_and_are_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            split = Path(temp)
            (split / "wav.scp").write_text("a /tmp/a.wav\n", encoding="utf-8")
            torch.save({"a": [0.1, 0.2]}, split / "utt2embedding.pt")
            torch.save({"a": [1, 2]}, split / "utt2speech_token.pt")
            torch.save({"owner": [0.1, 0.2]}, split / "spk2embedding.pt")
            self.assertEqual(validate_feature_artifacts(split), [])

    def test_feature_artifacts_reject_non_finite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            split = Path(temp)
            (split / "wav.scp").write_text("a /tmp/a.wav\n", encoding="utf-8")
            torch.save({"a": [float("nan")]}, split / "utt2embedding.pt")
            torch.save({"a": [1]}, split / "utt2speech_token.pt")
            torch.save({"owner": [0.1]}, split / "spk2embedding.pt")
            self.assertTrue(validate_feature_artifacts(split))

    @mock.patch("mac_voice.features._load_onnxruntime")
    def test_cpu_provider_is_explicit_and_no_cuda(self, load_ort):
        ort = mock.Mock()
        ort.get_available_providers.return_value = ["CPUExecutionProvider", "CoreMLExecutionProvider"]
        load_ort.return_value = ort
        result = select_provider("cpu")
        self.assertEqual(result.selected, "CPUExecutionProvider")
        ort.InferenceSession.assert_not_called()

    @mock.patch("mac_voice.features._load_onnxruntime")
    def test_cuda_is_rejected_before_runtime_import(self, load_ort):
        with self.assertRaises(ValueError):
            select_provider("cuda")
        load_ort.assert_not_called()
