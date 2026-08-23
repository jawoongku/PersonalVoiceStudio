import unittest
from unittest import mock

from mac_voice.similarity import cosine_similarity, evaluate_audio_similarity


class SimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [1, 2]), 1.0)

    def test_invalid_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])
        with self.assertRaises(ValueError):
            cosine_similarity([0], [1])

    def test_audio_report_uses_embedding_scorer_contract(self):
        with mock.patch("mac_voice.similarity.extract_campplus_embedding", side_effect=([1, 0], [1, 0])):
            report = evaluate_audio_similarity("reference.wav", "generated.wav", "campplus.onnx")
        self.assertEqual(report["embedding_dimension"], 2)
        self.assertAlmostEqual(report["speaker_similarity"], 1.0)
