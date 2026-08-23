import unittest

from mac_voice.similarity import cosine_similarity


class SimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [1, 2]), 1.0)

    def test_invalid_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])
        with self.assertRaises(ValueError):
            cosine_similarity([0], [1])
