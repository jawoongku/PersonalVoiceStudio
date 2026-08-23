import unittest

from mac_voice.narrate import split_text


class NarrateTests(unittest.TestCase):
    def test_split_text_respects_sentence_boundaries(self):
        chunks = split_text("첫 문장입니다. 두 번째 문장입니다!", max_chars=12)
        self.assertEqual(chunks, ["첫 문장입니다.", "두 번째 문장입니다!"])

    def test_empty_text(self):
        self.assertEqual(split_text("  "), [])
