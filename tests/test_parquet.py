import tempfile
import unittest
from pathlib import Path

from mac_voice.parquet import validate_data_list


class ParquetTests(unittest.TestCase):
    def test_missing_data_list(self):
        self.assertTrue(validate_data_list("/tmp/not-a-real-data-list"))

    def test_missing_parquet_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.list"
            path.write_text("missing.parquet\n", encoding="utf-8")
            errors = validate_data_list(path)
            self.assertIn("parquet file not found", errors[0])

    def test_nested_feature_lists_are_recognized_as_top_level_columns(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root / "parquet_000000000.tar"
            pd.DataFrame({
                "utt": ["a"], "audio_data": [b"wav"], "wav": ["a.wav"],
                "text": ["text"], "spk": ["owner"], "utt_embedding": [[0.1, 0.2]],
                "spk_embedding": [[0.1, 0.2]], "speech_token": [[1, 2]],
            }).to_parquet(shard)
            data_list = root / "data.list"
            data_list.write_text(str(shard) + "\n", encoding="utf-8")
            self.assertEqual(validate_data_list(data_list, require_features=True), [])
