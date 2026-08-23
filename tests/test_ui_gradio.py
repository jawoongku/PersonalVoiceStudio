import tempfile
import unittest
import wave
from pathlib import Path

from mac_voice.ui_gradio import RECOMMENDED_SENTENCES, inspect_recording, register_recording


class GradioPrototypeTests(unittest.TestCase):
    def test_recommended_sentences_are_non_empty(self):
        self.assertGreaterEqual(len(RECOMMENDED_SENTENCES), 5)
        self.assertTrue(all(sentence.strip() for sentence in RECOMMENDED_SENTENCES))

    def test_recording_inspection_reports_valid_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes((1000).to_bytes(2, "little", signed=True) * 24000)
            report = inspect_recording(path, "테스트 문장입니다.")
            self.assertIn("판정: 사용 가능", report)

    def test_recording_inspection_requires_transcript(self):
        self.assertIn("transcript", inspect_recording("/tmp/missing.wav", ""))

    def test_register_recording_copies_wav_and_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "recording.wav"
            dataset = Path(temp) / "dataset"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes((1000).to_bytes(2, "little", signed=True) * 24000)
            result = register_recording(dataset, source, "저장할 문장입니다.")
            self.assertIn("저장 완료", result)
            self.assertTrue((dataset / "raw" / "0001.wav").is_file())
            self.assertIn("저장할 문장입니다.", (dataset / "transcripts.csv").read_text(encoding="utf-8"))
