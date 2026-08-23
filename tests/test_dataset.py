import csv
import tempfile
import wave
from pathlib import Path
from unittest import TestCase

from mac_voice.dataset import prepare_dataset, validate_dataset


class DatasetTests(TestCase):
    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        raw = dataset / "raw"
        raw.mkdir(parents=True)
        with (dataset / "transcripts.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "text"])
            writer.writerow(["0001.wav", "테스트 문장입니다."])
        with wave.open(str(raw / "0001.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((1000).to_bytes(2, "little", signed=True) * 1600)
        return dataset

    def test_validate_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            records, errors = validate_dataset(self._make_dataset(Path(temp)))
            self.assertEqual(errors, [])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].errors, [])
            self.assertEqual(records[0].sample_rate, 16000)

    def test_prepare_dataset_creates_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "prepared"
            prepare_dataset(self._make_dataset(root), output)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "train" / "wav.scp").is_file())
            prepared_wav = next((output / "audio").rglob("*.wav"))
            with wave.open(str(prepared_wav), "rb") as handle:
                self.assertEqual(handle.getframerate(), 24000)
                self.assertEqual(handle.getnchannels(), 1)
