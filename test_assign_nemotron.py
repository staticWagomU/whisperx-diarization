import json
import tempfile
import unittest
from pathlib import Path

from assign_nemotron import main


class AssignNemotronTest(unittest.TestCase):
    def test_writes_speaker_labels_to_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = {
                "language": "ja",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "こんにちは", "words": [
                        {"start": 0.1, "end": 0.9, "word": "こん"},
                        {"start": 1.1, "end": 1.9, "word": "にちは"},
                    ]},
                ],
            }
            (root / "raw.json").write_text(json.dumps(transcript))
            (root / "speakers.rttm").write_text(
                "SPEAKER audio 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
                "SPEAKER audio 1 1.000 1.000 <NA> <NA> speaker_1 <NA> <NA>\n"
            )
            main(str(root / "raw.json"), str(root / "speakers.rttm"), "audio.wav", tmp)
            result = json.loads((root / "audio.json").read_text())
            self.assertEqual(result["segments"][0]["words"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(result["segments"][0]["words"][1]["speaker"], "SPEAKER_01")
            self.assertEqual(result["segments"][0]["speaker"], "SPEAKER_00")
            self.assertTrue(all((root / f"audio.{ext}").exists() for ext in ("txt", "srt", "vtt", "tsv")))


if __name__ == "__main__":
    unittest.main()
