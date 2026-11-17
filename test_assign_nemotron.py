import json
import os
import tempfile
import unittest
from pathlib import Path

from assign_nemotron import main, main_speech, split_by_speaker


class AssignNemotronTest(unittest.TestCase):
    def setUp(self):
        # 実際に登録した声を読まない
        self.voices = tempfile.TemporaryDirectory()
        os.environ["WHISPERX_VOICES"] = self.voices.name

    def tearDown(self):
        self.voices.cleanup()

    def test_writes_speaker_labels_to_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = {
                "language": "ja",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "こんにちは、良い天気ですね", "words": [
                        {"start": i * 0.1, "end": i * 0.1 + 0.1, "word": c}
                        for i, c in enumerate("こんにちはこんにちは良い天気ですね快晴だ")
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
            self.assertEqual(
                [(s["speaker"], s["text"], s["start"], s["end"]) for s in result["segments"]],
                [("SPEAKER_00", "こんにちはこんにちは", 0.0, 1.0), ("SPEAKER_01", "良い天気ですね快晴だ", 1.0, 2.0)],
            )
            self.assertTrue(all((root / f"audio.{ext}").exists() for ext in ("txt", "srt", "vtt", "tsv")))
            self.assertEqual(
                (root / "audio.txt").read_text().splitlines(),
                ["[SPEAKER_00]: こんにちはこんにちは", "[SPEAKER_01]: 良い天気ですね快晴だ"],
            )

    def test_writes_all_outputs_from_speech_cpp_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = "こんにちはこんにちは", "良い天気ですね快晴だ"
            tokens = [{"start": i * 0.1, "end": i * 0.1 + 0.1, "text": c} for i, c in enumerate(first + second)]
            (root / "speech.json").write_text(json.dumps({
                "text": first + second,
                "tokens": tokens,
                "regions": [{"start": 0.0, "end": 2.0, "text": first + second, "speakers": [
                    {"speaker": 0, "start": 0.0, "end": 1.0},
                    {"speaker": 1, "start": 1.0, "end": 2.0},
                ]}],
            }))
            main_speech(str(root / "speech.json"), "audio.wav", tmp)
            self.assertEqual(
                (root / "audio.txt").read_text().splitlines(),
                [f"[SPEAKER_00]: {first}", f"[SPEAKER_01]: {second}"],
            )
            self.assertTrue(all((root / f"audio.{ext}").exists() for ext in ("json", "srt", "vtt", "tsv")))

    def test_split_keeps_untimed_words_with_current_speaker(self):
        segments = split_by_speaker({"language": "en", "segments": [{"words": [
            {"word": "Hi", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"word": "42"},
            {"word": "yes", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_01"},
        ]}]}, min_words=1)["segments"]
        self.assertEqual(
            [(s["speaker"], s["text"]) for s in segments],
            [("SPEAKER_00", "Hi 42"), ("SPEAKER_01", "yes")],
        )

    def test_split_absorbs_short_speaker_flicker(self):
        segments = split_by_speaker({"language": "ja", "segments": [{"words": [
            {"word": "LIN", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"word": "E", "start": 2.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"word": "に", "start": 2.2, "end": 4.0, "speaker": "SPEAKER_00"},
        ]}]})["segments"]
        self.assertEqual([(s["speaker"], s["text"]) for s in segments], [("SPEAKER_00", "LINEに")])


if __name__ == "__main__":
    unittest.main()
