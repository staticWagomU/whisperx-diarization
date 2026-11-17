import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import voices


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class VoicesDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["WHISPERX_VOICES"] = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


class MatchTest(unittest.TestCase):
    def test_each_voice_goes_to_its_most_similar_speaker(self):
        mapping = voices.match(
            {"SPEAKER_00": unit(1, 0.1), "SPEAKER_01": unit(0.1, 1), "SPEAKER_02": unit(-1, 0)},
            {"hayashi": unit(1, 0), "maeda": unit(0, 1)},
            threshold=0.5,
        )
        self.assertEqual({k: v[0] for k, v in mapping.items()}, {"SPEAKER_00": "hayashi", "SPEAKER_01": "maeda"})
        self.assertAlmostEqual(mapping["SPEAKER_00"][1], float(unit(1, 0.1) @ unit(1, 0)), places=5)

    def test_one_voice_names_at_most_one_speaker(self):
        # 2 人とも林に近いが、結び付けるのは最も近い 1 人だけ
        mapping = voices.match({"SPEAKER_00": unit(1, 0.3), "SPEAKER_01": unit(1, 0.1)}, {"hayashi": unit(1, 0)}, threshold=0.5)
        self.assertEqual(list(mapping), ["SPEAKER_01"])

    def test_keeps_anonymous_label_below_threshold(self):
        self.assertEqual(voices.match({"SPEAKER_00": unit(1, 1)}, {"hayashi": unit(1, 0)}, threshold=0.9), {})


class SpeakerEmbeddingsTest(unittest.TestCase):
    def test_averages_long_enough_segments_per_speaker(self):
        audio = np.concatenate([np.full(16000 * 2, 1.0), np.full(16000 * 2, -1.0), np.full(1600, 1.0)]).astype(np.float32)
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 4.1, "speaker": "SPEAKER_01"},  # 短すぎて使わない
        ]
        embs = voices.speaker_embeddings(audio, segments, lambda x: np.array([x.mean(), 1.0], dtype=np.float32))
        self.assertEqual(set(embs), {"SPEAKER_00", "SPEAKER_01"})
        np.testing.assert_allclose(embs["SPEAKER_01"], unit(-1, 1), atol=1e-6)


class RenameTest(unittest.TestCase):
    def test_renames_segments_and_words(self):
        transcript = {"segments": [{"speaker": "SPEAKER_00", "words": [{"word": "は", "speaker": "SPEAKER_00"}]},
                                   {"speaker": "SPEAKER_01", "words": []}]}
        renamed = voices.rename(transcript, {"SPEAKER_00": "林"})
        self.assertEqual([s["speaker"] for s in renamed["segments"]], ["林", "SPEAKER_01"])
        self.assertEqual(renamed["segments"][0]["words"][0]["speaker"], "林")


class PeopleTest(VoicesDirTest):
    def test_identify_is_noop_without_enrolled_voices(self):
        voices.add_person("前田")  # 声のない人だけでは照合しない
        transcript = {"segments": [{"speaker": "SPEAKER_00"}]}
        self.assertIs(voices.identify(transcript, "missing.wav"), transcript)

    def test_people_are_found_by_id_and_name(self):
        p = voices.add_person("林")
        self.assertEqual(voices.person(p["id"])["name"], "林")
        self.assertEqual(voices.find("林")["id"], p["id"])
        self.assertIsNone(voices.find("前田"))

    def test_rejects_duplicate_names(self):
        voices.add_person("林")
        with self.assertRaises(ValueError):
            voices.add_person("林")

    def test_renaming_keeps_the_voice(self):
        p = voices.add_person("林")
        voices.save(p["id"], unit(3, 4), {"from": "会議 A", "detail": "話者 1 の発言"})
        voices.rename_person(p["id"], "林（自社）")
        self.assertEqual(voices.people()[0]["name"], "林（自社）")
        np.testing.assert_allclose(voices.load()[p["id"]], unit(3, 4), atol=1e-6)

    def test_enrolling_again_averages_recordings_equally(self):
        p = voices.add_person("林")
        voices.save(p["id"], unit(1, 0), {"from": "会議 A", "detail": ""})
        voices.save(p["id"], unit(0, 1), {"from": "会議 B", "detail": ""})
        np.testing.assert_allclose(voices.load()[p["id"]], unit(1, 1), atol=1e-6)
        self.assertEqual([r["from"] for r in voices.person(p["id"])["recs"]], ["会議 A", "会議 B"])

    def test_removing_a_recording_drops_its_vector(self):
        p = voices.add_person("林")
        voices.save(p["id"], unit(1, 0), {"from": "会議 A", "detail": ""})
        voices.save(p["id"], unit(0, 1), {"from": "会議 B", "detail": ""})
        voices.remove_recording(p["id"], 0)
        np.testing.assert_allclose(voices.load()[p["id"]], unit(0, 1), atol=1e-6)
        self.assertEqual([r["from"] for r in voices.person(p["id"])["recs"]], ["会議 B"])

    def test_forgetting_the_voice_keeps_the_person(self):
        p = voices.add_person("林")
        voices.save(p["id"], unit(1, 0), {"from": "会議 A", "detail": ""})
        voices.forget_voice(p["id"])
        self.assertEqual(voices.load(), {})
        self.assertEqual(voices.person(p["id"])["recs"], [])

    def test_migrates_name_files_from_earlier_format(self):
        np.save(Path(self.tmp.name) / "林.npy", unit(1, 0))  # 初期の形式は 1 次元
        np.save(Path(self.tmp.name) / "前田.npy", np.stack([unit(1, 0), unit(0, 1)]))
        hayashi, maeda = voices.find("林"), voices.find("前田")
        self.assertEqual((len(hayashi["recs"]), len(maeda["recs"])), (1, 2))
        np.testing.assert_allclose(voices.load()[maeda["id"]], unit(1, 1), atol=1e-6)
        self.assertFalse((Path(self.tmp.name) / "林.npy").exists())


if __name__ == "__main__":
    unittest.main()
