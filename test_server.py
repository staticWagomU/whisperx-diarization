import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from unittest import mock

import numpy as np

import server
import voices

SR = 16000
# SPEAKER_00 は +0.5、SPEAKER_01 は -0.5 の 3 秒ずつ
TRANSCRIPT = {"language": "ja", "segments": [
    {"start": 0.0, "end": 3.0, "text": "こんにちは", "speaker": "SPEAKER_00",
     "words": [{"word": "こんにちは", "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]},
    {"start": 3.0, "end": 6.0, "text": "よろしく", "speaker": "SPEAKER_01",
     "words": [{"word": "よろしく", "start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"}]},
]}


def fake_embed(x):
    return np.array([x.mean(), 0.2], dtype=np.float32)


def write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.asarray(samples) * 32767).astype("<i2").tobytes())


def fake_join(inputs, out, reuse):
    write_wav(out, np.concatenate([np.full(SR * 3, 0.5), np.full(SR * 3, -0.5)]))


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"WHISPERX_VOICES": str(root / "voices"),
                                                "WHISPERX_TRANSCRIPTS": str(root / "transcripts")})
        self.env.start()
        self.patches = [mock.patch.object(voices, "_embedder", lambda: fake_embed),
                        mock.patch.object(server, "join_audio", fake_join),
                        mock.patch.object(server, "transcribe", lambda wav, engine: TRANSCRIPT),
                        mock.patch.object(server, "probe_duration", lambda path: 42.0)]
        for p in self.patches:
            p.start()
        self.httpd = server.serve(0)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for p in self.patches:
            p.stop()
        self.env.stop()
        self.tmp.cleanup()

    def call(self, method, path, body=None, raw=None, headers=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                payload = r.read()
                return r.status, (json.loads(payload) if r.headers.get_content_type() == "application/json" else payload), r.headers
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), e.headers

    def transcribed(self, title="会議 A"):
        _, up, _ = self.call("POST", "/api/uploads?name=a.wav", raw=b"RIFF")
        _, m, _ = self.call("POST", "/api/transcripts", {"title": title, "uploads": [up["token"]], "reuse": False, "engine": "haas"})
        for _ in range(100):
            _, m, _ = self.call("GET", f"/api/transcripts/{m['id']}")
            if m["status"] in ("done", "failed"):
                return m
            time.sleep(0.05)
        self.fail("文字起こしが終わらない")

    def enrolled(self, name, value):
        p = voices.add_person(name)
        return voices.save(p["id"], fake_embed(np.full(10, value)), {"from": "以前", "detail": ""})

    def test_transcribes_upload_and_links_enrolled_voice(self):
        hayashi = self.enrolled("林", 0.5)
        m = self.transcribed()
        self.assertEqual(m["status"], "done")
        self.assertEqual([s["text"] for s in m["segments"]], ["こんにちは", "よろしく"])
        self.assertEqual(m["speakers"]["SPEAKER_00"]["person"], hayashi["id"])
        self.assertEqual(m["speakers"]["SPEAKER_00"]["how"], "voice")
        self.assertIsNone(m["speakers"]["SPEAKER_01"]["person"])
        self.assertEqual(m["dur"], 6.0)

    def test_failed_pipeline_reports_error(self):
        with mock.patch.object(server, "transcribe", side_effect=RuntimeError("speech.cpp が見つかりません")):
            m = self.transcribed()
        self.assertEqual((m["status"], m["error"]), ("failed", "speech.cpp が見つかりません"))

    def test_rejecting_a_voice_match_is_recorded(self):
        hayashi = self.enrolled("林", 0.5)
        m = self.transcribed()
        _, m, _ = self.call("PUT", f"/api/transcripts/{m['id']}/speakers/SPEAKER_00", {"person": None})
        self.assertEqual(m["speakers"]["SPEAKER_00"], {"person": None, "how": None, "score": None, "enrolled": False, "rejected": [hayashi["id"]]})

    def test_one_person_links_to_one_speaker(self):
        hayashi = self.enrolled("林", 0.5)
        m = self.transcribed()
        status, body, _ = self.call("PUT", f"/api/transcripts/{m['id']}/speakers/SPEAKER_01", {"person": hayashi["id"]})
        self.assertEqual(status, 400)
        self.assertIn("別の話者", body["error"])

    def test_download_uses_current_person_names(self):
        m = self.transcribed()
        _, maeda, _ = self.call("POST", "/api/people", {"name": "前田"})
        self.call("PUT", f"/api/transcripts/{m['id']}/speakers/SPEAKER_01", {"person": maeda["id"]})
        self.call("PATCH", f"/api/people/{maeda['id']}", {"name": "前田（自社）"})
        status, txt, headers = self.call("GET", f"/api/transcripts/{m['id']}/download/txt")
        self.assertEqual(status, 200)
        self.assertEqual(txt.decode(), "[話者 1]: こんにちは\n[前田（自社）]: よろしく\n")
        self.assertIn("attachment", headers["Content-Disposition"])

    def test_enrolling_a_linked_speaker_adds_a_recording(self):
        m = self.transcribed("会議 B")
        _, maeda, _ = self.call("POST", "/api/people", {"name": "前田"})
        self.call("PUT", f"/api/transcripts/{m['id']}/speakers/SPEAKER_01", {"person": maeda["id"]})
        _, body, _ = self.call("POST", f"/api/transcripts/{m['id']}/speakers/SPEAKER_01/enroll")
        self.assertEqual(body["person"]["recs"], [{"from": "会議 B", "detail": "話者 2 の発言"}])
        self.assertTrue(body["meta"]["speakers"]["SPEAKER_01"]["enrolled"])
        self.assertIn(maeda["id"], voices.load())

    def test_duplicate_person_names_are_rejected(self):
        self.call("POST", "/api/people", {"name": "林"})
        status, body, _ = self.call("POST", "/api/people", {"name": "林"})
        self.assertEqual(status, 400)
        self.assertIn("すでにいます", body["error"])

    def test_audio_supports_range_requests(self):
        m = self.transcribed()
        status, chunk, headers = self.call("GET", f"/api/transcripts/{m['id']}/audio", headers={"Range": "bytes=0-9"})
        self.assertEqual((status, len(chunk)), (206, 10))
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-9/"))

    def test_enrolling_from_short_file_is_rejected(self):
        _, p, _ = self.call("POST", "/api/people", {"name": "林"})
        _, up, _ = self.call("POST", "/api/uploads?name=a.wav", raw=b"RIFF")
        with mock.patch.object(server, "probe_duration", lambda path: 12.0):
            status, body, _ = self.call("POST", f"/api/people/{p['id']}/enroll", {"upload": up["token"]})
        self.assertEqual(status, 400)
        self.assertIn("30 秒", body["error"])

    def test_reuse_output_becomes_an_upload(self):
        _, up, _ = self.call("POST", "/api/uploads?name=" + urllib.parse.quote("会議.wav"), raw=b"RIFF")
        _, job, _ = self.call("POST", "/api/reuse", {"uploads": [up["token"]]})
        for _ in range(100):
            _, job, _ = self.call("GET", f"/api/reuse/{job['id']}")
            if job["status"] == "done":
                break
            time.sleep(0.05)
        self.assertEqual((job["status"], job["name"]), ("done", "会議_reuse.wav"))
        status, wav, _ = self.call("GET", f"/api/reuse/{job['id']}/audio")
        self.assertEqual((status, wav[:4]), (200, b"RIFF"))
        _, m, _ = self.call("POST", "/api/transcripts", {"title": "t", "uploads": [job["upload"]], "reuse": False, "engine": "haas"})
        self.assertEqual(m["status"], "queued")

    def test_state_lists_people_and_transcripts(self):
        self.enrolled("林", 0.5)
        m = self.transcribed()
        _, state, _ = self.call("GET", "/api/state")
        self.assertEqual([p["name"] for p in state["people"]], ["林"])
        self.assertEqual([t["id"] for t in state["transcripts"]], [m["id"]])
        self.assertNotIn("segments", state["transcripts"][0])


if __name__ == "__main__":
    unittest.main()
