"""文字起こし・結合・人と声・RE-USE を Web で扱う、手元用のサーバー。

  起動:   nix develop -c uv run python server.py [ポート（既定 8765）]
  取り込み: nix develop -c uv run python server.py import <題名> <音声> <文字起こし.json>
          （既存の文字起こしを、匿名ラベルのまま取り込んで声を照合する）
  保存先: WHISPERX_TRANSCRIPTS（既定 ~/.ai/whisperx-diarization/transcripts）、その隣の uploads、
          人と声は voices.py と同じ WHISPERX_VOICES
"""

import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

import voices

HERE = Path(__file__).resolve().parent
FORMATS = ("txt", "srt", "vtt", "tsv", "json")
MIN_ENROLL_SEC = 30
# ponytail: 人と文字起こしの書き込みを 1 本のロックでまとめて守る。1 人用なので取り合いは起きない
LOCK = threading.RLock()
JOBS: queue.Queue = queue.Queue()
REUSE: dict[str, dict] = {}


def transcripts_dir() -> Path:
    return Path(os.environ.get("WHISPERX_TRANSCRIPTS", "~/.ai/whisperx-diarization/transcripts")).expanduser()


def uploads_dir() -> Path:
    return transcripts_dir().parent / "uploads"


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


# ---------- パイプライン（テストでは差し替える） ----------

def run(cmd: list[str], **env: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else f"{cmd[0]} が失敗しました")


def probe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise ValueError(f"音声として読めません: {path.name}") from None


def join_audio(inputs: list[Path], out: Path, reuse: bool) -> None:
    """名前の順につなぎ、16kHz モノラルの WAV にする。reuse なら雑音も抑える。"""
    target = out.with_suffix(".joined.wav") if reuse else out
    args = [a for p in inputs for a in ("-i", str(p))]
    labels = "".join(f"[{i}:a]aresample=16000,aformat=channel_layouts=mono[a{i}];" for i in range(len(inputs)))
    concat = labels + "".join(f"[a{i}]" for i in range(len(inputs))) + f"concat=n={len(inputs)}:v=0:a=1[out]"
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args, "-filter_complex", concat, "-map", "[out]",
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target)])
    if reuse:
        run(["bash", str(HERE / "enhance_reuse.sh"), str(target), str(out)])
        target.unlink()


def transcribe(wav: Path, engine: str) -> dict:
    script = {"haas": "transcribe_haas.sh", "whispermlx": "transcribe.sh"}[engine]
    with tempfile.TemporaryDirectory() as tmp:
        no_voices = Path(tmp) / "voices"  # 照合はサーバーでするので、匿名ラベルのまま受け取る
        run(["bash", str(HERE / script), str(wav), tmp], WHISPERX_VOICES=str(no_voices), WHISPERX_REUSE="0")
        return json.loads((Path(tmp) / f"{wav.stem}.json").read_text())


def load_audio(path: Path) -> np.ndarray:
    from whispermlx.audio import load_audio as load

    return load(str(path))


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# ---------- 文字起こしの保存 ----------

def tdir(tid: str) -> Path:
    if not re.fullmatch(r"t[0-9a-f]+", tid):
        raise KeyError(tid)
    return transcripts_dir() / tid


def meta(tid: str) -> dict:
    path = tdir(tid) / "meta.json"
    if not path.exists():
        raise KeyError(tid)
    return json.loads(path.read_text())


def save_meta(m: dict) -> dict:
    _write_json(tdir(m["id"]) / "meta.json", m)
    return m


def segments(tid: str) -> list[dict]:
    return json.loads((tdir(tid) / "transcript.json").read_text())["segments"]


def detail(tid: str) -> dict:
    m = meta(tid)
    if m["status"] == "done":
        m["segments"] = [{k: s.get(k) for k in ("start", "end", "speaker", "text")} for s in segments(tid)]
    return m


def speaker_names(m: dict) -> dict[str, str]:
    names = {p["id"]: p["name"] for p in voices.people()}
    return {label: names.get(s["person"], f"話者 {i + 1}") for i, (label, s) in enumerate(sorted(m["speakers"].items()))}


def link(tid: str, label: str, pid: str | None) -> dict:
    with LOCK:
        m = meta(tid)
        s = m["speakers"][label]
        if pid == s["person"]:
            return m
        if pid is not None:
            voices.person(pid)
            if any(o["person"] == pid for o in m["speakers"].values()):
                raise ValueError("この人は別の話者に結び付いています")
        if s["how"] == "voice":
            s["rejected"].append(s["person"])  # 照合を見直す材料にする
        s.update(person=pid, how="hand" if pid else None, score=None, enrolled=False)
        return save_meta(m)


def enroll_speaker(tid: str, label: str) -> dict:
    m = meta(tid)
    s = m["speakers"][label]
    if not s["person"]:
        raise ValueError("先に人を選んでください")
    segs = [x for x in segments(tid) if x.get("speaker") == label]
    emb = voices.voice_of(load_audio(tdir(tid) / "audio.wav"), segs)
    n = sorted(m["speakers"]).index(label) + 1
    with LOCK:
        p = voices.save(s["person"], emb, {"from": m["title"], "detail": f"話者 {n} の発言"})
        m = meta(tid)
        m["speakers"][label]["enrolled"] = True
        return {"person": p, "meta": save_meta(m)}


def export(tid: str, fmt: str) -> tuple[str, bytes]:
    from whispermlx.utils import get_writer

    if fmt not in FORMATS:
        raise KeyError(fmt)
    m = meta(tid)
    stem = re.sub(r'[\\/:*?"<>|]', "_", m["title"])
    transcript = voices.rename(json.loads((tdir(tid) / "transcript.json").read_text()), speaker_names(m))
    with tempfile.TemporaryDirectory() as tmp:
        get_writer(fmt, tmp)(transcript, f"{tmp}/{stem}.wav",
                             {"highlight_words": False, "max_line_count": None, "max_line_width": None})
        return f"{stem}.{fmt}", (Path(tmp) / f"{stem}.{fmt}").read_bytes()


def identify(tid: str) -> None:
    mapping = voices.identify_speakers({"segments": segments(tid)}, load_audio(tdir(tid) / "audio.wav"))
    with LOCK:
        m = meta(tid)
        labels = sorted({s["speaker"] for s in segments(tid) if s.get("speaker")})
        m["speakers"] = {label: {"person": mapping[label][0] if label in mapping else None,
                                 "how": "voice" if label in mapping else None,
                                 "score": round(mapping[label][1], 2) if label in mapping else None,
                                 "enrolled": False, "rejected": []} for label in labels}
        save_meta(m)


def state() -> dict:
    ts = sorted((meta(p.name) for p in transcripts_dir().glob("t*") if (p / "meta.json").exists()),
                key=lambda m: m["created"], reverse=True)
    return {"people": voices.people(), "transcripts": ts}


# ---------- アップロードとジョブ ----------

def upload_path(token: str) -> Path:
    d = uploads_dir() / token
    files = list(d.iterdir()) if re.fullmatch(r"u[0-9a-f]+", token) and d.is_dir() else []
    if not files:
        raise KeyError(token)
    return files[0]


def save_upload(name: str, read, length: int) -> dict:
    name = Path(name).name or "audio"
    token = "u" + uuid.uuid4().hex[:12]
    path = uploads_dir() / token / name
    path.parent.mkdir(parents=True)
    with path.open("wb") as f:
        while length > 0:
            chunk = read(min(length, 1 << 20))
            if not chunk:
                break
            f.write(chunk)
            length -= len(chunk)
    try:
        return {"token": token, "name": name, "dur": probe_duration(path)}
    except ValueError:
        shutil.rmtree(path.parent)
        raise


def set_status(tid: str, **change) -> None:
    with LOCK:
        save_meta({**meta(tid), **change})


def process(tid: str) -> None:
    # 途中で止まっても最初からやり直せるよう、どの段階も出力を上書きする
    try:
        m = meta(tid)
        d = tdir(tid)
        set_status(tid, status="running", step="録音をつないでいます" if len(m["uploads"]) > 1 else "音声を整えています", error=None)
        if m["reuse"]:
            set_status(tid, step="雑音を抑えています")
        join_audio([upload_path(u) for u in m["uploads"]], d / "audio.wav", m["reuse"])
        set_status(tid, dur=wav_duration(d / "audio.wav"),
                   step="whispermlx で文字起こししています" if m["engine"] == "whispermlx" else "文字起こしと話者分離をしています")
        _write_json(d / "transcript.json", transcribe(d / "audio.wav", m["engine"]))
        set_status(tid, step="登録した声と照合しています")
        identify(tid)
        set_status(tid, status="done", step=None)
    except Exception as e:  # ワーカーを止めず、画面に理由を出す
        set_status(tid, status="failed", error=str(e) or type(e).__name__)


def create_transcript(title: str, uploads: list[str], reuse: bool, engine: str) -> dict:
    if not title.strip():
        raise ValueError("名前を入れてください")
    if not uploads:
        raise ValueError("録音を追加してください")
    if engine not in ("haas", "whispermlx"):
        raise ValueError(f"未対応の文字起こし: {engine}")
    paths = [upload_path(u) for u in uploads]
    order = [u for _, u in sorted(zip((p.name for p in paths), uploads))]
    tid = "t" + uuid.uuid4().hex[:10]
    tdir(tid).mkdir(parents=True)
    m = save_meta({"id": tid, "title": title.strip(), "created": datetime.now().isoformat(timespec="seconds"),
                   "dur": sum(probe_duration(p) for p in paths), "engine": engine, "reuse": bool(reuse),
                   "uploads": order, "status": "queued", "step": None, "error": None, "speakers": {}})
    JOBS.put(lambda: process(tid))
    return m


def process_reuse(rid: str) -> None:
    job = REUSE[rid]
    try:
        job.update(status="running", step="雑音を抑えています")
        token = "u" + uuid.uuid4().hex[:12]
        out = uploads_dir() / token / job["name"]
        out.parent.mkdir(parents=True)
        join_audio([upload_path(u) for u in job["uploads"]], out, True)
        job.update(status="done", step=None, upload=token, dur=wav_duration(out))
    except Exception as e:
        job.update(status="failed", error=str(e) or type(e).__name__)


def create_reuse(uploads: list[str]) -> dict:
    if not uploads:
        raise ValueError("録音を追加してください")
    first = sorted(upload_path(u).name for u in uploads)[0]
    name = Path(first).stem + (f"_ほか{len(uploads) - 1}本" if len(uploads) > 1 else "") + "_reuse.wav"
    rid = "r" + uuid.uuid4().hex[:10]
    REUSE[rid] = {"id": rid, "status": "queued", "step": None, "error": None, "dur": None,
                  "upload": None, "name": name, "uploads": sorted(uploads, key=lambda u: upload_path(u).name)}
    JOBS.put(lambda: process_reuse(rid))
    return REUSE[rid]


def enroll_upload(pid: str, token: str) -> dict:
    path = upload_path(token)
    if probe_duration(path) < MIN_ENROLL_SEC:
        raise ValueError(f"{MIN_ENROLL_SEC} 秒以上の音声を選んでください（短いと声の特徴が安定しません）")
    voices.person(pid)
    emb = voices.voice_of(load_audio(path))
    with LOCK:
        return voices.save(pid, emb, {"from": path.name, "detail": "音声ファイル"})


def worker() -> None:
    while True:
        JOBS.get()()


def import_transcript(title: str, audio: str, transcript_json: str) -> str:
    tid = "t" + uuid.uuid4().hex[:10]
    d = tdir(tid)
    d.mkdir(parents=True)
    join_audio([Path(audio)], d / "audio.wav", False)
    shutil.copy(transcript_json, d / "transcript.json")
    save_meta({"id": tid, "title": title, "created": datetime.now().isoformat(timespec="seconds"),
               "dur": wav_duration(d / "audio.wav"), "engine": "import", "reuse": False, "uploads": [],
               "status": "running", "step": "登録した声と照合しています", "error": None, "speakers": {}})
    identify(tid)
    set_status(tid, status="done", step=None)
    return tid


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, status: int, body: bytes, ctype: str, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k.replace("_", "-"), v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, data, status: int = 200) -> None:
        self.send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n)) if n else {}

    def file(self, path: Path, download: str | None = None) -> None:
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        extra = {"Accept_Ranges": "bytes"}
        if download:
            extra["Content_Disposition"] = f"attachment; filename*=UTF-8''{quote(download)}"
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end, status = 0, size - 1, 200
        if m and (m[1] or m[2]):
            start, end = (int(m[1]), int(m[2] or size - 1)) if m[1] else (size - int(m[2]), size - 1)
            end, status = min(end, size - 1), 206
            extra["Content_Range"] = f"bytes {start}-{end}/{size}"
        with path.open("rb") as f:
            f.seek(start)
            self.send(status, f.read(end - start + 1), ctype, **extra)

    def dispatch(self) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        for method, pattern, fn in ROUTES:
            m = re.fullmatch(pattern, path)
            if method == self.command and m:
                try:
                    fn(self, *m.groups(), query=parse_qs(url.query))
                except KeyError:
                    self.json({"error": "見つかりません"}, 404)
                except ValueError as e:
                    self.json({"error": str(e)}, 400)
                except Exception as e:
                    self.json({"error": str(e) or type(e).__name__}, 500)
                return
        self.json({"error": "見つかりません"}, 404)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = dispatch


def _download(h, tid, fmt, query):
    name, data = export(tid, fmt)
    h.send(200, data, "text/plain; charset=utf-8" if fmt != "json" else "application/octet-stream",
           Content_Disposition=f"attachment; filename*=UTF-8''{quote(name)}")


def _create_person(h, query):
    with LOCK:
        h.json(voices.add_person(h.body().get("name", "").strip()))


def _rename_person(h, pid, query):
    with LOCK:
        h.json(voices.rename_person(pid, h.body().get("name", "").strip()))


def _fields(body: dict, *keys: str) -> dict:
    missing = [k for k in keys if k not in body]
    if missing:
        raise ValueError(f"項目が足りません: {', '.join(missing)}")
    return {k: body[k] for k in keys}


def _locked(fn):
    def call(*args):
        with LOCK:
            return fn(*args)
    return call


ROUTES = [
    ("GET", r"/", lambda h, query: h.file(HERE / "web" / "index.html")),
    ("GET", r"/api/state", lambda h, query: h.json(state())),
    ("POST", r"/api/uploads", lambda h, query: h.json(save_upload(query.get("name", ["audio"])[0], h.rfile.read,
                                                                    int(h.headers.get("Content-Length") or 0)))),
    ("POST", r"/api/transcripts", lambda h, query: h.json(create_transcript(**_fields(h.body(), "title", "uploads", "reuse", "engine")))),
    ("GET", r"/api/transcripts/(\w+)", lambda h, tid, query: h.json(detail(tid))),
    ("GET", r"/api/transcripts/(\w+)/audio", lambda h, tid, query: h.file(tdir(tid) / "audio.wav")),
    ("GET", r"/api/transcripts/(\w+)/download/(\w+)", _download),
    ("PUT", r"/api/transcripts/(\w+)/speakers/(\w+)", lambda h, tid, label, query: h.json(link(tid, label, h.body().get("person")))),
    ("POST", r"/api/transcripts/(\w+)/speakers/(\w+)/enroll", lambda h, tid, label, query: h.json(enroll_speaker(tid, label))),
    ("POST", r"/api/people", _create_person),
    ("PATCH", r"/api/people/(\w+)", _rename_person),
    ("POST", r"/api/people/(\w+)/enroll", lambda h, pid, query: h.json(enroll_upload(pid, h.body().get("upload", "")))),
    ("DELETE", r"/api/people/(\w+)/recs/(\d+)", lambda h, pid, i, query: h.json(_locked(voices.remove_recording)(pid, int(i)))),
    ("DELETE", r"/api/people/(\w+)/voice", lambda h, pid, query: h.json(_locked(voices.forget_voice)(pid))),
    ("POST", r"/api/reuse", lambda h, query: h.json(create_reuse(h.body().get("uploads", [])))),
    ("GET", r"/api/reuse/(\w+)", lambda h, rid, query: h.json(REUSE[rid])),
    ("GET", r"/api/reuse/(\w+)/audio", lambda h, rid, query: h.file(upload_path(REUSE[rid]["upload"] or ""), REUSE[rid]["name"])),
]


def serve(port: int) -> ThreadingHTTPServer:
    transcripts_dir().mkdir(parents=True, exist_ok=True)
    for m in state()["transcripts"]:
        if m["status"] in ("queued", "running") and m["uploads"]:
            JOBS.put(lambda tid=m["id"]: process(tid))  # 止める前に終わらなかったものをやり直す
    threading.Thread(target=worker, daemon=True).start()
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    if sys.argv[1:2] == ["import"] and len(sys.argv) == 5:
        print(f"取り込みました: {import_transcript(*sys.argv[2:])}")
    elif len(sys.argv) <= 2:
        httpd = serve(int(sys.argv[1]) if len(sys.argv) == 2 else 8765)
        print(f"http://127.0.0.1:{httpd.server_address[1]}/ で開いています（止めるときは Ctrl-C）")
        httpd.serve_forever()
    else:
        sys.exit(__doc__)
