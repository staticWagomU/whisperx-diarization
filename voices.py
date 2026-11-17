"""人と声の登録。話者ラベル（SPEAKER_00 …）を、登録した人の声と照合して結び付ける。

  登録:     python voices.py enroll <名前> <音声> [<文字起こし.json> <話者ラベル>]
            文字起こしを渡すと、その話者の発話だけから登録する。渡さなければ音声全体を使う
            同じ名前で登録を重ねると録音ごとに追加し、照合では全録音を同じ重みで平均する
  一覧:     python voices.py list
  保存先:   WHISPERX_VOICES（既定 ~/.ai/whisperx-diarization/voices）。声紋は個人情報なのでリポジトリに置かない
            people.json に人（id・表示名・登録に使った録音）、<id>.npy に録音 1 本につき 1 行の声の特徴
"""

import json
import os
import sys
import uuid
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np

SR = 16000
MIN_SEG, MAX_SEG = 1.5, 10.0  # 短い相づちは埋め込みが不安定、長い区間は先頭だけで足りる
# 手元の会議録音（8 人）の正解ラベルでは、話者単位の類似度は本人の中央値 0.69〜0.84、他人の最大 0.56 だった
THRESHOLD = float(os.environ.get("WHISPERX_VOICE_THRESHOLD", "0.6"))

Embed = Callable[[np.ndarray], np.ndarray]


def voices_dir() -> Path:
    return Path(os.environ.get("WHISPERX_VOICES", "~/.ai/whisperx-diarization/voices")).expanduser()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def _rows(pid: str) -> np.ndarray:
    path = voices_dir() / f"{pid}.npy"
    return np.atleast_2d(np.load(path)) if path.exists() else np.empty((0, 0))


def _write_rows(pid: str, rows: np.ndarray) -> None:
    path = voices_dir() / f"{pid}.npy"
    if len(rows):
        np.save(path.with_suffix(".tmp.npy"), rows)
        path.with_suffix(".tmp.npy").replace(path)
    else:
        path.unlink(missing_ok=True)


def people() -> list[dict]:
    d = voices_dir()
    index = d / "people.json"
    if index.exists():
        return json.loads(index.read_text())["people"]
    found = []
    # 初期の形式は「名前.npy」だけで人の id がない
    for old in sorted(d.glob("*.npy")) if d.is_dir() else []:
        p = {"id": _new_id(), "name": old.stem, "recs": []}
        p["recs"] = [{"from": "以前の登録", "detail": ""} for _ in np.atleast_2d(np.load(old))]
        old.replace(d / f"{p['id']}.npy")
        found.append(p)
    if found:
        _write_people(found)
    return found


def _write_people(ps: list[dict]) -> None:
    voices_dir().mkdir(parents=True, exist_ok=True)
    _write_json(voices_dir() / "people.json", {"people": ps})


def _new_id() -> str:
    return "p" + uuid.uuid4().hex[:8]


def person(pid: str) -> dict:
    p = next((p for p in people() if p["id"] == pid), None)
    if p is None:
        raise KeyError(pid)
    return p


def find(name: str) -> dict | None:
    return next((p for p in people() if p["name"] == name), None)


def _update(pid: str, change: Callable[[dict], None]) -> dict:
    ps = people()
    p = next((p for p in ps if p["id"] == pid), None)
    if p is None:
        raise KeyError(pid)
    change(p)
    _write_people(ps)
    return p


def _check_name(name: str, pid: str | None = None) -> None:
    if not name.strip():
        raise ValueError("名前を入れてください")
    if any(p["name"] == name and p["id"] != pid for p in people()):
        raise ValueError(f"「{name}」はすでにいます")


def add_person(name: str) -> dict:
    _check_name(name)
    p = {"id": _new_id(), "name": name, "recs": []}
    _write_people([*people(), p])
    return p


def rename_person(pid: str, name: str) -> dict:
    _check_name(name, pid)
    return _update(pid, lambda p: p.update(name=name))


def save(pid: str, emb: np.ndarray, rec: dict) -> dict:
    """録音を 1 本足す。照合ではどの録音も同じ重みで平均する。"""
    rows = _rows(pid)
    p = _update(pid, lambda p: p["recs"].append(rec))
    _write_rows(pid, np.vstack([rows, _unit(emb)]) if rows.size else _unit(emb)[None])
    return p


def remove_recording(pid: str, i: int) -> dict:
    rows = _rows(pid)
    p = _update(pid, lambda p: p["recs"].pop(i))
    _write_rows(pid, np.delete(rows, i, axis=0))
    return p


def forget_voice(pid: str) -> dict:
    p = _update(pid, lambda p: p.update(recs=[]))
    _write_rows(pid, np.empty((0, 0)))
    return p


def load() -> dict[str, np.ndarray]:
    """声を登録した人ごとの、全録音の平均。"""
    return {p["id"]: _unit(rows.mean(axis=0)) for p in people() if (rows := _rows(p["id"])).size}


def speaker_embeddings(audio: np.ndarray, segments: list[dict], embed: Embed) -> dict[str, np.ndarray]:
    """話者ごとに、十分な長さの発話の埋め込みを平均する。"""
    acc: dict[str, list[np.ndarray]] = {}
    for s in segments:
        if s.get("speaker") is None or s["end"] - s["start"] < MIN_SEG:
            continue
        clip = audio[int(s["start"] * SR):int(min(s["end"], s["start"] + MAX_SEG) * SR)]
        acc.setdefault(s["speaker"], []).append(_unit(embed(clip)))
    return {k: _unit(np.mean(v, axis=0)) for k, v in acc.items()}


def match(speakers: dict[str, np.ndarray], known: dict[str, np.ndarray],
          threshold: float = THRESHOLD) -> dict[str, tuple[str, float]]:
    """類似度の高い組から 1 対 1 で割り当てる。閾値未満の話者は匿名のまま残す。"""
    pairs = sorted(((float(e @ v), spk, pid) for spk, e in speakers.items() for pid, v in known.items()), reverse=True)
    mapping: dict[str, tuple[str, float]] = {}
    for score, spk, pid in pairs:
        if score >= threshold and spk not in mapping and pid not in (m[0] for m in mapping.values()):
            mapping[spk] = (pid, score)
    return mapping


def rename(transcript: dict, mapping: dict[str, str]) -> dict:
    def sub(item: dict) -> dict:
        return {**item, "speaker": mapping.get(item["speaker"], item["speaker"])} if "speaker" in item else item

    return {**transcript, "segments": [
        {**sub(s), **({"words": [sub(w) for w in s["words"]]} if "words" in s else {})} for s in transcript["segments"]
    ]}


def identify_speakers(transcript: dict, audio: np.ndarray) -> dict[str, tuple[str, float]]:
    """話者ラベル → (人の id, 類似度)。声を登録した人がいなければ空。"""
    known = load()
    return match(speaker_embeddings(audio, transcript["segments"], _embedder()), known) if known else {}


def identify(transcript: dict, audio_path: str) -> dict:
    """声が登録されていれば、話者ラベルを人の名前に置き換える。未登録なら何もしない。"""
    if not load():
        return transcript
    from whispermlx.audio import load_audio

    mapping = identify_speakers(transcript, load_audio(audio_path))
    names = {p["id"]: p["name"] for p in people()}
    for spk, (pid, score) in sorted(mapping.items()):
        print(f"声の照合: {spk} → {names[pid]}（類似度 {score:.2f}）", file=sys.stderr)
    return rename(transcript, {spk: names[pid] for spk, (pid, _) in mapping.items()})


@cache
def _embedder() -> Embed:
    # 日本語で学習した k-washi ECAPA は会議録音で EER 約 38% と、この WeSpeaker（約 20%）に大きく劣った
    import torch
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=os.environ.get("HF_TOKEN"))
    inference = Inference(model, window="whole")
    return lambda x: np.asarray(inference({"waveform": torch.from_numpy(x)[None], "sample_rate": SR})).reshape(-1)


def voice_of(audio: np.ndarray, segments: list[dict] | None = None) -> np.ndarray:
    """発話区間（なければ音声全体）から声の特徴を 1 本作る。"""
    if segments is None:
        segments = [{"start": t, "end": t + MAX_SEG} for t in np.arange(0, len(audio) / SR - MIN_SEG, MAX_SEG)]
    embs = speaker_embeddings(audio, [{**s, "speaker": "x"} for s in segments], _embedder())
    if "x" not in embs:
        raise ValueError(f"{MIN_SEG} 秒以上の発話が見つかりません")
    return embs["x"]


def enroll(name: str, audio_path: str, transcript_path: str | None = None, speaker: str | None = None) -> dict:
    from whispermlx.audio import load_audio

    segments = None
    if transcript_path:
        segments = [s for s in json.loads(Path(transcript_path).read_text())["segments"] if s.get("speaker") == speaker]
    p = find(name) or add_person(name)
    rec = {"from": Path(audio_path).name, "detail": f"{speaker} の発言" if speaker else "音声ファイル"}
    return save(p["id"], voice_of(load_audio(audio_path), segments), rec)


if __name__ == "__main__":
    if sys.argv[1:2] == ["enroll"] and len(sys.argv) in (4, 6):
        p = enroll(*sys.argv[2:])
        print(f"登録しました: {p['name']}（録音 {len(p['recs'])} 本）")
    elif sys.argv[1:2] == ["list"]:
        print("\n".join(f"{p['name']}（録音 {len(p['recs'])} 本）" for p in people()) or "登録した人はいません")
    else:
        sys.exit(__doc__)
