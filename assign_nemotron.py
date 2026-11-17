"""Nemotron の RTTM を whispermlx の文字起こしに付与して各形式に書き出す。"""

import json
import sys
from pathlib import Path

import pandas as pd
from whispermlx.diarize import assign_word_speakers
from whispermlx.utils import LANGUAGES_WITHOUT_SPACES, get_writer

import voices


def assign(transcript: dict, lines: list[str]) -> dict:
    rows = []
    for line in lines:
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"RTTM の形式が不正です: {line}")
        start, duration = float(fields[3]), float(fields[4])
        rows.append((start, start + duration, f"SPEAKER_{int(fields[7].removeprefix('speaker_')):02d}"))
    return assign_word_speakers(pd.DataFrame(rows, columns=["start", "end", "speaker"]), transcript)


def split_by_speaker(transcript: dict, min_words: int = 8) -> dict:
    """Whisper の文区切りを捨て、単語の話者が変わる位置でセグメントを作り直す。

    日本語は 1 文字が 1 単語になり、文字の時刻と話者境界のずれで話者が文字単位に
    ちらつく。min_words 未満の連続は直前の話者に含める。
    """
    # ponytail: 短い相づち（「はい」等）も直前の話者に吸収される。必要なら min_words を下げる
    sep = "" if transcript.get("language") in LANGUAGES_WITHOUT_SPACES else " "
    runs: list[tuple[str | None, list[dict]]] = []
    for word in (w for s in transcript["segments"] for w in s.get("words", [])):
        # 数字などは整列で時刻も話者も付かないことがあるので、直前の話者に含める
        speaker = word.get("speaker", runs[-1][0] if runs else None)
        if runs and speaker == runs[-1][0]:
            runs[-1][1].append(word)
        else:
            runs.append((speaker, [word]))
    merged: list[tuple[str | None, list[dict]]] = []
    for speaker, words in runs:
        if merged and (len(words) < min_words or speaker == merged[-1][0]):
            merged[-1][1].extend(words)
        else:
            merged.append((speaker, words))
    segments = []
    for speaker, words in merged:
        timed = [w for w in words if "start" in w]
        if timed:
            segments.append({
                "start": timed[0]["start"], "end": timed[-1]["end"],
                "text": sep.join(w["word"] for w in words),
                "speaker": speaker, "words": words,
            })
    return {**transcript, "segments": segments}


def write_all(transcript: dict, lines: list[str], audio_path: str, outdir: str) -> None:
    get_writer("all", outdir)(
        voices.identify(split_by_speaker(assign(transcript, lines)), audio_path), audio_path,
        {"highlight_words": False, "max_line_count": None, "max_line_width": None},
    )


def main(json_path: str, rttm_path: str, audio_path: str, outdir: str) -> None:
    transcript = json.loads(Path(json_path).read_text())
    write_all(transcript, Path(rttm_path).read_text().splitlines(), audio_path, outdir)


def main_speech(json_path: str, audio_path: str, outdir: str) -> None:
    """speech.cpp（Haas + Nemotron）の JSON の文字時刻と話者区間から各形式を書き出す。"""
    result = json.loads(Path(json_path).read_text())
    words = [{"word": t["text"], "start": t["start"], "end": t["end"]} for t in result["tokens"] if t["text"].strip()]
    # split_by_speaker が文区切りを捨てるので、全文字を 1 セグメントにまとめて渡す
    segments = [{"start": words[0]["start"], "end": words[-1]["end"],
                 "text": "".join(w["word"] for w in words), "words": words}] if words else []
    lines = [
        f"SPEAKER audio 1 {s['start']:.3f} {s['end'] - s['start']:.3f} <NA> <NA> speaker_{s['speaker']} <NA> <NA>"
        for region in result["regions"] for s in region["speakers"]
    ]
    write_all({"language": "ja", "segments": segments}, lines, audio_path, outdir)


if __name__ == "__main__":
    if sys.argv[1] == "--speech":
        main_speech(*sys.argv[2:5])
    else:
        main(*sys.argv[1:5])
