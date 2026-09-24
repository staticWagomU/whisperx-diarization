"""Nemotron の RTTM を whispermlx の文字起こしに付与して各形式に書き出す。"""

import json
import sys
from pathlib import Path

import pandas as pd
from whispermlx.diarize import assign_word_speakers
from whispermlx.utils import get_writer


def assign(transcript: dict, lines: list[str]) -> dict:
    rows = []
    for line in lines:
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"RTTM の形式が不正です: {line}")
        start, duration = float(fields[3]), float(fields[4])
        rows.append((start, start + duration, f"SPEAKER_{int(fields[7].removeprefix('speaker_')):02d}"))
    return assign_word_speakers(pd.DataFrame(rows, columns=["start", "end", "speaker"]), transcript)


def main(json_path: str, rttm_path: str, audio_path: str, outdir: str) -> None:
    transcript = json.loads(Path(json_path).read_text())
    lines = Path(rttm_path).read_text().splitlines()
    get_writer("all", outdir)(
        assign(transcript, lines), audio_path,
        {"highlight_words": False, "max_line_count": None, "max_line_width": None},
    )


if __name__ == "__main__":
    main(*sys.argv[1:5])
