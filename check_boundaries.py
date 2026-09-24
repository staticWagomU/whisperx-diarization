"""各音声ファイルの「末尾」と「先頭」だけを文字起こしし、連結順の妥当性を目視判定するための下調べ。

録音を誤って停止した場合、前ファイルの末尾と次ファイルの先頭は文の途中で繋がる。
全編を文字起こしすると数時間かかるため、境界の 60 秒だけを見る。
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SNIPPET_SEC = 60


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract(src: str, dst: str, start: float, dur: float) -> None:
    """whisper が扱う 16kHz モノラル wav に切り出す。"""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
         "-i", src, "-ac", "1", "-ar", "16000", dst],
        check=True,
    )


def main(files: list[str]) -> None:
    from whispermlx import load_model

    model_name = os.environ.get("WHISPERX_MODEL", "large-v3")
    language = os.environ.get("WHISPERX_LANGUAGE", "ja")
    model = load_model(
        model_name,
        device=os.environ.get("WHISPERX_DEVICE", "cpu"),
        compute_type=os.environ.get("WHISPERX_COMPUTE", "default"),
        language=language,
        vad_method=os.environ.get("WHISPERX_VAD_METHOD", "pyannote"),
    )
    research_dir = Path.home() / ".ai" / "whisperx-diarization"
    research_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="boundary-", dir=research_dir) as tmp:
        for idx, path in enumerate(files, 1):
            dur = probe_duration(path)
            name = os.path.basename(path)
            for pos in ("head", "tail"):
                start = 0.0 if pos == "head" else max(0.0, dur - SNIPPET_SEC)
                wav = os.path.join(tmp, f"{idx:02d}-{pos}.wav")
                extract(path, wav, start, SNIPPET_SEC)
                result = model.transcribe(wav, language=language)
                text = "".join(s["text"] for s in result["segments"]).strip()
                print(f"[{idx:02d}] {name}  ({dur/60:.1f}分)  {pos.upper()}")
                print(f"     {text}")
                sys.stdout.flush()
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
