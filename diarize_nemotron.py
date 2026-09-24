# /// script
# requires-python = ">=3.11"
# dependencies = [
#   # Nemotron 3 Diarization は PR #970 で入り、まだリリースに含まれていない（最新は 2026-09-21 の 0.5.5）
#   "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@03a4d99e6cac5c13523c1027761ab33abfdcd30d",
# ]
# ///
"""Nemotron-3-Diarization（mlx-audio）で話者分離し、RTTM を書き出す。

mlx-audio は transformers>=5.14 / huggingface_hub>=1.0 を要求し、whispermlx が固定している版と
両立しない。そのためプロジェクトの環境には入れず、uv run --script で専用の環境を作って動かす。

  使い方:   uv run --script diarize_nemotron.py <16kHz モノラル wav> <出力 .rttm>
  標準出力: 話者分離にかかった秒数（モデル読み込みを除く）
"""

import sys
from pathlib import Path

from mlx_audio.vad import load

MODEL = "mlx-community/Nemotron-3-Diarization"


def main(wav: str, rttm: str) -> None:
    model = load(MODEL, strict=True)
    result = model.generate(wav)
    Path(rttm).write_text(result.text + "\n")
    print(result.total_time)


if __name__ == "__main__":
    main(*sys.argv[1:3])
