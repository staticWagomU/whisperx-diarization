"""compare_diarization.py が見つけた食い違いのうち、「発話か無音か」を whisper で判定する（issue #1）。

片方のモデルだけが発話と見た区間を切り出して書き起こし、言葉が出てくればそのモデルが正しいとみなす。
両方とも発話と見ていて話者だけが違う区間は、文字起こしでは決められないので「要聴取」として残す。

whispermlx のパイプラインは VAD に pyannote を使うので、判定が pyannote に寄らないよう
mlx_whisper を VAD 無しで直接呼ぶ。

  使い方: uv run python check_disagreements.py [音声ファイル ...]   # 省略時は output/compare/ の全件
  出力:   output/compare/<音声名>/disagreements.md
"""

import sys
from collections import Counter
from pathlib import Path

from compare_diarization import SCRIPT_DIR, align, clock, disagreements, load_pair

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"  # transcribe.sh の既定 large-v3 と同じ
SILENCE_DBFS = -50.0  # Kikimimic の無音ゲートと同じ値
NO_SPEECH_PROB = 0.6
# whisper が無音から作りがちな定型文（YouTube 字幕で学習した名残）
HALLUCINATIONS = ("ご視聴", "チャンネル登録")


def judge(
    ref_names: tuple[str, ...], hyp_names: tuple[str, ...], text: str, no_speech_prob: float, dbfs: float
) -> str:
    if ref_names and hyp_names:
        return "要聴取"
    text = text.strip()
    speech = (
        bool(text)
        and dbfs >= SILENCE_DBFS
        and no_speech_prob < NO_SPEECH_PROB
        and not any(h in text for h in HALLUCINATIONS)
    )
    speaker_side, silent_side = ("Nemotron", "pyannote") if hyp_names else ("pyannote", "Nemotron")
    return f"{speaker_side if speech else silent_side}が正しい"


def transcribe(audio) -> tuple[str, float]:
    """(書き起こし, 発話が無い確率) を返す。区間ごとに独立させるため前の文脈は引き継がない。"""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=WHISPER_MODEL,
        language="ja",
        condition_on_previous_text=False,
        verbose=None,
    )
    segments = result["segments"]
    no_speech = min((s["no_speech_prob"] for s in segments), default=1.0)
    return result["text"].strip(), no_speech


def check(out: Path) -> str:
    import numpy as np
    from whispermlx.audio import load_audio

    ref, hyp = load_pair(out)
    _, mapped, _ = align(ref, hyp)
    audio = load_audio(str(out / "audio.wav"))
    rows, verdicts = [], Counter()
    for start, end, r_names, h_names in disagreements(ref, mapped):
        clip = audio[int(start * 16000) : int(end * 16000)]
        dbfs = float(20 * np.log10(np.sqrt(np.mean(clip**2)) + 1e-10))
        needs_ear = bool(r_names and h_names)
        text, no_speech = ("", 1.0) if needs_ear else transcribe(clip)
        verdict = judge(r_names, h_names, text, no_speech, dbfs)
        verdicts[verdict] += end - start
        rows.append(
            f"| {clock(start)} | {end - start:.1f}秒 | {', '.join(r_names) or '（無音）'} | {', '.join(h_names) or '（無音）'}"
            f" | {dbfs:.0f} | {'-' if needs_ear else f'{no_speech:.2f}'} | {text.replace('|', '｜')} | {verdict} |"
        )
        print(f"  {clock(start)} {verdict} {text[:40]}", file=sys.stderr)

    lines = [
        f"# 食い違った区間の判定: {out.name}",
        "",
        "| 判定 | 合計 |",
        "|---|---|",
        *(f"| {v} | {clock(s)} |" for v, s in verdicts.most_common()),
        "",
        f"判定のしかた: 片方だけが発話と見た区間を whisper（{WHISPER_MODEL}、VAD 無し）で書き起こし、"
        f"言葉が出て、whisper の no_speech_prob が {NO_SPEECH_PROB} 未満で、音量が {SILENCE_DBFS:.0f}dBFS 以上なら発話とみなす。",
        "",
        "| 開始 | 長さ | pyannote | Nemotron | dBFS | no_speech | 書き起こし | 判定 |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    return "\n".join(lines) + "\n"


def main(paths: list[str]) -> None:
    root = SCRIPT_DIR / "output" / "compare"
    outs = [root / Path(p).stem for p in paths] or sorted(
        d for d in root.iterdir() if (d / "nemotron.rttm").exists()
    )
    for out in outs:
        print(f"▶ {out.name}", file=sys.stderr)
        text = check(out)
        (out / "disagreements.md").write_text(text)
        print(text.split("\n\n判定のしかた")[0] + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
