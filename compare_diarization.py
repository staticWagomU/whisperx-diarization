"""同じ音声を pyannote と Nemotron-3-Diarization にかけ、話者分離の結果を比べる（issue #1）。

正解ラベルが無いので、pyannote を基準にした相互 DER と、2 つが食い違った区間の一覧を出す。
一覧の時刻を実際に聞いて、どちらが正しいかを判断する。

  使い方: uv run python compare_diarization.py <音声ファイル> [...]
  出力:   output/compare/<音声名>/ に audio.wav / pyannote.rttm / nemotron.rttm / timing.json / report.md

できあがったファイルは次回スキップするので、分離をやり直すときは該当の .rttm を消す。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"  # whispermlx の既定と同じ
TOP_N = 20

Segment = tuple[float, float, str]


def disagreements(
    ref: list[Segment], hyp: list[Segment], min_len: float = 1.0, step: float = 0.01
) -> list[tuple[float, float, tuple[str, ...], tuple[str, ...]]]:
    """2 つの分離結果が食い違う区間を (開始, 終了, ref の話者, hyp の話者) で返す。

    10ms のフレームごとに「誰が話しているか」の集合を比べるので、同時発話もそのまま扱える。
    hyp の話者名は ref の話者名に対応付けておくこと。
    """
    labels = sorted({label for *_, label in ref + hyp})
    bit = {label: 1 << i for i, label in enumerate(labels)}
    n = round(max((end for _, end, _ in ref + hyp), default=0) / step)

    def frames(segments: list[Segment]) -> list[int]:
        out = [0] * n
        for start, end, label in segments:
            for i in range(round(start / step), round(end / step)):
                out[i] |= bit[label]
        return out

    def names(mask: int) -> tuple[str, ...]:
        return tuple(label for label in labels if mask & bit[label])

    r, h = frames(ref), frames(hyp)
    regions = []
    i = 0
    while i < n:
        if r[i] == h[i]:
            i += 1
            continue
        # 話者が入れ替わりながら続く食い違いは、聞き直すときに 1 か所なので 1 区間にまとめる
        j, r_mask, h_mask = i, 0, 0
        while j < n and r[j] != h[j]:
            r_mask |= r[j]
            h_mask |= h[j]
            j += 1
        if (j - i) * step >= min_len:
            regions.append((round(i * step, 2), round(j * step, 2), names(r_mask), names(h_mask)))
        i = j
    return regions


def read_rttm(path: Path) -> list[Segment]:
    segments = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER":
            start, duration = float(fields[3]), float(fields[4])
            segments.append((start, start + duration, fields[7]))
    return segments


def to_wav(audio: Path, wav: Path) -> None:
    """両モデルが受け取る 16kHz モノラル wav に揃える。"""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )


def run_pyannote(wav: Path, rttm: Path) -> float:
    """話者分離にかかった秒数を返す（モデル読み込みを除く）。"""
    import time

    import torch
    from pyannote.audio import Pipeline
    from whispermlx.audio import load_audio

    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN が未設定です（pyannote のモデル取得に必要）")
    device = torch.device(os.environ.get("WHISPERX_DEVICE", "cpu"))
    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=token).to(device)
    # ファイルパスを渡すと torchcodec が ffmpeg を探して失敗するので、whispermlx と同じく波形で渡す
    waveform = torch.from_numpy(load_audio(str(wav))[None, :])
    started = time.perf_counter()
    output = pipeline({"waveform": waveform, "sample_rate": 16000})
    elapsed = time.perf_counter() - started
    rttm.write_text(
        "".join(
            f"SPEAKER audio 1 {turn.start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
        )
    )
    return elapsed


def run_nemotron(wav: Path, rttm: Path) -> float:
    """mlx-audio の依存が whispermlx と両立しないので、別環境の子プロセスで動かす。"""
    out = subprocess.run(
        ["uv", "run", "--script", str(SCRIPT_DIR / "diarize_nemotron.py"), str(wav), str(rttm)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return float(out.split()[-1])


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def annotation(segments: list[Segment]):
    from pyannote.core import Annotation
    from pyannote.core import Segment as Span

    ann = Annotation()
    for i, (start, end, label) in enumerate(segments):
        ann[Span(start, end), i] = label
    return ann


def align(ref: list[Segment], hyp: list[Segment]):
    """相互 DER の内訳と、話者を pyannote 側に対応付けた hyp と、その対応表を返す。"""
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_ann, hyp_ann = annotation(ref), annotation(hyp)
    metric = DiarizationErrorRate()
    detail = metric(ref_ann, hyp_ann, detailed=True)
    mapping = metric.optimal_mapping(ref_ann, hyp_ann)  # Nemotron の話者 → pyannote の話者
    mapped = [(start, end, mapping.get(label, f"N:{label}")) for start, end, label in hyp]
    return detail, mapped, mapping


def load_pair(out: Path) -> tuple[list[Segment], list[Segment]]:
    return read_rttm(out / "pyannote.rttm"), read_rttm(out / "nemotron.rttm")


def report(name: str, ref: list[Segment], hyp: list[Segment], timing: dict[str, float]) -> str:
    detail, mapped, mapping = align(ref, hyp)
    regions = sorted(disagreements(ref, mapped), key=lambda r: r[0] - r[1])

    total = detail["total"] or 1.0

    def pct(key: str) -> str:
        return f"{detail[key] / total:.1%}"

    lines = [
        f"# 話者分離の比較: {name}",
        "",
        "| | 話者数 | 発話時間 | 処理時間 |",
        "|---|---|---|---|",
    ]
    for label, segments in (("pyannote", ref), ("Nemotron", hyp)):
        speech = annotation(segments).get_timeline().support().duration()
        lines.append(
            f"| {label} | {len({s[2] for s in segments})} | {clock(speech)} | {timing.get(label.lower(), 0):.1f}秒 |"
        )
    lines += [
        "",
        f"**相互DER（pyannoteを基準）: {detail['diarization error rate']:.1%}**"
        f" — 話者の取り違え {pct('confusion')} / Nemotronだけ無音 {pct('missed detection')}"
        f" / Nemotronだけ発話 {pct('false alarm')}",
        "",
        "話者の対応（Nemotron → pyannote）: "
        + (", ".join(f"{k} → {v}" for k, v in sorted(mapping.items())) or "なし"),
        "",
        f"## 食い違った区間（1秒以上: {len(regions)}か所・計{clock(sum(e - s for s, e, *_ in regions))}、長い順に{TOP_N}件）",
        "",
        "Nemotron の話者名は pyannote 側に対応付け済み。`N:` 付きは対応先の無い話者。",
        "",
        "| 開始 | 終了 | 長さ | pyannote | Nemotron |",
        "|---|---|---|---|---|",
    ]
    for start, end, r_names, h_names in regions[:TOP_N]:
        lines.append(
            f"| {clock(start)} | {clock(end)} | {end - start:.1f}秒 | {', '.join(r_names) or '（無音）'} | {', '.join(h_names) or '（無音）'} |"
        )
    return "\n".join(lines) + "\n"


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit("使い方: uv run python compare_diarization.py <音声ファイル> [...]")
    for audio in map(Path, paths):
        out = SCRIPT_DIR / "output" / "compare" / audio.stem
        out.mkdir(parents=True, exist_ok=True)
        wav, ref_rttm, hyp_rttm = out / "audio.wav", out / "pyannote.rttm", out / "nemotron.rttm"
        timing_path = out / "timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

        if not wav.exists():
            print(f"▶ 16kHz wav に変換: {audio}", file=sys.stderr)
            to_wav(audio, wav)
        if not ref_rttm.exists():
            print("▶ pyannote で話者分離", file=sys.stderr)
            timing["pyannote"] = run_pyannote(wav, ref_rttm)
            timing_path.write_text(json.dumps(timing))
        if not hyp_rttm.exists():
            print("▶ Nemotron で話者分離", file=sys.stderr)
            timing["nemotron"] = run_nemotron(wav, hyp_rttm)
            timing_path.write_text(json.dumps(timing))

        text = report(audio.name, *load_pair(out), timing)
        (out / "report.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
