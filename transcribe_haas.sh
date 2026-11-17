#!/usr/bin/env bash
# Haas + Nemotron through speech.cpp. Run inside nix develop for ffmpeg and uv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIO="${1:?使い方: ./transcribe_haas.sh 音声ファイル [出力ディレクトリ]}"
OUTDIR="${2:-$SCRIPT_DIR/output/haas}"
SPEECH="${SPEECH_BIN:-$HOME/.local/bin/speech}"
[[ -f "$AUDIO" ]] || { echo "音声が見つかりません: $AUDIO" >&2; exit 1; }
[[ -x "$SPEECH" ]] || { echo "speech.cpp をインストールするか SPEECH_BIN を指定してください" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "nix develop 内で実行してください（ffmpeg が必要です）" >&2; exit 1; }
mkdir -p "$OUTDIR"
STEM="$(basename "${AUDIO%.*}")"
WORK="$(mktemp -d "$OUTDIR/.haas.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
if [[ "${WHISPERX_REUSE:-0}" == 1 ]]; then
    bash "$SCRIPT_DIR/enhance_reuse.sh" "$AUDIO" "$WORK/audio.wav"
else
    ffmpeg -nostdin -v error -i "$AUDIO" -ar 16000 -ac 1 -c:a pcm_s16le "$WORK/audio.wav"
fi
"$SPEECH" asr haas-ja-v1.0 --vad silero-vad --diarize nemotron-3-diarization --timestamps --format json "$WORK/audio.wav" > "$WORK/result.json"
uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/assign_nemotron.py" --speech "$WORK/result.json" "$AUDIO" "$OUTDIR"
mv "$WORK/result.json" "$OUTDIR/$STEM.speech.json"
echo "完了: $OUTDIR に出力しました（txt / srt / vtt / json / tsv、speech.cpp の原本は $STEM.speech.json）"
