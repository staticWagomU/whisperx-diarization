#!/usr/bin/env bash
# 日本語音声を Nemotron 話者分離つきで文字起こしするラッパー。
#   使い方: ./transcribe.sh <音声ファイル> [出力ディレクトリ]
#   例:     ./transcribe.sh meeting.m4a
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AUDIO="${1:?音声ファイルを指定してください（例: ./transcribe.sh audio.m4a）}"
OUTDIR="${2:-$SCRIPT_DIR/output}"

MODEL="${WHISPERX_MODEL:-large-v3}"
COMPUTE="${WHISPERX_COMPUTE:-default}"
DIARIZE="${WHISPERX_DIARIZE:-1}"
DEVICE="${WHISPERX_DEVICE:-cpu}"
ALIGN="${WHISPERX_ALIGN:-1}"
VAD_METHOD="${WHISPERX_VAD_METHOD:-pyannote}"
# 言語を誤ると Whisper は「翻訳もどき」を出力し、同じ語句を延々と繰り返す
# 反復ループに陥る。音声の言語は必ず合わせること。
LANGUAGE="${WHISPERX_LANGUAGE:-ja}"

if [ ! -f "$AUDIO" ]; then
	echo "エラー: ファイルが見つかりません: $AUDIO" >&2
	exit 1
fi

if [ "$DIARIZE" != "0" ] && [ -n "${WHISPERX_MIN_SPEAKERS:-}${WHISPERX_MAX_SPEAKERS:-}" ]; then
	echo "エラー: Nemotron は WHISPERX_MIN_SPEAKERS / WHISPERX_MAX_SPEAKERS に対応していません。" >&2
	exit 1
fi

# uv がプロジェクトの pyproject.toml / venv を見つけられるよう、
# ユーザー指定の相対パスは先に絶対パス化してから移動する。
AUDIO="$(cd "$(dirname "$AUDIO")" && pwd)/$(basename "$AUDIO")"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
cd "$SCRIPT_DIR"

ASR_OUTDIR="$OUTDIR"
if [ "$DIARIZE" != "0" ]; then
	ASR_OUTDIR="$(mktemp -d "$OUTDIR/.nemotron.XXXXXX")"
	trap 'rm -rf "$ASR_OUTDIR"' EXIT
fi

echo "▶ 文字起こし開始: $AUDIO  (engine=whispermlx, model=$MODEL, compute=$COMPUTE, device=$DEVICE, vad=$VAD_METHOD, lang=$LANGUAGE, align=$ALIGN, diarize=$DIARIZE)"

# uv run で venv を有効化して whispermlx を実行（ASR は MLX で Apple Silicon GPU を使う）。
cmd=(
	uv run whispermlx "$AUDIO"
	--model "$MODEL"
	--language "$LANGUAGE"
	--device "$DEVICE"
	--compute_type "$COMPUTE"
	--vad_method "$VAD_METHOD"
	--output_dir "$ASR_OUTDIR"
	--output_format all
	--print_progress True
)

if [ "$ALIGN" = "0" ]; then
	cmd+=(--no_align)
fi

"${cmd[@]}"

if [ "$DIARIZE" != "0" ]; then
	uv run --script diarize_nemotron.py "$AUDIO" "$ASR_OUTDIR/diarization.rttm"
	uv run python assign_nemotron.py "$ASR_OUTDIR/$(basename "${AUDIO%.*}").json" "$ASR_OUTDIR/diarization.rttm" "$AUDIO" "$OUTDIR"
fi

echo "✔ 完了: $OUTDIR に出力しました（txt / srt / vtt / json / tsv）"
