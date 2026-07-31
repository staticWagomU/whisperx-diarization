#!/usr/bin/env bash
# 日本語音声を話者分離つきで文字起こしする WhisperX ラッパー。
#   使い方: ./transcribe.sh <音声ファイル> [出力ディレクトリ]
#   例:     ./transcribe.sh meeting.m4a
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AUDIO="${1:?音声ファイルを指定してください（例: ./transcribe.sh audio.m4a）}"
OUTDIR="${2:-$SCRIPT_DIR/output}"

MODEL="${WHISPERX_MODEL:-large-v3}"
COMPUTE="${WHISPERX_COMPUTE:-int8}"

if [ ! -f "$AUDIO" ]; then
	echo "エラー: ファイルが見つかりません: $AUDIO" >&2
	exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
	echo "エラー: HF_TOKEN が未設定です（話者分離に必要）。" >&2
	echo "  .env に  export HF_TOKEN=hf_xxx  を書いて direnv reload、または export してください。" >&2
	exit 1
fi

# uv がプロジェクトの pyproject.toml / venv を見つけられるよう、
# ユーザー指定の相対パスは先に絶対パス化してから移動する。
AUDIO="$(cd "$(dirname "$AUDIO")" && pwd)/$(basename "$AUDIO")"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
cd "$SCRIPT_DIR"

# 任意指定の話者数（対談なら 2 など）。未指定なら渡さない。
speaker_args=()
[ -n "${WHISPERX_MIN_SPEAKERS:-}" ] && speaker_args+=(--min_speakers "$WHISPERX_MIN_SPEAKERS")
[ -n "${WHISPERX_MAX_SPEAKERS:-}" ] && speaker_args+=(--max_speakers "$WHISPERX_MAX_SPEAKERS")

echo "▶ 文字起こし開始: $AUDIO  (model=$MODEL, compute=$COMPUTE)"

# uv run で venv を有効化して whisperx を実行（wheel でインストール済み）。
uv run whisperx "$AUDIO" \
	--model "$MODEL" \
	--language ja \
	--device cpu \
	--compute_type "$COMPUTE" \
	--diarize \
	--hf_token "$HF_TOKEN" \
	--output_dir "$OUTDIR" \
	--output_format all \
	--print_progress True \
	"${speaker_args[@]}"

echo "✔ 完了: $OUTDIR に出力しました（txt / srt / vtt / json / tsv）"
