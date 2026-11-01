#!/usr/bin/env bash
# SpeechLens CLI + NemotronLabs RE-USE MLX weights.
set -euo pipefail
AUDIO="${1:?入力音声を指定してください}"
OUTPUT="${2:?出力WAVを指定してください}"
REUSE_ROOT="${WHISPERX_REUSE_ROOT:-$HOME/.ai/whisperx-diarization/re-use}"
CLI="${WHISPERX_REUSE_CLI:-$REUSE_ROOT/cli/speechlens-cli/speechlens-cli}"
WEIGHTS="${WHISPERX_REUSE_WEIGHTS:-$REUSE_ROOT/model_mlx.safetensors}"
if [ ! -x "$CLI" ] || [ ! -f "$WEIGHTS" ]; then
	echo "エラー: RE-USE CLI / MLX重みがありません。WHISPERX_REUSE_CLI と WHISPERX_REUSE_WEIGHTS を確認してください。" >&2
	exit 1
fi
# 同梱ffmpegはAAC非対応のため、プロジェクト環境のffmpegを使用する。
command -v ffmpeg >/dev/null || { echo 'エラー: ffmpeg が必要です。nix develop 内で実行してください。' >&2; exit 1; }
TMPDIR_REUSE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_REUSE"' EXIT
# ASR向け16kHz mono。無音削除は行わず原音の時間軸を維持する。
ffmpeg -nostdin -v error -i "$AUDIO" -ac 1 -ar 16000 "$TMPDIR_REUSE/input.wav"
echo "▶ RE-USE 前処理: $AUDIO" >&2
"$CLI" --weights "$WEIGHTS" --input "$TMPDIR_REUSE/input.wav" --output "$OUTPUT" --chunk-seconds 5
