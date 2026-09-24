#!/usr/bin/env bash
# 連番プレフィックス付きの音声（NN_タイトル.m4a）をまとめて文字起こしする。
#   使い方: ./transcribe_all.sh
#
# 数時間かかるバッチなので、次の2点を意図的に守っている:
#   - 出力済み（.txt がある）はスキップする → 中断しても再実行で続きから
#   - 1本失敗しても止めない            → 残りを最後まで回して後でまとめて確認する
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 比較スクリプトなどで使う .env を、必要なら環境変数として読み込む。
if [ -f .env ]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

OUTDIR="$SCRIPT_DIR/output"
mkdir -p "$OUTDIR"

shopt -s nullglob
files=(input/[0-9][0-9]_*.m4a)
shopt -u nullglob

total=${#files[@]}
if [ "$total" -eq 0 ]; then
	echo "対象ファイルが見つかりません（input/NN_*.m4a）" >&2
	exit 1
fi

failed=()
skipped=()
batch_start=$(date +%s)

echo "===== 一括文字起こし開始: ${total}本 ====="

i=0
for audio in "${files[@]}"; do
	i=$((i + 1))
	base="$(basename "$audio" .m4a)"

	if [ -f "$OUTDIR/$base.txt" ]; then
		echo "[$i/$total] スキップ（出力済み）: $base"
		skipped+=("$base")
		continue
	fi

	# 講演は独話なのでラベルの価値が薄く、既定では話者分離を切る。
	# 11 は複数人の対談（登壇者2名＋司会）なので話者分離を有効にする。
	#
	# 言語は必ず音声に合わせる。日本語を指定したまま英語音声を流すと、
	# 壊れた翻訳と反復ループが混ざった使い物にならない出力になる。
	# 03 / 09 / 12 は英語セッション（冒頭の司会紹介だけが日本語）。
	case "$base" in
	11_*)
		file_env=(WHISPERX_LANGUAGE=ja WHISPERX_DIARIZE=1)
		;;
	03_* | 09_* | 12_*)
		file_env=(WHISPERX_LANGUAGE=en WHISPERX_DIARIZE=0)
		;;
	*)
		file_env=(WHISPERX_LANGUAGE=ja WHISPERX_DIARIZE=0)
		;;
	esac

	echo
	echo "----- [$i/$total] $base (${file_env[*]}) -----"
	start=$(date +%s)

	if env "${file_env[@]}" ./transcribe.sh "$audio" "$OUTDIR"; then
		echo "[$i/$total] 完了 ($((($(date +%s) - start) / 60))分): $base"
	else
		echo "[$i/$total] 失敗: $base" >&2
		failed+=("$base")
	fi
done

echo
echo "===== 全体完了: $((($(date +%s) - batch_start) / 60))分 ====="
echo "成功 $((total - ${#failed[@]} - ${#skipped[@]}) ) / スキップ ${#skipped[@]} / 失敗 ${#failed[@]}"
for f in "${failed[@]}"; do echo "  失敗: $f"; done

[ ${#failed[@]} -eq 0 ]
