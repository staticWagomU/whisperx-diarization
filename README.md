# whisperx-diarization

whispermlx による Apple Silicon 向けの高精度・話者分離つき音声文字起こし環境（日本語向け）。

- **環境（ツール）** は Nix flake で固定（`uv` と `ffmpeg`）
- **ライブラリ** は `uv` が PyPI の macOS arm64 wheel から導入（`uv.lock` で固定）

この分担により、torch / pyannote.audio などをソースビルドせず、数分でセットアップできます。

## 何ができるか

- **文字起こし**: whispermlx / mlx-whisper（`large-v3`）で Apple Silicon GPU を使って書き起こし
- **単語レベルの整列**: wav2vec2 で各単語に正確なタイムスタンプを付与
- **話者分離**: Nemotron-3-Diarization（MLX）で「誰が話したか」を区別（`SPEAKER_00`, `SPEAKER_01` …）

## 前提

- Nix（flakes 有効）
- direnv（任意。使うと `cd` するだけで環境有効化）

## セットアップ

### 1. 環境（ツール）を有効化

```bash
direnv allow      # direnv を使う場合
# または
nix develop       # 使わない場合
```

これで `uv` と `ffmpeg` が使えるようになります。

### 2. ライブラリを導入

```bash
uv sync
```

`whispermlx` / `mlx-whisper` / `torch` / `pyannote-audio` などが wheel で入り、`.venv/` が作られます。

### 3. 比較スクリプト用の HuggingFace トークン（任意）

通常の文字起こしにトークンは不要です。後述の pyannote 比較を実行する場合だけ、
pyannote モデルの利用規約に同意してトークンを設定してください。

1. https://huggingface.co/settings/tokens で **Read** 権限のトークンを作成
2. 以下のモデルページで **利用規約に同意**（"Agree and access" をクリック）
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   > ※ 実行時に `Could not download ... you may need to accept the user conditions`
   > と出たら、表示された URL のモデルにも同意してください。
3. トークンを `.env` に保存（このファイルは `.gitignore` 済み）

```bash
echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxx' > .env
direnv reload   # または nix develop に入り直す
```

## 使い方

```bash
./transcribe.sh path/to/audio.m4a
# 出力は ./output/ に txt / srt / vtt / json / tsv で保存される
```

内部では whispermlx で文字起こし・整列を行い、Nemotron の RTTM を使って話者を付与します。
Nemotron は依存関係が whispermlx と競合するため `uv run --script diarize_nemotron.py` の別環境で実行します。
初回は Whisper モデル（`large-v3` は約 3GB）と整列・話者分離モデルのダウンロードに時間がかかります。

Nemotron は最大 8 人まで対応します。`WHISPERX_MIN_SPEAKERS` / `WHISPERX_MAX_SPEAKERS` は使えません。
同時発話は検出できますが、各単語・文への話者付与には重なり時間が最長の 1 人を採用します。
Issue #1 の比較では発話区間の検出に改善が見られた一方、話者の割り当ては未検証です。

### 環境変数で挙動を調整

| 変数 | 既定 | 説明 |
|------|------|------|
| `HF_TOKEN` | （未指定） | pyannote 比較用の HuggingFace トークン |
| `WHISPERX_MODEL` | `large-v3` | Whisper モデル。速度優先なら `turbo` |
| `WHISPERX_DEVICE` | `cpu` | VAD / 整列に使う PyTorch device。ASR と Nemotron は MLX を使う |
| `WHISPERX_COMPUTE` | `default` | 互換用。whispermlx の MLX ASR では実質無視される |
| `WHISPERX_VAD_METHOD` | `pyannote` | VAD。`silero` も指定可能 |
| `WHISPERX_ALIGN` | `1` | `0` で単語レベル整列を無効化して高速化 |
| `WHISPERX_DIARIZE` | `1` | `0` で Nemotron による話者分離を無効化 |

### 話者分離を切って高速化する

講演やセミナーのような独話で話者ラベルが不要なら、Nemotron を切れます。

```bash
WHISPERX_DIARIZE=0 ./transcribe.sh audio.m4a
```

単語レベルのタイムスタンプが不要なら、整列も切るとさらに速くなります。

```bash
WHISPERX_DIARIZE=0 WHISPERX_ALIGN=0 ./transcribe.sh audio.m4a
```

## まとめて文字起こしする

`input/NN_タイトル.m4a` の形式で連番を振ったファイルを一括処理します。

```bash
./transcribe_all.sh
```

- 出力済み（`output/` に `.txt` がある）はスキップするので、**中断しても再実行で続きから**
- 1本失敗しても止まらず、最後に成功/スキップ/失敗の件数をまとめて報告する
- ファイルごとの話者分離設定はスクリプト内の `case` で切り替える

## 話者分離モデルを比べる（pyannote / Nemotron）

同じ音声を pyannote と [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
（mlx-audio）にかけ、結果を比べます（issue #1）。

```bash
uv run python compare_diarization.py input/対談.m4a
```

- `output/compare/<音声名>/report.md` に、話者数・発話時間・処理時間、pyannote を基準にした相互 DER、
  1 秒以上食い違った区間の一覧（長い順）が出る。正解ラベルは無いので、一覧の時刻を実際に聞いて判断する
- Nemotron は `diarize_nemotron.py` が別環境（`uv run --script`）で動かす。mlx-audio の依存が
  whispermlx の固定版と両立しないため。初回は mlx-audio のビルドとモデルのダウンロードで数分かかる
- できあがった `.rttm` は再実行時にスキップされる。分離をやり直すときは該当ファイルを消す
- `disagreements` のテスト: `uv run python test_compare_diarization.py`

## macOS での注意

- 文字起こし本体は whispermlx / mlx-whisper により Apple Silicon GPU で実行されます。
- `--device cpu` は VAD / 整列の PyTorch 側の処理に対する既定です。ASR と Nemotron を CPU に固定する指定ではありません。
- 速度優先なら `WHISPERX_MODEL=turbo` を試してください。品質優先なら既定の `large-v3` のままにします。

### 既知の無害な警告

実行時に次のような警告が出ることがありますが、**無視して問題ありません**。

```
FFmpeg version 6: ... libtorchcodec_core6.dylib ... libavutil.58.dylib (no such file)
```

torchaudio が引き込む `torchcodec` が自前の ffmpeg 共有ライブラリを探して失敗する
ものですが、whispermlx の音声読み込みは ffmpeg をサブプロセスとして呼ぶ別経路を使うため、
文字起こしは正常に動作します。

## 構成ファイル

| ファイル | 役割 |
|----------|------|
| `flake.nix` | `uv` + `ffmpeg` を固定供給する devShell |
| `pyproject.toml` / `uv.lock` | whispermlx などライブラリのバージョン固定 |
| `transcribe.sh` / `assign_nemotron.py` | 日本語・Nemotron 話者分離つき文字起こし |
| `compare_diarization.py` / `diarize_nemotron.py` | pyannote と Nemotron の話者分離の比較 |
| `.envrc` | direnv 連携（flake 有効化 + `.env` 読み込み） |
