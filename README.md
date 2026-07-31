# whisperx-diarization

WhisperX による高精度・話者分離つき音声文字起こし環境（日本語向け）。

- **環境（ツール）** は Nix flake で固定（`uv` と `ffmpeg`）
- **ライブラリ** は `uv` が PyPI の macOS arm64 wheel から導入（`uv.lock` で固定）

この分担により、torch / onnxruntime などをソースビルドせず、数分でセットアップできます。

## 何ができるか

- **文字起こし**: faster-whisper（`large-v3`）で日本語を高精度に書き起こし
- **単語レベルの整列**: wav2vec2 で各単語に正確なタイムスタンプを付与
- **話者分離**: pyannote.audio で「誰が話したか」を区別（`SPEAKER_00`, `SPEAKER_01` …）

## 前提

- Nix（flakes 有効）
- direnv（任意。使うと `cd` するだけで環境有効化）
- **HuggingFace トークン**（話者分離に必須。下記参照）

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

`whisperx` / `torch` / `pyannote-audio` などが wheel で入り、`.venv/` が作られます。

### 3. HuggingFace トークンの用意（話者分離に必須）

pyannote の話者分離モデルは利用規約への同意が必要です。

1. https://huggingface.co/settings/tokens で **Read** 権限のトークンを作成
2. 以下のモデルページで **利用規約に同意**（"Agree and access" をクリック）
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   > ※ 実行時に `Could not download ... you may need to accept the user conditions`
   > と出たら、表示された URL のモデルにも同意してください。
3. トークンを `.env` に保存（このファイルは `.gitignore` 済み）

```bash
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxx' > .env
direnv reload   # または nix develop に入り直す
```

## 使い方

```bash
./transcribe.sh path/to/audio.m4a
# 出力は ./output/ に txt / srt / vtt / json / tsv で保存される
```

内部では `uv run whisperx ... --diarize` を実行しています。初回は Whisper モデル
（`large-v3` は約 3GB）と整列・話者分離モデルをダウンロードするため時間がかかります。

### 話者数が分かっている場合（精度が上がる）

```bash
WHISPERX_MIN_SPEAKERS=2 WHISPERX_MAX_SPEAKERS=2 ./transcribe.sh audio.m4a
```

### 環境変数で挙動を調整

| 変数 | 既定 | 説明 |
|------|------|------|
| `HF_TOKEN` | （必須） | HuggingFace トークン |
| `WHISPERX_MODEL` | `large-v3` | Whisper モデル。速度優先なら `medium` |
| `WHISPERX_COMPUTE` | `int8` | 計算精度。`int8` は速い / `float32` は高精度 |
| `WHISPERX_MIN_SPEAKERS` | （未指定） | 最小話者数 |
| `WHISPERX_MAX_SPEAKERS` | （未指定） | 最大話者数 |

## macOS での注意

- WhisperX の文字起こしは CTranslate2 が Metal 非対応のため **CPU 実行**です
  （M4 Pro なら実用速度。目安: 音声長の 0.3〜0.8 倍程度）。
- そのため `--device cpu` を既定にしています。

### 既知の無害な警告

実行時に次のような警告が出ることがありますが、**無視して問題ありません**。

```
FFmpeg version 6: ... libtorchcodec_core6.dylib ... libavutil.58.dylib (no such file)
```

torchaudio が引き込む `torchcodec` が自前の ffmpeg 共有ライブラリを探して失敗する
ものですが、WhisperX の音声読み込みは ffmpeg をサブプロセスとして呼ぶ別経路を使うため、
文字起こしは正常に動作します。

## 構成ファイル

| ファイル | 役割 |
|----------|------|
| `flake.nix` | `uv` + `ffmpeg` を固定供給する devShell |
| `pyproject.toml` / `uv.lock` | whisperx などライブラリのバージョン固定 |
| `transcribe.sh` | 日本語・話者分離つき文字起こしラッパー |
| `.envrc` | direnv 連携（flake 有効化 + `.env` 読み込み） |
