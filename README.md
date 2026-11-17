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

### Haas + Nemotron（speech.cpp / Metal）

speech.cpp をインストールした環境で、次を実行します。

```bash
nix develop -c ./transcribe_haas.sh path/to/audio.m4a
# RE-USE 前処理を使う場合
WHISPERX_REUSE=1 nix develop -c ./transcribe_haas.sh path/to/audio.m4a
```

`haas-ja-v1.0` + `silero-vad` + `nemotron-3-diarization` を使い、
`output/haas/` に whispermlx 経路と同じ txt / srt / vtt / json / tsv を保存します。
モデルは初回に取得します。`SPEECH_BIN` で実行ファイルを指定できます
（既定は `~/.local/bin/speech`）。第2引数で出力先を指定できます。
Haas の文字ごとの時刻に Nemotron の話者区間を重ね、話者が変わる位置で行を区切ります
（区切り方は後述の whispermlx 経路と同じです）。
speech.cpp の出力そのものは `<音声名>.speech.json` に残します。
その `regions` には発話区間ごとの本文と話者の時間範囲が入り、区間内に複数人が話す場合があります。
RE-USE 有効時は前処理した音声を Haas と Nemotron の両方に渡します。
重複発話の副話者が抑えられる可能性があるため、既定では無効です。
RE-USE の導入・重みの指定方法は下記の前処理の節を参照してください。

```bash
./transcribe.sh path/to/audio.m4a
# 出力は ./output/ に txt / srt / vtt / json / tsv で保存される
```

内部では whispermlx で文字起こし・整列を行い、Nemotron の RTTM を使って話者を付与します。
Nemotron は依存関係が whispermlx と競合するため `uv run --script diarize_nemotron.py` の別環境で実行します。
初回は Whisper モデル（`large-v3` は約 3GB）と整列・話者分離モデルのダウンロードに時間がかかります。

Nemotron は最大 8 人まで対応します。`WHISPERX_MIN_SPEAKERS` / `WHISPERX_MAX_SPEAKERS` は使えません。
同時発話は検出できますが、各単語・文への話者付与には重なり時間が最長の 1 人を採用します。
出力の各行は Whisper の文区切りではなく、話者が変わる位置で区切り直します。日本語は文字単位で話者がちらつくため、
8 単語（日本語では 8 文字）未満の短い区間は直前の話者に含めます。このため短い相づちは独立した行になりません。
Issue #1 の比較では発話区間の検出に改善が見られた一方、話者の割り当ては未検証です。

### Web で使う

文字起こし（複数の録音をつなぐ・RE-USE・Haas / whispermlx）、話者と人の結び付け、声の登録、RE-USE だけの処理をブラウザから行えます。

```bash
nix develop -c uv run python server.py          # http://127.0.0.1:8765/ を開く
# 既存の文字起こし（匿名ラベルの .json）を取り込む
nix develop -c uv run python server.py import "2026-11-16 定例会議" input/meeting.WAV output/haas/meeting.json
```

話者は名前ではなく人の id で人と結び付けます。人の表示名を直すと、結び付いたすべての文字起こしとダウンロードに反映されます。
声で自動判定した結び付けを「違う」で外すと、その人を `rejected` として記録します。
文字起こしは `WHISPERX_TRANSCRIPTS`（既定 `~/.ai/whisperx-diarization/transcripts`）に、アップロードはその隣の `uploads` に保存します。
127.0.0.1 でだけ待ち受け、認証はありません。校正版（.md）の生成はまだ Web からはできません。

### 声を登録して話者名を付ける

声を登録しておくと、出力の `SPEAKER_00` などを登録した名前に置き換えます（whispermlx・Haas の両経路）。
過去の文字起こしの話者を指定して登録するか、本人だけが話す音声をまるごと渡します。

```bash
nix develop -c uv run python voices.py enroll 林 input/meeting.WAV output/haas/meeting.json SPEAKER_00
nix develop -c uv run python voices.py enroll 林 input/my-voice.m4a
nix develop -c uv run python voices.py list
```

話者ごとに 1.5 秒以上の発話の埋め込み（pyannote の WeSpeaker）を平均し、登録した声とのコサイン類似度が
閾値以上の組を、類似度の高い順に 1 対 1 で割り当てます。閾値に届かない話者は匿名のまま残ります。
同じ名前で登録を重ねると録音ごとに追加し、照合ではどの録音も同じ重みで平均します。
マイクや部屋が違う会議から登録を重ねるほど照合が安定します。録音を外す・声の登録を消すのは Web の「人と声」から行えます。
保存先には `people.json`（人の id・表示名・登録に使った録音）と `<id>.npy`（録音 1 本につき 1 行）を置きます。以前の `<名前>.npy` は初回の読み込みで移行します。
照合の結果は標準エラーに類似度つきで表示します。話者分離が 2 人を 1 人にまとめた場合、名前は付いても分離は直りません。
声紋は個人情報なので、リポジトリの外（既定 `~/.ai/whisperx-diarization/voices/`）に保存します。
先方の声を登録する場合は同意を得てください。

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
| `WHISPERX_VOICES` | `~/.ai/whisperx-diarization/voices` | 登録した声の保存先 |
| `WHISPERX_VOICE_THRESHOLD` | `0.6` | 声の照合で名前を付ける類似度の下限 |
| `WHISPERX_TRANSCRIPTS` | `~/.ai/whisperx-diarization/transcripts` | Web 版の文字起こしの保存先 |

### RE-USEを任意の前処理として使う（Mac）

```bash
WHISPERX_REUSE=1 ./transcribe.sh input/meeting.m4a
```

既定は無効です。有効時はNemotronLabs RE-USEで処理した音声を文字起こし・整列に使い、
話者分離には原音を使います。同時発話では小さい声や相づちを抑える可能性があるため、
録音に応じて有効にしてください。原音は変更しません。処理用WAVは終了時に削除します。
出力名は元音声の名前を維持します。前処理に失敗した場合はそこで停止します。

[SpeechLens CLI](https://github.com/faraday/SpeechLens/releases) と、対応する形式に変換した
[NemotronLabs RE-USE公式重み](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) が必要です。
CLIは同梱の `mlx.metallib`・ffmpeg・ffprobeと同じディレクトリに置いてください。
入力のデコードにはNix環境のffmpegを使い、16kHz mono WAVへ変換します。

このマシンでは検証済みファイルを `~/.ai/whisperx-diarization/re-use/` に保存しており、
そのまま上記コマンドで使用できます。他の場所に配置する場合は指定してください。

```bash
export WHISPERX_REUSE_CLI=/absolute/path/to/speechlens-cli
export WHISPERX_REUSE_WEIGHTS=/absolute/path/to/model_mlx.safetensors
```

既定のCLIは `~/.ai/whisperx-diarization/re-use/cli/speechlens-cli/speechlens-cli`、
重みは `~/.ai/whisperx-diarization/re-use/model_mlx.safetensors` です。
`WHISPERX_REUSE_ROOT` で共通の保存先も変更できます。
SpeechLens配布の旧RE-USE変換済み重みと、新しいNemotronLabs版ではライセンスが異なるため、
使用する重みのモデルカードを確認してください。

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
| `voices.py` | 人と声の登録・照合 |
| `server.py` / `web/index.html` | Web 版のサーバーと画面 |
| `.envrc` | direnv 連携（flake 有効化 + `.env` 読み込み） |

## インタラクティブに選ぶ

```bash
./tui.sh "path/to/audio.m4a"
# 保存先を指定する場合（まだ存在しないディレクトリ）
./tui.sh "path/to/audio.m4a" --output ./output/new-session
```

Python 3 の curses を使うターミナル画面です。`j` / `k` で項目を移動し、
2択の設定は `Space` でその場で切り替えます。3択以上は `Space` で選択肢を開き、
`j` / `k` → `Space` で変更を確定します。
`Enter` は実行確認へ進む・実行する操作、`Esc` は取消・戻る・終了です。
実行中の `Esc` は停止確認を開き、`Enter` で子プロセスを含め停止します。

文字起こしモデル、話者分離の有無とモデル（Nemotron / pyannote community-1）、
単語時刻の整列、言語、VAD、整列用deviceを選べます。既存の環境変数指定を初期値に使います。
pyannoteには `HF_TOKEN` とモデル利用条件への同意が必要です。
出力は実行ごとの新規ディレクトリに保存し、`run.log` に処理ログを残します。
モデル比較は従来の `compare_diarization.py` を使用してください。
