"""Keyboard-driven transcription launcher; no model imports until execution."""
import argparse
import curses
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
FIELDS = [
    ('文字起こしモデル', 'WHISPERX_MODEL', ['large-v3', 'turbo'], 'large-v3: 標準 / turbo: 速度を優先する候補'),
    ('話者分離', 'WHISPERX_DIARIZE', ['1', '0'], '有効にすると SPEAKER_00 などの話者ラベルを付けます。'),
    ('話者分離モデル', 'WHISPERX_DIARIZATION_MODEL', ['nemotron', 'pyannote'], 'Nemotron: 最大8人 / pyannote community-1: HF_TOKENと利用同意が必要'),
    ('単語ごとの時刻', 'WHISPERX_ALIGN', ['1', '0'], '音声と単語の位置を合わせます。省くと処理を短縮できます。'),
    ('言語', 'WHISPERX_LANGUAGE', ['ja', 'en', 'zh', 'ko'], '音声の言語を選んでください。ja=日本語 en=英語 zh=中国語 ko=韓国語'),
    ('VAD（発話検出）', 'WHISPERX_VAD_METHOD', ['pyannote', 'silero'], '音声から発話区間を検出する方法です。'),
    ('整列用デバイス', 'WHISPERX_DEVICE', ['cpu', 'mps'], '文字起こしとNemotronは、この指定に関わらずMLXで動きます。'),
]


def settings():
    return {key: os.environ.get(key, choices[0]) for _, key, choices, _ in FIELDS}


def launch(audio, out, values):
    env = dict(os.environ, **values)
    out.mkdir(parents=True, exist_ok=False)
    log = open(out / 'run.log', 'w')
    try:
        process = subprocess.Popen(['bash', str(ROOT / 'transcribe.sh'), str(audio), str(out)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return process, log


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def ui(screen, audio, out):
    curses.curs_set(0)
    screen.timeout(150)
    values = settings()
    index = 0
    choosing = False
    candidate = 0
    page = 'settings'
    message = ''
    process = log = None
    cancelled = False
    try:
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            def put(y, text, selected=False):
                if 0 <= y < height - 1:
                    try:
                        screen.addnstr(y, 1, text, max(0, width - 3), curses.A_REVERSE if selected else 0)
                    except curses.error:
                        pass
            put(0, '音声文字起こし')
            put(2, f'入力: {audio}')
            if page in ('settings', 'confirm'):
                for i, (label, key, choices, help_text) in enumerate(FIELDS):
                    value = values[key]
                    if key in ('WHISPERX_DIARIZE', 'WHISPERX_ALIGN'):
                        value = '有効' if value != '0' else '無効'
                    if key == 'WHISPERX_DIARIZATION_MODEL' and values['WHISPERX_DIARIZE'] == '0':
                        value = '—（話者分離が無効）'
                    put(4+i, f'{label}: {value}', page == 'settings' and i == index)
                put(12, f'出力先: {out}')
                if page == 'confirm':
                    put(14, '初回はモデルのダウンロードが必要な場合があります。')
                    put(16, '[ Enterで実行する / Escで設定に戻る ]')
                elif choosing:
                    label, key, choices, help_text = FIELDS[index]
                    put(14, label + 'を選択')
                    for n, value in enumerate(choices):
                        put(15+n, value, n == candidate)
                    put(21, 'Space 確定 / Esc 取消')
                else:
                    put(14, FIELDS[index][3])
                    put(17, 'j/k 移動  Space 設定変更  Enter 確認へ  Esc 終了')
            else:
                code = process.poll()
                if code is None:
                    put(4, '処理中…（進捗の詳細はログに表示されます）')
                    put(6, 'Esc 停止の確認')
                else:
                    put(4, '停止しました' if cancelled else ('完了' if code == 0 else f'失敗（終了コード {code}）'))
                    put(6, 'Enter / Esc 終了')
                put(8, f'保存先: {out}')
                put(9, f'ログ: {out / "run.log"}')
                if page == 'stop':
                    put(11, '停止しますか？ Enter 停止 / Esc 処理に戻る')
                try:
                    tail = (out / 'run.log').read_text(errors='replace').splitlines()[-max(1, height-15):]
                    for n, line in enumerate(tail):
                        put(13+n, line)
                except OSError:
                    pass
            put(height-2, message)
            screen.refresh()
            ch = screen.getch()
            if ch == -1:
                continue
            message = ''
            if page in ('running', 'stop'):
                if process.poll() is not None and ch in (10, 13, 27):
                    return
                if page == 'stop' and ch in (10, 13):
                    stop(process)
                    cancelled = True
                    page = 'running'
                elif ch == 27:
                    page = 'running' if page == 'stop' else 'stop'
                continue
            if page == 'confirm':
                if ch == 27:
                    page = 'settings'
                elif ch in (10, 13):
                    if values['WHISPERX_DIARIZE'] != '0' and values['WHISPERX_DIARIZATION_MODEL'] == 'pyannote' and not os.environ.get('HF_TOKEN'):
                        message = 'HF_TOKENが未設定です。設定してから起動してください。'
                        continue
                    try:
                        process, log = launch(audio, out, values)
                        page = 'running'
                    except OSError as exc:
                        message = str(exc)
                continue
            label, key, choices, _ = FIELDS[index]
            if choosing:
                if ch in (ord('j'), ord('k')):
                    candidate = (candidate + (1 if ch == ord('j') else -1)) % len(choices)
                elif ch == 32:
                    values[key] = choices[candidate]
                    choosing = False
                elif ch == 27:
                    choosing = False
            elif ch in (ord('j'), ord('k')):
                index = (index + (1 if ch == ord('j') else -1)) % len(FIELDS)
            elif ch == 32:
                if key == 'WHISPERX_DIARIZATION_MODEL' and values['WHISPERX_DIARIZE'] == '0':
                    continue
                candidate = choices.index(values[key]) if values[key] in choices else 0
                if len(choices) == 2:
                    values[key] = choices[(candidate + 1) % 2]
                else:
                    choosing = True
            elif ch in (10, 13):
                page = 'confirm'
            elif ch == 27:
                return
    finally:
        if process is not None and process.poll() is None:
            stop(process)
        if log:
            log.close()


def main():
    parser = argparse.ArgumentParser(description='j/k・Spaceで設定、Enterで実行する文字起こしTUI')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--output', type=Path, help='新しく作成する出力ディレクトリ')
    args = parser.parse_args()
    audio = args.audio.expanduser().resolve()
    if not audio.is_file():
        parser.error(f'ファイルが見つかりません: {audio}')
    out = (args.output or ROOT / 'output' / f'{audio.stem}-{datetime.datetime.now():%Y%m%d-%H%M%S-%f}').expanduser().resolve()
    if out.exists():
        parser.error('出力先は既に存在します。新しいディレクトリを指定してください。')
    curses.wrapper(ui, audio, out)


if __name__ == '__main__':
    main()
