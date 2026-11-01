"""Run the same community-1 pipeline used by the comparison script."""
import sys
import tempfile
from pathlib import Path
from compare_diarization import run_pyannote, to_wav


def main():
    audio, rttm = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / 'audio.wav'
        to_wav(audio, wav)
        run_pyannote(wav, rttm)


if __name__ == '__main__':
    main()
