{
  description = "High-accuracy speaker-diarized transcription with whispermlx (Japanese-focused)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        # flake は「ツール」を固定供給する役割に徹する:
        #   - uv        : Python 本体と whispermlx などの wheel を管理（再現性は uv.lock）
        #   - ffmpeg    : 音声デコードに必須
        # whispermlx 本体は PyPI の macOS arm64 wheel から入るため、
        # torch / pyannote.audio などをソースビルドせずに済む。
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.ffmpeg
          ];

          # uv がダウンロードする CPython を使う（Nix Python と wheel の ABI 齟齬を避ける）。
          env = {
            UV_PYTHON_PREFERENCE = "only-managed";
          };

          shellHook = ''
            echo "── whispermlx 話者分離 文字起こし環境 (uv) ──"
            echo "uv     : $(uv --version 2>/dev/null)"
            echo "ffmpeg : $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
            echo ""
            echo "初回セットアップ:  uv sync"
            echo "実行:            ./transcribe.sh <音声ファイル>"
          '';
        };
      }
    );
}
