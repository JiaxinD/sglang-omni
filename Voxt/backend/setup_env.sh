#!/usr/bin/env bash
# Creates the pinned Python environment the Voxt Omni backend runs in.
#
#   setup_env.sh <environment-directory>
#
# Installs SGLang v0.5.21 from source and this checkout's sglang-omni into a
# Python 3.12 virtual environment, then pins every package to
# requirements-mac.lock. Needs uv, Homebrew ffmpeg@7 and a Rust toolchain
# (SGLang builds a Rust extension). Nothing outside the target directory changes.
set -euo pipefail

target="${1:?usage: setup_env.sh <environment-directory>}"
backend_dir="$(cd "$(dirname "$0")" && pwd)"
omni_dir="$(cd "$backend_dir/../.." && pwd)"
sglang_commit=e00930c5489053f26d86b179cee0d087f846acbb

test "$(uname -m)" = arm64 || { echo "Apple Silicon is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
command -v cargo >/dev/null || { echo "a Rust toolchain is required (brew install rust)" >&2; exit 1; }
brew list ffmpeg@7 >/dev/null 2>&1 || { echo "brew install ffmpeg@7 first" >&2; exit 1; }

mkdir -p "$target"
if [[ ! -d "$target/sglang" ]]; then
  git clone -q --branch v0.5.21 --depth 1 https://github.com/sgl-project/sglang.git "$target/sglang"
fi
test "$(git -C "$target/sglang" rev-parse HEAD)" = "$sglang_commit" || {
  echo "unexpected SGLang revision in $target/sglang" >&2; exit 1; }
cp "$target/sglang/python/pyproject_other.toml" "$target/sglang/python/pyproject.toml"

uv venv -p 3.12 "$target/venv"
export VIRTUAL_ENV="$target/venv"
uv pip install -e "$target/sglang/python[all_mps]"
uv pip install -e "$omni_dir"
uv pip install -r "$backend_dir/requirements-mac.lock"

SGLANG_USE_MLX=1 DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib" "$target/venv/bin/python" - <<'PY'
import mlx.core as mx
from torchcodec.decoders import AudioDecoder
import sglang_omni.models.moss_transcribe_diarize.mlx.runner
import sglang_omni.models.whisper_asr.mlx.runner
assert mx.metal.is_available()
print("environment ready")
PY
echo "VOXT_OMNI_PYTHON=$target/venv/bin/python"
