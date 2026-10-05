# Voxt on the local SGLang-Omni server

This directory runs Voxt's local speech models on a local SGLang-Omni server
with native MLX on Apple Silicon. Voxt routes only Qwen3-ASR 0.6B 4-bit to it
today; every other model keeps Voxt's original Swift backend.

| Checkpoint | Omni backend | Voxt routing | Voxt behavior kept |
| --- | --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | Qwen3-ASR MLX runner (existing) | On | Final with context bias and language hint, 1200 s energy-cut chunks sharing one token budget, first detected language carried forward; live preview over the realtime transcription socket |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | MOSS MLX runner (new), bf16 checkpoint as installed | Off: server-tested, not yet accepted in the app | Timestamped diarization or plain text by prompt, chunk timestamps on the recording timeline, speaker segments; live preview with the original 4 s window schedule |
| `mlx-community/whisper-large-v3-turbo` | Whisper MLX encoder-decoder runner (new) | Off: server-tested, not yet accepted in the app | Independent 30 s windows, no language token without a hint, generation-config suppression with timestamps masked; batch preview unchanged |

To route MOSS or Whisper as well, add its repository to
`OmniASRBackend.modelKindsByRepo` in `Voxt/Transcription/OmniASRBackend.swift`.

Not migrated, and still on the Swift backend: the other Qwen3-ASR and Whisper
variants, Cohere, Parakeet, Nemotron, SenseVoice, speaker analysis, VAD and the
local LLMs.

## Set up

Requirements: an Apple Silicon Mac, Xcode, [uv](https://docs.astral.sh/uv/),
Homebrew `ffmpeg@7` and a Rust toolchain (`brew install rust`; SGLang builds a
Rust extension).

```bash
Voxt/backend/setup_env.sh ~/voxt-omni-env
```

The script prints the `VOXT_OMNI_PYTHON` to use. It installs SGLang v0.5.21
from its pinned commit and this checkout's sglang-omni, then pins every package
to `requirements-mac.lock`.

## Build and run

```bash
Voxt/backend/run_omni_dev.sh build
VOXT_OMNI_PYTHON=~/voxt-omni-env/venv/bin/python Voxt/backend/run_omni_dev.sh run
```

"Voxt Omni Dev" has its own bundle identifier, runs without the sandbox so it
can start the backend, and sees `~/.voxt-omni-dev` as its home, so its database,
history, preferences and models stay apart from any installed Voxt. Set
`VOXT_SHARED_MODELS` to an existing `<root>/mlx-audio` directory to reuse
downloaded weights. `run_omni_dev.sh run --swift-backend` runs the same build on
the original Swift backend for comparison.

Download the models from Voxt's model settings as usual. With the Omni backend
enabled, selecting one of the three checkpoints starts a server for it on a free
loopback port; switching models, idle unload, deletion and quitting stop it.
After the models are cached, dictation needs no network.

## How it fits together

- `voxt_omni_backend/supervisor.py` owns one `sgl-omni serve` process per
  loaded model. It reports `ready` only after the server answers with the
  unique model name it was started with, and stops the server and every process
  it started when Voxt sends `shutdown`, when Voxt's control pipe closes (Voxt
  quit or crashed) or on a termination signal. It never signals other processes.
- `voxt_omni_backend/model_views.py` gives the server an HF-format view of
  Voxt's Whisper directory (config and feature-extractor files, plus links to the
  installed weights and tokenizer files).
- `Voxt/Transcription/Omni*.swift` is the client: `OmniASRRuntime` (launch,
  requests, an awaitable retire that drains in-flight work), the per-model
  request planning, the live sessions and their adapter to Voxt's streaming
  session interface.

## Tests

```bash
cd Voxt/backend && "$VOXT_OMNI_PYTHON" -m pytest tests
cd ../.. && "$VOXT_OMNI_PYTHON" -m pytest tests/unit_test/moss_transcribe_diarize tests/unit_test/whisper_asr tests/unit_test/qwen3_asr
```

## Known limitations

- Greedy decoding only on the Omni path. Whisper with a non-zero temperature
  setting fails with a clear error instead of sampling.
- The dev build is ad hoc signed without keychain access groups, so remote
  provider API keys may not persist in it.
- The server accepts requests from any local client on its loopback port and
  sends permissive CORS headers; it holds no user data beyond in-flight audio.
- Performance and quality acceptance against the original backend is pending;
  see the project's acceptance records before relying on any speed claim.
