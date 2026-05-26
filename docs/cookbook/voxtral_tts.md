# Voxtral TTS

[Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) is an open-weights
text-to-speech model from Mistral AI built on a Ministral-3B backbone. It generates lifelike
24 kHz speech with natural prosody across 9 languages and ships with a set of preset named
voices. In SGLang-Omni, Voxtral runs as a `preprocessing → tts_generation → vocoder` pipeline
and is served through the OpenAI-compatible `/v1/audio/speech` endpoint.

> Voxtral-4B-TTS is released under **CC BY-NC 4.0** (non-commercial use only).

## Prerequisites

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v .

# Voxtral preprocessing uses Mistral's Tekken tokenizer from mistral-common.
uv pip install 'mistral-common[audio]>=1.8.0'

hf download mistralai/Voxtral-4B-TTS-2603
```

The model repository is public, so no Hugging Face token is required.

## Server Configuration

The pipeline is `preprocessing → tts_generation → vocoder`.

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts.yaml \
  --port 8000
```

## Synthesizing Speech

### Zero-shot

With no voice specified, Voxtral falls back to its default voice (`cheerful_female`).

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

### Named Voices

Voxtral speaks with **preset named voices** (it does not clone from a reference clip). Select
one with the `voice` field:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "voice": "casual_male",
    "max_new_tokens": 4096
  }' \
  --output output.wav
```

The available voices ship inside the checkpoint as `voice_embedding/*.pt` files. List them
from your downloaded snapshot:

```bash
ls "$(hf download mistralai/Voxtral-4B-TTS-2603)/voice_embedding"
```

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "voice": "casual_male",
        "max_new_tokens": 4096,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Streaming

Set `"stream": true` to receive audio chunks in real time over Server-Sent Events (SSE):

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "voice": "casual_male",
    "stream": true
  }'
```

Each event carries a base64-encoded audio chunk; the stream ends with `data: [DONE]`. See the
[TTS usage guide](../basic_usage/tts.md) for a full Python SSE consumer.

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `voice` | `cheerful_female` | Preset voice name from the checkpoint's `voice_embedding/` directory |
| `max_new_tokens` | `4096` | Maximum number of generated acoustic tokens |
| `response_format` | `wav` | Output container |
| `stream` | `false` | Stream audio chunks over SSE |

> Voxtral generation is **deterministic**: the engine fixes `temperature` to `0.0`, so sampling
> parameters such as `top_p`, `top_k`, and `temperature` are not used. Reference-clip voice
> cloning (`references`) is **not** supported for Voxtral — use a preset `voice` instead.

## Performance

Per the [model card](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603), Voxtral targets
production voice agents: ~70 ms first-audio latency at concurrency 1 (RTF ≈ 0.10) and high
throughput under load on a single GPU with ≥16 GB of memory. Output is 24 kHz.

## Known Limitations

- **Preset voices only.** Voxtral selects from named voices baked into the checkpoint; it does
  not clone an arbitrary speaker from a reference clip in this engine.
- **Deterministic decoding.** `temperature` is fixed at `0.0`; you cannot trade determinism for
  diversity through sampling parameters.
- **Language coverage.** Quality is tuned for the 9 supported languages (English, French,
  Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi).
- **Non-commercial license.** The weights are CC BY-NC 4.0; commercial use is not permitted.
