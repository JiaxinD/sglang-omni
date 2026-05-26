# Qwen3 TTS

[Qwen3-TTS-12Hz-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is a discrete
multi-codebook text-to-speech model from the Qwen team. It performs fast voice cloning from a
short reference clip, supports 10 languages, and streams 24 kHz speech with low latency. The
`12Hz` in the name refers to the codec **frame rate** (12 acoustic frames per second), not the
playback sample rate. SGLang-Omni serves two checkpoints — `0.6B` and `1.7B` — through the same
`preprocessing → tts_engine → vocoder` pipeline and the OpenAI-compatible `/v1/audio/speech`
endpoint.

> Qwen3-TTS Base weights are released under **Apache-2.0**.

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
```

Qwen3-TTS Base uses the upstream `qwen-tts` package, which currently pins Transformers 4.57.3.
Install it only in environments that serve Qwen3-TTS:

```bash
uv pip install transformers==4.57.3 accelerate==1.12.0 sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

> Do **not** add `--upgrade` here. It pulls a newer `torch`/`numpy`/CUDA stack and breaks
> inference (mismatched cuDNN, `numba` requires NumPy ≤ 2.3). Pin only what is listed above so
> the image's existing `torch` build is left untouched.

Download a checkpoint (both repositories are public, no token required):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`.

```bash
# 0.6B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

```bash
# 1.7B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

## Synthesizing Speech

### Zero-shot

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

Qwen3-TTS Base is a cloning model: without reference audio the voice sounds robotic. For
natural results, provide a reference clip as shown below.

### Voice Cloning

The `references` field accepts `audio_path` (a local path or HTTP URL) and `text` (the
transcript of that clip). Supplying the transcript enables in-context-learning (ICL) mode and
materially improves cloning quality; omitting it falls back to speaker-embedding (x-vector)
mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and
`references[0].text`.

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Language Hint

`language` biases the model toward a target language. It defaults to `auto` (let the model
detect). Supported languages are Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天天气不错。",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/zh/prompt-wavs/AISHELL3_SSB00050007.wav",
      "text": "对，这样就可以了。"
    }],
    "language": "Chinese"
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` to receive audio chunks in real time over Server-Sent Events (SSE):

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true
  }'
```

Each event carries a base64-encoded audio chunk; the stream ends with `data: [DONE]`. See the
[TTS usage guide](../basic_usage/tts.md) for a full Python SSE consumer.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `references` | `null` | Reference clip for cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `language` | `auto` | Target-language hint (see list above) |
| `temperature` | `0.9` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | `50` | Top-k sampling |
| `repetition_penalty` | `1.05` | Repetition penalty |
| `max_new_tokens` | `2048` | Maximum number of generated codec tokens |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Stream audio chunks over SSE |

## Model Variants

| Checkpoint | Parameters | Config |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | `examples/configs/qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `examples/configs/qwen3_tts_1_7b.yaml` |

Both expose an identical request API. The 1.7B model has higher capacity (typically better
quality) at a larger memory and latency cost; the 0.6B model is lighter and faster.

## Known Limitations

- **Reference audio recommended.** As a cloning model, Qwen3-TTS Base produces robotic speech
  without a reference clip.
- **Transcript improves cloning.** Providing `text` in `references` (ICL mode) yields better
  speaker similarity than speaker-embedding-only (x-vector) mode.
- **Language detection.** `language: auto` may misdetect for short or code-switched inputs;
  set `language` explicitly when you know the target language.
