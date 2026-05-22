# SGLang Omni Refactor Tracking

---

Follows from [sglang#16546](https://github.com/sgl-project/sglang/issues/16546). Addresses problems in [#188](https://github.com/sgl-project/sglang-omni/issues/188).

## Architecture

### System Overview

```
HTTP API → Client → Coordinator → Stage → [Scheduler → ModelRunner → forward]
```

### Layer Responsibilities

| Layer           | Responsibility                                                                     | Model-aware? |
| --------------- | ---------------------------------------------------------------------------------- | ------------ |
| **Coordinator** | Request lifecycle, routing to entry stage, multi-terminal merge, abort broadcast   | No           |
| **Stage**       | IO shell — ZMQ control plane, relay data plane, fan-in aggregation, stream routing | No           |
| **Scheduler**   | Batch selection, KV cache management, compute dispatch                             | Partially    |
| **ModelRunner** | Forward pass, sampling, model-specific hooks                                       | Yes          |

### Directory Layout

```
sglang_omni/
├── pipeline/           # Inter-stage orchestration (model-agnostic)
├── scheduling/         # Scheduling loops (OmniScheduler, SimpleScheduler)
├── model_runner/       # Model runner base + shared FeedbackARModelRunner
├── models/             # Model definitions + pipeline configs
├── config/             # Pipeline config schema + compiler
├── relay/              # Data transfer backends (SHM, NCCL, NixL)
├── serve/              # HTTP server, OpenAI API
├── client/             # Client library
└── proto/              # Message types
```

### Class Diagram

```mermaid
classDiagram
    class Coordinator {
        +entry_stage: str
        +register_stage(name, endpoint)
        +submit(request_id, request) result
        +stream(request_id, request) AsyncIterator
        +abort(request_id) bool
        +run_completion_loop()
    }

    class Stage {
        +name: str
        +control_plane: StageControlPlane
        +relay: Relay
        +scheduler: Scheduler
        +start()
        +run()
        +stop()
    }

    class Scheduler {
        <<interface>>
        +inbox: Queue~IncomingMessage~
        +outbox: Queue~OutgoingMessage~
        +start()
        +stop()
        +abort(request_id)
    }

    class OmniScheduler {
        +request_builder: Callable
        +recv_requests()
        +process_input_requests()
        +run_batch()
    }

    class SimpleScheduler {
        -compute_fn: Callable
    }

    class Code2WavScheduler {
        -model: Code2Wav
    }

    class ModelRunner {
        +execute(scheduler_output) ModelRunnerOutput
        +prepare_forward()* hook
        +post_forward()* hook
    }

    class ThinkerModelRunner {
        +prepare_forward(): inject multimodal embeds
    }

    class FeedbackARModelRunner {
        +write_buffers_fn: Callable
        +extract_output_fn: Callable
        +prefill_forward_fn: Callable
    }

    Coordinator --> Stage : routes requests to
    Stage --> Scheduler : inbox/outbox
    Scheduler <|-- OmniScheduler
    Scheduler <|-- SimpleScheduler
    Scheduler <|-- Code2WavScheduler
    OmniScheduler --> ModelRunner : run_batch delegates to
    ModelRunner <|-- ThinkerModelRunner
    ModelRunner <|-- FeedbackARModelRunner

```

> **Pending — Huapeng**: Architecture overview needs three additions before this section is final.
> 1. Restore the `HTTP API → Client` lifecycle edge to the system overview (currently missing). (raised by Chenyang)
> 2. Add the WebSocket entrypoint to the overview. (raised by Chenyang)
> 3. Add an HTTPS design layer subsection (entry point, request routing, WebSocket bridge). (existing TODO)
>
> Two diagram-level suggestions from Yichi are folded into the same revision: (a) illustrate the ModelRunner hook by splitting `prepare_prefill` / `decode` and `post_prefill` / `decode` rather than the generic `prepare` / `post_forward` (thinker and talker behave differently here); (b) decide whether a `Client` section belongs in this doc.

---

## Pipeline Layer

### HTTP/Websocket

HTTP and websocket is sglang omni's request endpoint which exposes to the outside, as user we can get the response we want via these endpoint.

- **HTTP:**
  - `POST /v1/chat/completions`
  - `POST /v1/audio/speech`
  - `GET /v1/models`
  - `GET /health`
- **WebSocket:**
  - `WS /v1/realtime`

For the difference between http and websocket, you can think it's like when you chat, you use messages or phone call, one is request and get, one is duplex communication. PR ref

### Client

Client is like the adapter between http/websocket and coordinator, it's like the real implementation of endpoint.

Like the `/speech`, it refines the input calls the generate and return the refined output.

Code part:

```python
async for chunk in self.generate(request, request_id=request_id):
    if chunk.audio_data is not None:
        audio_chunks.append(chunk.audio_data)
    if chunk.sample_rate is not None:
        sample_rate = chunk.sample_rate
    last_chunk = chunk
```

### Coordinator

Global request router. Tracks the request lifecycle across stages.

1. Routes new requests to the entry stage
2. Collects completions from terminal stages
3. Merges results when multiple terminal stages exist (e.g., `decode` + `code2wav`)
4. Broadcasts abort to all stages

### Stage

IO shell. Every stage has one scheduler (no branching). Handles all inter-stage communication.

```python
class Stage:
    def __init__(self, name, control_plane, relay, route_fn,
                 input_handler, scheduler, stream_targets, same_gpu_targets):
        self.scheduler = scheduler  # always present
```

Responsibilities:

- **Control plane (ZMQ):** receive `SubmitMessage`, `DataReadyMessage`, `AbortMessage`
- **Data plane (Relay):** read/write tensors between stages via SHM/NCCL/NixL
- **Input aggregation:** wait for multiple upstream stages before dispatching (`AggregatedInput`)
- **Stream routing:** receive/send streaming chunks (hidden states, codec codes)
- **Dispatch:** push all messages into `scheduler.inbox`, drain `scheduler.outbox`

One code path. Stage never checks scheduler type.

### Inter-Stage Communication

```mermaid
sequenceDiagram
   participant A as Stage A
   participant Relay as Relay (SHM/NCCL/NixL)
   participant ZMQ as Control Plane (ZMQ)
   participant B as Stage B
   A->>Relay: write(tensors)
   A->>ZMQ: send(DataReadyMessage)
   ZMQ->>B: recv(DataReadyMessage)
   B->>Relay: read(tensors)
```

- **Control plane (ZMQ):** small messages — Submit, DataReady, Abort, Shutdown. PUB/SUB for abort broadcast, PUSH/PULL for point-to-point.
- **Data plane (Relay):** large tensors. Pluggable backends: SHM (single machine), NCCL (multi-GPU), NixL (RDMA multi-node), Mooncake. Same-GPU stages use CUDA IPC zero-copy automatically.
- **Streaming:** hidden states / codec codes flow via `DataReadyMessage` with `chunk_id` and `is_done` fields, parallel to normal result routing.

The split between control and data planes is the core architectural decision: ZMQ stays out of the tensor path, the relay stays out of the coordination path, and either can be swapped independently.

#### Why not Ray?

The thinker→talker relationship is fundamentally producer–consumer: thinker produces hidden states / text tokens, talker consumes them and emits audio codes. Structurally this is the same pattern as RL training (actor produces trajectories, learner consumes them), where Ray is the dominant orchestration layer. So it is a fair first-principles question whether we should use Ray for inter-stage scheduling instead of hand-rolling ZMQ + relay.

We chose not to. Ray's overhead — extra runtime, object-store semantics, and operational footprint — is significant for our shape: a small fixed number of stages on a small fixed number of GPUs, with no autoscaling. ZMQ + relay covers the topology with much less moving infrastructure. The decision is revisitable; nothing in the pipeline layer hard-bakes "no Ray" beyond the relay backend list.

### `relay_io` Utility Module

Utility module providing:

- **User-facing API**
  - `write_payload` / `read_payload` — full `StagePayload` serialization via relay
  - `send_stream_chunk` — handles same-GPU IPC vs cross-GPU relay, NIXL credit deadlock avoidance
- **Internal-facing API**
  - `write_blob` / `read_blob` — raw tensor transfer for streaming chunks
  - `extract_tensors` / `restore_tensors` — recursive tensor extraction from nested dicts

**API layering.** Stages only call the user-facing API. `write_blob` / `read_blob` get wrapped by `send_stream_chunk` for cross-GPU transfer; `extract_tensors` / `restore_tensors` get used inside `write_payload` / `read_payload` to pull tensors out of nested dicts. The internal layer exists so the user-facing surface stays small.

**Asymmetric stream API — no `recv_stream_chunk`.** Only `send_stream_chunk` exists because the sender has real decisions to make (same-GPU IPC vs cross-GPU relay, NIXL credit management), while the receiver just calls `read_blob` from the stage's main message loop. Wrapping a one-liner into `recv_stream_chunk` would add a layer without adding any value.

---

## Scheduling Layer

All schedulers share the same interface: `inbox`, `outbox`, `start()`, `stop()`, `abort()`.

### OmniScheduler — Composition with SGLang

![OmniScheduler — Composition with SGLang](images/omni_scheduler_composition.png)

For AR stages. Subset of SGLang Scheduler — reuses `get_next_batch_to_run()`, `run_batch()`, `process_batch_result()`, `event_loop_normal()`, overlap scheduling.

- **Reused from SGLang:** `get_next_batch_to_run()`, `process_batch_result()`, `self_check_during_idle()` — KV cache management, prefill/decode scheduling, tree cache, dLLM support
- **Overridden:** `init` (skip ZMQ/tokenizer/metrics), `recv_requests()` (drain inbox, route stream chunks to per-request state), `process_input_requests()` (`request_builder` conversion), `run_batch()` (delegate to ModelRunner), `send_to_tokenizer()` (no-op)
- **Not used from SGLang:** ZMQ channels, tokenizer init, grammar backend, metrics exporter, disaggregation, LoRA, speculative decoding, PP, watchdog

Runs in a dedicated thread. Stage communicates via thread-safe queues.

#### Composition boundary with SGLang

Composing on top of SGLang's `Scheduler` is the right call, but only if we pin the boundary deliberately — RL forks of SGLang have hit upgrade pain by letting composition drift into a de facto fork. Three rules govern the boundary:

1. **Pin, don't track.** Tracking SGLang `main` fits the same-umbrella relationship, but only if CI runs against the real Scheduler rather than mocks. Running that CI is currently too expensive, so we pin.
2. **Minimize reuse surface.** Treat `PrefillManager` and `DecodeManager` as black boxes — public methods only, no reads or writes of internal attributes. The moment we touch internals, composition becomes a fork.
3. **Upstream-first, when affordable.** If OmniScheduler needs something SGLang doesn't cleanly expose, the preferred fix is a hook or factored-out method in SGLang `main` rather than a downstream patch. Today the cost of upstream PRs is high enough that we don't routinely do this; it remains the long-term direction.

`CodePredictor` is placed under Talker, but whether `ThinkerScheduler` and `CodePredictorScheduler` need a separate documented split (their KV cache shapes are quite different) is still an open design question.[^q-thinker-codepredictor-split]

### Error handling

Previously the same OOM produced different externally-visible behaviors across models: S2-Pro and Voxtral returned HTTP 500 (correct), Ming-Omni returned HTTP 200 with `waveform=None` (silent failure), Qwen3-Omni returned HTTP 200 with a zero tensor (looks like a valid waveform downstream). The first two were correct only because they did nothing; the latter two failed because broad `except Exception` blocks in the executor swallowed the error. The fix lives at the Scheduler layer (#449):

1. **Unified catch in `run_batch()`** — Scheduler wraps forward, catches exceptions, marks the request failed, propagates via outbox → Coordinator → HTTP 500 for non-streaming. For streaming responses HTTP headers are already flushed before the first chunk, so HTTP 500 cannot fire mid-stream — successful streams terminate with an explicit completion sentinel frame, and failed streams abort the connection before emitting it. Absent sentinel + premature close is the client-side failure signal.
2. **Model executors forbidden from writing `except Exception`** — model-side code is pure functional and exceptions propagate naturally. Specific expected exceptions must catch specific types, never base `Exception`. Enforced via lint rule, not review discipline: rule 1's catch is path-local to `run_batch()`, so a broad catch in `add_request` or embed-load paths slips past it.
3. **Fallbacks architecturally disallowed** — executor either succeeds or hands off to Scheduler. No third path returning a "fake success" indistinguishable from a real result.
4. **CI fault injection** — inject OOM and verify the correct failure signal per model: HTTP 500 for non-streaming, sentinel-absent + premature close for streaming. Detection complements but does not substitute for rule 2's lint enforcement.

The short-term bridge fix (#302) landed without an `is_oom_error()` helper, since that helper would become dead code once the Scheduler-layer catch lands.

### SimpleScheduler

For non-AR stages (preprocessing, encoders, aggregate, decode). No KV cache, no batching. Just `inbox.get()` → `fn(data)` → `outbox.put()`. Supports inbox/outbox and basic forward operation; batched processing supported where useful.

### Code2WavScheduler

Streaming vocoder. Handles:

- `new_request` → init
- `stream_chunk` → accumulate + decode
- `stream_done` → flush + output

### Message Types

```python
class IncomingMessage:
    request_id: str
    type: "new_request" | "stream_chunk" | "stream_done"
    data: Any

class OutgoingMessage:
    request_id: str
    type: "result" | "stream"
    data: Any
    target: str | None  # for stream: downstream stage name
```

---

## Model Runner + Callbacks

```
ForwardBatch → prepare_forward() → forward() → post_forward() → sample() → ModelRunnerOutput
                    ↑ hook                          ↑ hook
```

| Runner                    | Used by                | Hook behavior                                                        |
| ------------------------- | ---------------------- | -------------------------------------------------------------------- |
| **ThinkerModelRunner**    | Qwen3 / Ming thinker   | prepare_forward: inject multimodal embeddings                        |
| **FeedbackARModelRunner** | Qwen3 talker, Fish TTS | 3 callbacks: write_buffers_fn, extract_output_fn, prefill_forward_fn |

CUDA Graph and `torch.compile` are class-shareable rather than configured model by model — the ModelRunner abstraction exists in part to make this the default; special models can still override.

DiffusionModelRunner is no longer speculative — Ming-Omni (#236) requires it, and image-gen diffusion is functionally working. It should be added as a first-class runner type alongside `ThinkerModelRunner` and `FeedbackARModelRunner` later.

### `ModelRunner` (Base)

Shared execute pipeline for all AR models.

```python
class ModelRunner:
    def execute(self, scheduler_output):
        forward_batch = ForwardBatch.init_new(...)
        batch_result = self.prepare_forward(...)  # hook
        if batch_result is None:
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)
        self.post_forward(batch_result, ...)       # hook
        # sample, logit processing, output extraction
        return ModelRunnerOutput(...)
```

Shared: `ForwardBatch` construction, sampling, repetition penalty, codec suppression, output processing.

> **Pending — Jingwen**: Refactor proposal (raised by Chenyang) — split the current `prepare_forward` hook into `before_forward` (always mutates `forward_batch` in place) and an explicit `custom_forward` branch. The current "the hook returned a value, so short-circuit" pattern is misleading; the explicit split makes the prefill-with-injection path (Fish TTS) honest. Awaiting confirmation of whether this landed in the runner code. Proposed shape:

```python
def execute(self, scheduler_output):
    forward_batch = ForwardBatch.init_new(...)

    # Mutate forward_batch in place (e.g. inject multimodal embeds).
    self.before_forward(forward_batch, ...)

    # Two mutually exclusive paths:
    #   - custom_forward: model-specific forward (e.g. Fish TTS prefill
    #     with VQ embedding injection).
    #   - default forward: standard tp_worker.forward_batch_generation.
    if self.has_custom_forward:
        forward_output = self.custom_forward(forward_batch, ...)
    else:
        forward_output = self.tp_worker.forward_batch_generation(forward_batch)

    self.post_forward(forward_output, ...)
    return ModelRunnerOutput(...)
```

### `ThinkerModelRunner`

Injects multimodal embeddings (image / video / audio) + deepstack before forward.

```python
class ThinkerModelRunner(ModelRunner):
    def prepare_forward(self, ...):
        # Inject multimodal embeds into forward_batch
        ...
```

### `FeedbackARModelRunner`

Shared model runner for all AR + codebook models (Qwen3 talker, Fish TTS, future models) whose feedback loop is **self-contained within a single ModelRunner instance** — both the feedback producer and the receiver live inside one decode step. Cross-stage feedback (producer and receiver in separate schedulers, communicating via relay) is out of scope for this abstraction and would need a different design; today nothing requires it, but stating the boundary up front prevents future contributors from bending the abstraction to cover topologies it was not designed for.

The model's `forward()` handles backbone + secondary head internally. This runner writes / reads model buffers around forward.

```python
class FeedbackARModelRunner(ModelRunner):
    def __init__(self, tp_worker, output_processor, outbox, *,
                 write_buffers_fn, extract_output_fn, prefill_forward_fn=None):
        ...

    def prepare_forward(self, ...):
        if decode:
            self._write_buffers(model, schedule_batch, requests)
        elif prefill and self._prefill_forward:
            return self._prefill_forward(tp_worker, forward_batch, ...)
        return None

    def post_forward(self, ...):
        self._extract_output(model, schedule_batch, requests, outbox)
```

Model-specific behavior via three callbacks:

- `write_buffers_fn`: write previous step's feedback into model buffers
- `extract_output_fn`: read codes / feedback from model after forward
- `prefill_forward_fn`: custom forward for prefill (optional)

#### Callback pattern: bare functions vs Strategy object

Each model currently provides three bare functions that get passed in individually. This works, but the three are semantically coupled — they all operate on the same model's buffers — and nothing at the type level enforces that coupling.

A lightweight improvement is to collect them into a Strategy object:

```python
class FeedbackStrategy(Protocol):
    def write_buffers(self, model, schedule_batch, requests) -> None: ...
    def extract_output(self, model, schedule_batch, requests, outbox) -> None: ...
    def prefill_forward(self, tp_worker, forward_batch, ...) -> Optional[BatchResult]: ...

class QwenTalkerStrategy:
    def write_buffers(self, model, schedule_batch, requests):
        # feedback_embeds + trailing/pad → model._feedback_buffer
    def extract_output(self, model, schedule_batch, requests, outbox):
        # model._output_codes → outbox, _output_embeds → feedback
    def prefill_forward(self, tp_worker, forward_batch, ...):
        # projected input_embeds prefill

class FishTTSStrategy:
    def write_buffers(self, ...): ...
    def extract_output(self, ...): ...
    def prefill_forward(self, ...): ...
```

The bare-function form is what ships today; the Strategy form is the recommended evolution if a third self-contained model joins. The trade-off is mostly typing surface vs explicitness — neither blocks the other.

### Callback Pattern

Each model provides a `callbacks.py` with three functions:

**Qwen3 Talker** — `models/qwen3_omni/callbacks.py`:

1. `write_talker_buffers`: `feedback_embeds` + trailing/pad → `model._feedback_buffer`
2. `extract_talker_output`: `model._output_codes` → outbox, `_output_embeds` → feedback
3. `talker_prefill_forward`: projected `input_embeds` prefill

**Fish TTS** — `models/fishaudio_s2_pro/callbacks.py`:

1. `write_fish_buffers`: codebook values → `model._vq_codes`
2. `extract_fish_output`: `model._output_codes` → per-request output
3. `fish_prefill_forward`: VQ embedding injection into `input_embeds`

Adding a third model = write a new `callbacks.py` with three functions.

### Model.forward(): One Decode Step (AR + Codebook)

Both Qwen3 talker and Fish TTS follow the same internal pattern:

1. Read previous step's feedback from model buffers (written by `FeedbackARModelRunner`)
2. AR backbone → hidden states → logits
3. Sample first code from logits
4. Secondary head predicts remaining codebook layers autoregressively
5. Store combined output → buffers for next step
6. Output: multi-layer codes + feedback

The model class handles steps 1–6 inside `forward()`. `FeedbackARModelRunner` handles writing (before) and reading (after).

---

## Model Directory Convention

Every model follows the same file structure:

```
models/<model_name>/
├── config.py              — Pipeline config (stage definitions, GPU placement)
├── stages.py              — Stage factories (returns callable or OmniScheduler)
├── routing.py             — Stage routing functions (which stage follows which)
├── request_builders.py    — Inter-stage data transform (build engine requests)
├── payload_types.py       — Model-specific pipeline state
├── callbacks.py           — FeedbackARModelRunner callbacks
├── __init__.py
└── components/            — Model-specific torch modules, preprocessors, encoders
```

`routing.py` and `request_builders.py` are kept separate because they answer different questions: `routing.py` decides *which* stage runs next (topology, often deterministic), while `request_builders.py` formats data for models that need special input shapes — e.g. the Qwen3-Omni thinker → talker request transform. Localizing the model-specific format logic in `request_builders.py` keeps `routing.py` thin and framework-shaped.

### Qwen3-Omni

```
models/qwen3_omni/
├── config.py              — 8-stage speech, 6-stage text
├── stages.py              — 8 factories
├── routing.py             — 8 routing functions
├── request_builders.py    — build thinker/talker/encoder requests
├── payload_types.py       — PipelineState, OmniEvent, ThinkerOutput
├── callbacks.py           — write_talker_buffers, extract_talker_output, talker_prefill_forward
├── hf_config.py           — HF config classes
├── merge.py               — Merge 3 encoder outputs for thinker
├── components/
│   ├── thinker.py         — Model loader/wrapper (Qwen3OmniSplitThinker)
│   ├── thinker_model.py   — SGLang thinker model definition
│   ├── talker.py          — SGLang talker model (fused MTP)
│   ├── preprocessor.py    — Tokenize, load media, apply HF processor
│   ├── image_encoder.py   — Image tower
│   ├── audio_encoder.py   — Audio tower
│   ├── talker_input.py    — Build talker prefill
│   ├── streaming_detokenizer.py — Streaming text detokenizer scheduler
│   ├── code2wav_scheduler.py — Vocoder streaming scheduler
│   └── common.py          — Shared helpers
```

Speech pipeline (8 stages): `preprocessing → image_encoder → audio_encoder → aggregate → thinker → decode → talker → code2wav`. The `image_encoder` and `audio_encoder` stages run in parallel — the arrow above is for layout, not sequencing — and the design leaves room for offloading either tower to CPU.

The "tower" terminology for image/audio encoders follows the official Qwen3-Omni names; we keep that vocabulary here rather than introducing a divergent local one.

> **Pending — Jingwen**: `PipelineState` and `OmniEvent` are model-specific but read as framework-level types. Rename to disambiguate (suggested by Chenyang). Tracked until Jingwen confirms whether the rename landed.

### Fish Audio S2-Pro

```
models/fishaudio_s2_pro/
├── config.py              — 3-stage TTS
├── stages.py              — 3 factories
├── routing.py             — 3 routing functions
├── request_builders.py    — build_sglang_tts_request, apply_tts_result
├── payload_types.py       — S2ProState
├── callbacks.py           — write_fish_buffers, extract_fish_output, fish_prefill_forward
├── sglang_model.py        — SGLang model registration
├── tokenizer.py           — Tokenizer wrapper
└── fish_speech/           — Model definitions (text2semantic, DAC codec)
```

Pipeline (3 stages): `preprocessing → tts_engine → vocoder`

---

## Declarative Config

### Example

```python
stages = [
    StageConfig(name="preprocessing",
                factory="...create_preprocessing_executor",
                route_fn="...routing.preprocessing_next"),

    StageConfig(name="image_encoder",
                factory="...create_image_encoder_executor",
                gpu=0, next="mm_aggregate"),

    StageConfig(name="mm_aggregate",
                factory="...create_aggregate_executor",
                wait_for=["preprocessing", "image_encoder", "audio_encoder"],
                merge_fn="...merge_for_thinker",
                next="thinker"),

    StageConfig(name="thinker",
                factory="...create_thinker_executor",
                factory_args={"speech_enabled": True},
                gpu=0, next=["decode", "talker_ar"],
                stream_to=["talker_ar", "decode"]),

    StageConfig(name="decode", factory="...create_decode", terminal=True),

    StageConfig(name="code2wav", factory="...create_code2wav", gpu=1, terminal=True),
]
```

Realtime streaming-input support is a separate workstream owned by #385 and is intentionally outside the scope of this config example.[^q-realtime-streaming]

Routing rule: exactly one of `next`, `route_fn`, or `terminal=True`. There is no "one thinker, multiple talkers" fan-out — talker decode requires the thinker's hidden state as prefix, so driving two independent talkers from the same prefix carries no useful semantics. Derived from stages: `entry_stage` (first stage), `terminal_stages`, `gpu_placement`, relay device.

### `StageConfig` reference

| Field        | Type             | Default    | Description                                                                        |
| ------------ | ---------------- | ---------- | ---------------------------------------------------------------------------------- |
| name         | str              | _required_ | Unique stage identifier                                                            |
| factory      | str              | _required_ | Dotted import path to factory function                                             |
| factory_args | dict             | {}         | Args forwarded to factory (model_path, gpu_id auto-injected)                       |
| next         | str \| list[str] | None       | Static routing: downstream stage(s). Replaces routing functions for most stages    |
| route_fn     | str              | None       | Dynamic routing: dotted path to fn(request_id, output) → str \| list[str] \| None  |
| terminal     | bool             | FALSE      | Terminal stage — no downstream. Coordinator collects the result here               |
| gpu          | int \| list[int] | None       | GPU id(s). None = CPU stage. List for TP (one GPU per rank)                        |
| tp_size      | int              | 1          | Tensor parallelism ranks. Must match len(gpu) if gpu is a list                     |
| wait_for     | list[str]        | None       | Fan-in: wait for these upstream stages before dispatching                          |
| merge_fn     | str              | None       | Dotted path to fn(dict[str, StagePayload]) -> StagePayload. Required with wait_for |
| stream_to    | list[str]        | []         | Stream hidden states / codes to these stages (parallel to normal routing)          |
| relay        | RelayConfig      | None       | Override relay settings. Auto-inferred from gpu if not set                         |

#### `route_fn` contract

`route_fn` has narrow utility: Qwen3-Omni and Fish S2 Pro are fully covered by `next` + `stream_to`. It is only needed when the hidden state itself carries a modality tag and the downstream branch must be decided from the data (e.g. Ming, where the output if/else's into a video or audio head). The contract is therefore narrow on purpose:

- Return value must be a stage already declared in `next`, so the topology stays statically derivable.
- Returning `None` is disallowed — drops belong in an explicit terminal sink, not hidden inside routing.
- The docstring restricts use to data-driven modality dispatch.

One field, narrow contract, easy to widen later when a real consumer shows up.

#### Runtime parameter plumbing

Critical runtime params (`mem_fraction_static`, `thinker_max_seq_len`, and soon `video_fps`) are today either hardcoded deep in the stack or routed through ad-hoc overrides that nobody fully understands. CLI, config-file, and override paths do not compose, and every new param reinvents its own precedence resolution.

The refactor should consolidate this into one canonical mechanism: a typed, stage-addressable override primitive at the `PipelineConfig` layer, with CLI / config / env as thin adapters on top. Every runtime param then flows through the same primitive. A related symmetry gap to fix at the same time: length validation guards only the thinker input side. Talker also needs an output-length cap so an unbounded decode loop (missed stop token, hallucination loop) cannot drive OOM or tail latency the same way. Both belong on the same plumbing once it exists.

#### Stage placement — same-GPU co-location

Stages may share GPUs. Earlier topologies hard-rejected same-GPU speech-stage placement, which left Talker on H200 at <2% utilization long-term. Informed by Ratish's vLLM-Omni investigation (vLLM co-locates thinker + talker on a single device via per-stage memory budgeting + NVML accounting), the placement model now treats "any stage on any GPU" as first-class, with budgeting that accounts for co-tenants rather than rejecting the topology. See [Design Decision History § PR #430](#2026-05-12--pr-430-colocated-stage-execution-colocation) for the typed runtime config + placement planner that shipped this.

Memory-fraction semantics have also been pinned down: vLLM's `gpu_memory_utilization` is a fraction of total VRAM, while SGLang's `mem_fraction_static` is a fraction of remaining VRAM after weights load — more principled for single-stage LLM, but ambiguous for omni where stages load sequentially and "remaining" depends on load order. The placement model now uses one explicit semantics rather than inheriting the ambiguity.

Whether `factory` and `factory_args` should collapse into a single field is still open.[^q-factory-args-merge]

### `PipelineConfig` reference

Derived (computed from stages, not set manually): `terminal_stages`, `gpu_placement`.

There is no compiler class. An earlier proposal threaded pipeline construction through a `compiler_pipeline()` entry point, but the multi-process path (`mp_runner._build_stage_groups`) re-implemented most of the same logic independently, with two near-duplicate `_resolve_factory_args` helpers. The compiler class was removed in [#447](#2026-05-15--pr-447-unify-serving-on-multiprocess-runner-rfc) — pipeline construction now happens through a plain init function per model, which is sufficient given how few pipelines we maintain.

The `Pipeline` vs `Stages` distinction in code still needs to be sharper: both names appear in different places without a crisp mental model. This should be pinned down before the field set grows further.

---

## Multi-Process Runner

```
pipeline/
├── stage_process.py    # StageProcessSpec (picklable) + subprocess entrypoint
├── stage_group.py      # StageGroup — manages N processes per stage
└── mp_runner.py        # MultiProcessRunner — orchestrates all groups
```

```mermaid
graph TB
    Main["Main Process<br/><b>Coordinator</b>"]

    subgraph "StageGroup: thinker (tp_size=2)"
        P0["Process tp_rank=0<br/>GPU 0"]
        P1["Process tp_rank=1<br/>GPU 1"]
        P0 <-->|NCCL| P1
    end

    subgraph "StageGroup: talker (tp_size=1)"
        P2["Process<br/>GPU 2"]
    end

    subgraph "StageGroup: preprocessing (tp_size=1)"
        P3["Process<br/>CPU"]
    end

    Main -->|"ZMQ (rank 0 only)"| P0
    Main -->|ZMQ| P2
    Main -->|ZMQ| P3
```

> **Pending — Jingwen**: Merge `stage_group.py` + `stage_process.py` into a single `stage_workers.py` (suggested by Chenyang). `StageGroup` is the only consumer of `StageProcessSpec`, the subprocess entrypoint is ~40 lines, and the spec is a small dataclass — none of the three justifies its own file. Consolidating keeps "how a stage's processes get defined, spawned, and managed" in one place and leaves `mp_runner.py` focused on cross-stage orchestration. Awaiting confirmation of whether this landed.

### `StageProcessSpec`

A fully-resolved, picklable dataclass built once in the main process. Subprocesses never re-compile the pipeline config — they just construct a `Stage` from the spec and run it.

A rename to `StageLaunchConfig` would carry the same meaning more clearly; this remains an open suggestion.[^q-spec-rename]

The main process resolves all dotted strings, injects `model_path` / `gpu_id` into factory args, allocates ZMQ endpoints, and computes stream targets and relay config. The spec captures everything the child process needs.

The subprocess entrypoint (`stage_process_main`) is ~40 lines: import factory, call it, build routing callable from `route_fn` or `next_stages`, build input handler from `wait_for` / `merge_fn`, construct `Stage`, run.

#### Parallelism axes — TP today, extension path

`StageProcessSpec` exposes `tp_size` as a top-level field. This treats TP as a special axis, but it is only one of several plausible parallelism strategies — Qwen3-Omni's Thinker is MoE and could want EP, and throughput-oriented stages might want DP across replicas. If we add either later, we will accumulate `tp_size` / `ep_size` / `dp_size` at the top level.

The cleaner long-term shape is to group them under a single `parallelism: ParallelismConfig` field — `ParallelismConfig(tp=N)` reads as clearly as `tp_size=N` and leaves room to add `ep` and `dp` without further schema churn. We are intentionally not making that change now: with only TP in use, a `ParallelismConfig` would have exactly one attribute and add visual weight without adding capability. The intended migration is to introduce the group field at the same time as the second parallelism axis lands.

### `StageGroup`

Manages the lifecycle (spawn, `wait_ready`, shutdown, health monitoring) of all OS processes backing one logical stage. For `tp_size == 1` (default), one process. For `tp_size > 1`, spawns one process per TP rank with appropriate `tp_rank` / `gpu_id`.

### `MultiProcessRunner`

Orchestrates startup across all `StageGroup`s. `_build_stage_groups(config)` turns a `PipelineConfig` into `list[StageGroup]` by iterating over stages, resolving factory args, allocating endpoints, and building one `StageProcessSpec` per TP rank per stage. The Coordinator runs in the main process and only talks to rank 0 of each group.

### Tensor Parallelism Support

TP within a stage is orthogonal to pipeline parallelism between stages. The `StageGroup` spawns `tp_size` processes per AR stage. Each process runs a full `OmniScheduler` + `ModelWorker` with a different `tp_rank` and `gpu_id`. NCCL collectives inside the model forward keep TP ranks in lockstep. The Coordinator is TP-unaware — it only talks to rank 0 of each group.

Within a TP group, rank 0 receives from the control plane and broadcasts to peer ranks. All ranks make identical scheduling decisions. Only rank 0 sends results downstream. Each stage gets its own NCCL port (`_NcclPortAllocator` in `mp_runner.py`).

Declaring TP in a stage:

```python
StageConfig(name="thinker", factory="...", gpu=[0, 1, 2, 3], tp_size=4)
```

`StageGroup` spawns 4 processes. NCCL collectives inside the model forward keep them in lockstep. The coordinator only talks to rank 0. Each stage gets its own NCCL port.

---

## Supported Pipelines

### Qwen3-Omni (8-stage speech)

```mermaid
graph LR
      P[preprocessing] --> IE[image_encoder]
      P --> AE[audio_encoder]
      P --> AG[mm_aggregate]
      IE --> AG
      AE --> AG
      AG --> T[thinker<br/>GPU 0]
      T -->|result| D[decode<br/>terminal]
      T -.->|stream token ids| D
      T -.->|stream hidden states| TA[talker_ar<br/>GPU 1]
      TA -.->|stream codes| C2W[code2wav<br/>GPU 1<br/>terminal]
```

- `result`: from decode (terminal)
- `stream token ids`: thinker → decode
- `stream hidden states`: thinker → talker_ar
- `stream codes`: talker_ar → code2wav

Thinker streams hidden states to talker while simultaneously outputting text. Coordinator merges both terminals.

### Fish Audio S2-Pro (3-stage TTS)

```mermaid
graph LR
    P[preprocessing] --> E[tts_engine<br/>GPU 0] --> V[vocoder<br/>terminal]
```

### MiMo-Audio ([#249](https://github.com/sgl-project/sglang-omni/issues/249)) — planned

```mermaid
graph LR
    P[preprocessing] --> T["thinker<br/>(text + audio codes)"] --> CP[code_predictor] --> V[vocoder<br/>terminal]
    T -->|text-only| Term[terminal]
```

4-stage, single GPU. Thinker generates text + audio codes in one pass. No new abstractions needed.

### Ming-Omni ([#236](https://github.com/sgl-project/sglang-omni/issues/236)) — planned

```mermaid
graph LR
   P[preprocessing] --> AE[audio_encoder] --> AG[aggregate]
   P --> AG
   AG --> T["thinker<br/>100B MoE<br/>GPU 0-3, tp_size=4"]
   T --> D[decode<br/>terminal]
   T --> TK["talker<br/>(CFM+DiT diffusion)<br/>GPU 4<br/>terminal"]
```

---

## Adding a New Model

1. Create `models/<name>/config.py` — `PipelineConfig` subclass with stage definitions (routing, GPU placement, fan-in all inline via `next` / `wait_for` / `gpu`)
2. Create `models/<name>/stages.py` — factory per stage (return callable for `SimpleScheduler`, or `OmniScheduler` for AR)
3. Create `models/<name>/callbacks.py` — if AR + codebook: three functions for `FeedbackARModelRunner`
4. Create `models/<name>/components/` — model definitions, preprocessor, encoders
5. (Optional) Create `models/<name>/routing.py` — only if a stage needs dynamic routing (`route_fn`)

Everything else (`Stage`, `Coordinator`, `OmniScheduler`, `ModelRunner`, relay, compiler, mp_runner) is reused as-is.

---

## tp_size

> **Pending — Jingwen**: empty section in the original Lark export — likely intended as a TP-specific subsection but never filled. Resolve by either writing it out or dropping the heading. Tracked here until a decision lands.

---

## Open design questions

These are surfaced for visibility — none block current work. Footnotes from earlier sections land here.

[^q-thinker-codepredictor-split]: **Thinker vs CodePredictor scheduler split.** `CodePredictor` is currently placed under Talker, but the KV cache shape diverges from `ThinkerScheduler` enough that a documented separation may be warranted. No proposal is on the table yet; tracking here so future scheduler refactors revisit it. (raised by Chenyang)

[^q-realtime-streaming]: **Realtime streaming-input semantics.** PR [#385](#2026-05-04--pr-385-openai-realtime-websocket-endpoint-v1-feature) introduces a `/realtime` endpoint for streaming-in audio with WebSocket-backed SSE response, aligned with the OpenAI realtime interface. The detailed protocol — chunk framing, partial-result emission, cancellation semantics — is still being worked out in #385 and is intentionally not specified here. (raised by Huapeng)

[^q-factory-args-merge]: **Collapse `factory` and `factory_args`.** Currently `StageConfig` carries `factory` (dotted path) and `factory_args` (dict) as separate fields. They could plausibly be one field — open question whether the gain in conciseness is worth losing the per-field type. (raised by Chenyang)

[^q-spec-rename]: **`StageProcessSpec` → `StageLaunchConfig` rename.** "Spec" is vague; `StageLaunchConfig` reads as what it is — the picklable record of everything a subprocess needs to launch a stage. Cosmetic but worth doing the next time the class is touched. (raised by Chenyang)

---

## Progress Tracking

[PR #334](https://github.com/sgl-project/sglang-omni/pull/334) — V1 pipeline is still being debugged to pass all CIs.

Following the suggestion in [#188 (comment)](https://github.com/sgl-project/sglang-omni/issues/188#issuecomment-4161198732), we should also track how many files need to be touched and the upper bound of the cost of integrating a new model. Boson's upcoming model will serve as the first concrete data point.

---

## Design Decision History

This section consolidates the design rationale from RFC-style PRs that shaped the architecture, ordered by PR creation date. Each header links back to the PR for the full body and discussion. State and merge date appear in the italic line below the header. Each entry has the same shape: a one- or two-sentence summary, a few bullets expanding the scope, and a "Why it matters" note on the role the PR plays in the broader refactor.

### [2026-04-15 — PR #294: Alternative pipeline added [RFC]](https://github.com/sgl-project/sglang-omni/pull/294)

_State: CLOSED (kickoff; superseded by the per-phase RFCs below)._

Opening RFC of the V1 refactor series, sketching a four-phase plan for replacing the V0 pipeline without a long-lived divergent branch.

- **Phase 1:** add the alternative pipeline alongside the legacy one (feature-flagged)
- **Phase 2:** port Fish Audio onto the new path, remove legacy Fish support
- **Phase 3:** port Qwen-Omni onto the new path, remove legacy Qwen support
- **Phase 4:** clear out the old pipeline once nothing depends on it

**Why it matters:** This is the canonical statement of refactor intent. The PR closed without merging, but the side-by-side migration discipline it established — every intermediate state has a working server — is the rule every subsequent PR in this history followed.

### [2026-04-23 — PR #334: V1 pipeline added [RFC]](https://github.com/sgl-project/sglang-omni/pull/334)

_State: MERGED 2026-05-02. Run with `--version v1`._

Introduced the V1 pipeline as an opt-in path via `--version v1`, and published the project trackboard that anchored the rest of the refactor.

- **Code-quality cleanup:** AI-generated boilerplate, silent fallbacks, over-chatty comments
- **Benchmark coverage:** validate Qwen-Omni and Fish on V1 for correctness and speed
- **In-progress features:** Ming-Omni, flow-matching / diffusion, streaming realtime input, TP, same-GPU memory management, server-arg config plumbing
- **Qwen-Omni optimizations:** piecewise CUDA Graph for talker, high-performance code-predictor backend

**Why it matters:** The discoverability anchor for the V0 → V1 cutover. Names individual owners per work item so contributors can pick up threads independently; several follow-up issues and PRs in this history root back to this trackboard.

### [2026-05-04 — PR #385: OpenAI Realtime WebSocket endpoint [V1, Feature]](https://github.com/sgl-project/sglang-omni/pull/385)

_State: MERGED 2026-05-18. Disabled by default; opt in with `--enable-realtime`._

Mounts `/v1/realtime`, an OpenAI-Realtime-compatible WebSocket API on top of V1, enabling streaming audio in and streaming transcript deltas out for low-latency voice agents and live transcription / translation.

- **`events.py`** — Pydantic schemas for the OpenAI Realtime client/server event vocabulary
- **`audio_buffer.py`** — append-only PCM16 rolling buffer
- **`session.py`** — per-WebSocket state machine; dispatches client events, drives the engine via `Coordinator.stream()`, translates engine deltas back to Realtime server events
- **`manager.py`** — in-memory `session_id → RealtimeSession` registry
- **Scope:** OpenAI Realtime superset, broader than the transcription-only sglang upstream RFC

**Why it matters:** Demonstrates that the V1 Coordinator entry point is general enough to host OpenAI-spec endpoints as thin adapters rather than parallel engine paths. Validates the bidirectional Coordinator stream API as the intended way to add future protocols.

### [2026-05-05 — PR #397: V1 unit test rewrite top-down [RFC]](https://github.com/sgl-project/sglang-omni/pull/397)

_State: MERGED 2026-05-12. No runtime changes; reorganization + contract tests._

Reorganized the V1 unit tests into component-focused folders so each file maps directly to the behavior it protects.

- **`tests/unit_test/pipeline/`** — framework contracts: compile-time schema validation, coordinator multi-terminal completion + abort, stage per-request aggregation + relay tensor round-trips, scheduler batch success / error emission
- **`tests/unit_test/qwen3_omni/`** — Qwen3-Omni topology, request / result tensor shapes, scheduler behavior
- **`tests/unit_test/fishaudio_s2_pro/`** — Fish topology, VQ prompt injection, vocoder batching
- **Deliberate restraint:** "protect the most important protocols, leave deeper tests for follow-up"

**Why it matters:** Establishes the test baseline that subsequent feature PRs extend rather than reinvent. The restraint prevents the historical drift where unit tests grow into a parallel re-implementation that decays in lockstep with the real code.

### [2026-05-06 — PR #401: SGLang-Omni Router for V1 [Router]](https://github.com/sgl-project/sglang-omni/pull/401)

_State: MERGED 2026-05-13. Part of #376._

Adds the SGLang-Omni Router: a standalone process (`sgl-omni-router`) that fronts complete V1 server replicas behind one OpenAI-compatible endpoint. Selects one routable worker per request and forwards the original bytes.

- **Worker sources:** homogeneous URL pool (`--worker-urls`), heterogeneous JSON manifest with per-worker capabilities (`--worker-config`), or managed local launcher from YAML (`--launcher-config`)
- **Selection pipeline:** payload-size guard → bounded metadata extraction → routable / capability / model filters → safe-superset resolution → policy (`round_robin` / `least_request` / `random`)
- **Health and admin:** per-worker failure tracking drives `/ready`; exposes admin and merged `/v1/models` endpoints
- **Managed launcher:** spawns workers from YAML and waits for all to pass `/health` in parallel before accepting traffic

**Why it matters:** Decouples horizontal scaling from the pipeline architecture. Each worker remains a full V1 replica with its own Coordinator — the router never splits a request across stages — which keeps the V1 boundary intact while letting deployments scale by replication.

### [2026-05-07 — PR #406: Qwen3-Omni V1 real text and audio streaming [V1 Feature]](https://github.com/sgl-project/sglang-omni/pull/406)

_State: MERGED 2026-05-15._

Turns Qwen3-Omni V1 from "`stream=true` is a no-op" into real per-token text streaming on `thinker → decode` and real per-window audio streaming on `talker_ar → code2wav → Coordinator`.

- **New first-class V1 concept:** terminal stage forwards `target=None` stream chunks to the Coordinator (SSE), backed by `Stage.is_terminal` and `_send_stream_to_coordinator`
- **`StreamingDetokenizeScheduler`** — consumes per-token stream chunks, emits UTF-8-boundary-safe text deltas
- **`Qwen3OmniCode2WavScheduler`** — latches the streaming flag per request, emits one audio frame per decoded window
- **Slim final `result`** under `stream=true` (`{modality, sample_rate}` only) to avoid duplicate full-payload IPC
- **Hard failure** if a non-terminal stage emits `target=None` — previously a silent drop

**Why it matters:** Lifts streaming from a model-specific feature into a V1 framework primitive (terminal-stage forwarding + per-request streaming-flag latching). Future modal endpoints inherit this pattern rather than re-inventing it.

### [2026-05-07 — PR #407: Unify V1 launcher on multiprocess runner [Bugfix, RFC]](https://github.com/sgl-project/sglang-omni/pull/407)

_State: CLOSED (folded into #447 / launcher consolidation work)._

Argued that the V1 single-process launcher is just the multi-process launcher with one stage — keeping both is redundant double maintenance for endpoint allocation, factory-arg resolution, and process spawning.

- **Diagnosis:** the dual launcher path is historical baggage, not a meaningful deployment distinction
- **Bugfix:** `mp_runner._build_stage_groups` was launching the endpoint process twice
- **Proposal:** route every launch through the multi-process launcher
- **Outcome:** closed without merging; consolidation absorbed into #447

**Why it matters:** Captures the rationale that drove the launcher consolidation. The diagnosis and bugfix informed #447 and downstream cleanups even though no commits from this branch shipped — kept here to attribute both the design decision and the double-launch bugfix correctly.

### [2026-05-12 — PR #430: Colocated Stage Execution [Colocation]](https://github.com/sgl-project/sglang-omni/pull/430)

_State: MERGED 2026-05-16. Follows colocation RFC + #329 / #376._

Implements the colocated-stage execution path for Omni V1, making Qwen3-Omni speech runnable as a single colocated v1 server while preserving the V1 architecture boundary.

- **V1 boundary preserved:** `PipelineConfig → typed runtime config → placement plan → stage process launch → backend adapter → SGLang ModelRunner / KV pool sizing`
- **Omni placement semantics, not SGLang global-free-memory:** per-stage `runtime.resources.total_gpu_memory_fraction` budget
- **KV headroom for SGLang AR stages:** `available_kv_bytes = total_gpu_memory_bytes * fraction - accounted_stage_memory_bytes`
- **Model-agnostic planner** (`config/placement.py`): sums per-GPU stage budgets, rejects over-budget colocated groups, computes same-GPU stream targets before processes start
- **Qwen3-Omni placement policy:** rejects unsupported topologies (standalone `code_predictor`, unsupported thinker / talker TP); admits same-GPU thinker / talker only via `Qwen3OmniSpeechColocatedPipelineConfig`

**Why it matters:** The biggest single deployment-shape win in the V1 refactor. Lets a thinker + talker speech model run as one process on one GPU rather than two separate stages, dramatically reducing the resource footprint for inference clusters that don't need horizontal stage parallelism.

### [2026-05-15 — PR #447: Unify serving on multiprocess runner [RFC]](https://github.com/sgl-project/sglang-omni/pull/447)

_State: CLOSED. Compiler delete + endpoint allocation move; ideas referenced inline above._

Proposed unifying pipeline serving on `MultiProcessPipelineRunner` and removing the legacy direct compiler / runtime path that had grown ad-hoc helpers across the codebase.

- **Delete** `sglang_omni.config.compiler` entirely
- **Move endpoint allocation + IPC runtime-dir ownership** to `sglang_omni.pipeline.endpoints`
- **Move factory-args, relay config, stream-target helpers** to `sglang_omni.pipeline.runtime_config`
- **Always start** pipelines through `MultiProcessPipelineRunner`; CPU stages keep `gpu_id=None`, TP stages require explicit GPU placement
- **Expose stage endpoints** from the MP runner so the profiler can attach
- **Harden stage routing** so invalid downstream / stream targets fail explicitly rather than silently dropping

**Why it matters:** The structural cleanup that finally retired the dual launcher architecture. Although the PR closed without merging through this branch, the compiler removal and `_resolve_factory_args` deduplication landed via this work — the PipelineConfig section above references that resolution.

### [2026-05-17 — PR #461: Stage-GPU-process topology [RFC]](https://github.com/sgl-project/sglang-omni/pull/461)

_State: MERGED 2026-05-17. Resolves issue #459._

Locks in the stage → GPU → process topology mapping as the canonical V1 placement model.

- **Stage** → placement entry → one or more OS processes, each pinned to specific GPU ids
- **TP groups** within a stage spawn one process per rank
- **Stage groups** remain the unit of lifecycle ownership (spawn, `wait_ready`, shutdown, health monitoring)
- **Coordinator** only talks to rank 0 of each group, keeping it TP-unaware
- **Co-location** on shared GPUs allowed; over-budget colocated groups rejected up front

**Why it matters:** Crystallizes the placement rules that earlier RFCs introduced piecewise. After this PR there is one canonical way to express "where does a stage run" — every subsequent feature (router, realtime, colocation refinements) builds on this topology contract rather than inventing its own.

### [2026-05-21 — PR #509: Remove TCP control-plane endpoints [RFC, Feat]](https://github.com/sgl-project/sglang-omni/pull/509)

_State: OPEN as of this writing._

Removes TCP endpoint support from the pipeline control plane and makes IPC the only supported transport.

- **Scope:** control plane carries local coordination messages (completion, abort) between processes on a single node
- **Why IPC-only:** IPC sockets are the natural fit; TCP doesn't unlock any useful deployment mode here
- **TCP was fragile in practice:** endpoint reservation / allocation could diverge from the endpoint that later got bound — a known source of "works locally, fails in CI" bugs
- **Cleanup:** deletes the TCP branch entirely and simplifies endpoint allocation around IPC

**Why it matters:** Final cleanup of port-based control-plane configuration. Removes a known fragile transport and shrinks the surface area the rest of the framework has to support. The last open-RFC item in this history; expected to land soon.
