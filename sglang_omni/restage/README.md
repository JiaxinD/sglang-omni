# Restage integration

Restage plans stage placement, process replicas and resource configuration.
The planning CLI exports **unmeasured candidates**. The source-checkout
benchmark modules below measure supplied candidates and select a winner
within the tested configurations and load grid, subject to quality and SLO
checks. Hardware calibration and prediction-based candidate ranking are not
yet connected to that workflow.

## Generate candidates

Use a serving YAML that names the model's `config_cls` and `model_path`, then:

```bash
sgl-omni autotune plan \
  --config pipeline.yaml \
  --search-space examples/configs/restage_qwen3_tts_search.json \
  --output restage-plan \
  --max-candidates 256
```

The example varies Qwen3-TTS engine memory budgets on two visible GPUs,
setting both the declared stage budget and `engine.mem_fraction_static`.
Its values are search choices, not recommended memory settings. Use device IDs
in the same visible-device namespace as the serving process. A four-GPU
budget can use `"devices": [0, 1, 2, 3]`.

`replica_counts` lists the choices for each GPU process. Each named dimension
contains alternative groups of ordinary serving CLI overrides. Related
changes, such as `thinker.tp_size` and `thinker.gpu`, belong in the same
group. Different dimensions are combined. Only use TP sizes supported by
the model; schema acceptance alone does not establish runtime support.

Every accepted placement obeys the existing process and declared memory
constraints. Actual memory usage and model correctness still need runtime
validation. The planner preserves CPU stages and expands all members of a
replicated process together.

The new output directory contains:

- `search-space.json`: the declared choices;
- `candidates.jsonl`: parameter choices, placements, config filenames or
  rejection reasons;
- `candidate-*.yaml`: configurations readable by the serving config loader;
- `summary.json`: examined, accepted and rejected counts, and whether the
  declared space was exhausted. `complete` refers only to enumeration.

`--max-candidates` counts accepted and rejected records. A truncated search
does not cover the full declared space; enumeration order is not a ranking.
No GPU serving process starts during planning.

## Explore MPS client limits

`examples/configs/restage_qwen3_tts_mps_search.json` uses the existing native
MPS runtime and stage environment defaults. It puts preprocessing, the TTS
engine and vocoder in separate processes, and compares MPS off, MPS on at
100/100, and client percentages of 50/50, 75/25 and 25/75. Keeping an uncapped
MPS-on control separates MPS scheduling effects from the effect of caps.
Also include the shipped configuration as the campaign baseline.

The example sets both `tts_engine.gpu_memory_fraction` and
`tts_engine.engine.mem_fraction_static` to 0.4. The former declares a placement
budget; the current Qwen3-TTS factory does not consume that value as an engine
allocation setting. The latter reaches SGLang through `server_args_overrides`.
Without it, the engine can retain its 0.85 default despite a smaller declared
stage budget. These settings are not hard memory isolation; verify actual
allocation before treating colocated candidates as feasible.

These values are experimental choices, not measured recommendations.
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` limits the available client execution
resources; it does not reserve an exclusive SM partition. Native MPS currently
supports non-TP processes on one physical GPU each. Stages sharing a process
share its client limit; different limits require separate processes.

Use `mps=on` for capped trials: auto mode can leave single-client GPUs without
MPS. Stage env values are defaults, so remove an inherited
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` when testing these choices. Keep daemon
limits and per-context partition settings fixed across comparisons, and
record them. Actual MPS attachment and the SM count visible to each CUDA
worker/context must be verified on the target stack before accepting a
capped result. The CPU contract test proves configuration-to-child-environment
propagation only. Green Context and exclusive SM partitioning remain separate
runtime work. See the [NVIDIA MPS environment reference](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html).

## Measure a TTS candidate

After allocating the visible GPUs, run one candidate with a JSON trial spec:

```bash
python -m benchmarks.benchmarker.restage_tts --spec trial.json --output trial-results
```

The spec supplies `config_path`, `model_path`, `asr_config_path`,
`asr_model_path`, `port`, `rate`, `lang`, `max_wer`, `slo`, and `samples`.
Each sample has `sample_id`, `ref_text`, `ref_audio`, and `target_text`.
Use unique sample IDs, immutable checkpoint paths and a fixed evaluation
dataset. Config paths are relative to the spec; audio paths should be absolute.
Optional `sender_options` are the existing TTS sender options, such as
`stream`, `voice`, `task_type`, and `no_ref_audio`. Set them for the checkpoint's
supported task. `warmup` defaults to one request.

The trial launches TTS, records an open-loop arrival cohort, stops TTS, then
launches the configured ASR model on the same allocated devices and port.
ASR does not run during the timed TTS cohort. Both services clean up only
their own process group. The caller must ensure the GPU allocation remains
available throughout the trial; this command is not an allocation manager.

Outputs include the workload and sender settings, incrementally saved raw request records,
service logs, the ASR quality protocol, per-request audio/WER evidence and
joint SLO results. Missing or failed transcription is not quality success.
WER measures transcript agreement; speaker similarity and perceptual quality
require separate evaluation. A single trial does not establish a best topology
or the maximum sustainable arrival rate.

For an Omni model's read-aloud workload, set `api` to `chat` in the TTS
trial spec (or campaign `trial_options`). This reuses the existing Omni
SeedTTS sender at `/v1/chat/completions`. Supply its required `sender_options`:
`voice_clone`, `speaker`, `max_tokens`, `temperature`, and `stream`;
`system_prompt` is optional. Choose these for the checkpoint and workload.
For example, Ming may need a read-aloud system prompt to avoid chat responses.
The audio must agree with the full `target_text` under the same WER check.

This covers speech generation through the chat API, not arbitrary multimodal
conversation or image/video understanding. The chat sender records first-audio
time, per-chunk PCM duration and maximum playback underrun for streaming
responses. The default `api`
is `speech`, preserving `/v1/audio/speech` trials.
For either API, `max_ttfa_s` requires `sender_options.stream=true` because
non-streaming responses do not provide a first-audio timestamp.
`max_underrun_s` also requires streaming. As with the existing speech sender,
fewer than two chunks leaves continuity unmeasured; such a request does not
pass an explicit playback constraint. Playback starts at the first chunk,
without an additional client buffering allowance.

Omni speech request records also include `server_request_id`, the unique ID
sent through the chat API and used by server request profiling. It differs
from the dataset `request_id`: repeated samples and warmup calls get distinct
server IDs. Use it to join raw request events to the measured workload when
profiling is enabled. This does not itself enable profiling or turn stage
residence time into isolated service time. The `/v1/audio/speech` sender
records the same field from the server's `X-Request-ID` response header for
both streaming and non-streaming audio. Older servers without that header
leave it unset. ASR sender linkage remains pending.

Set `profile=true` in a TTS trial or its campaign `trial_options` to collect
the existing JSONL request profiler alongside serving. This starts before
warmup and stops after the measured cohort; the report joins only measured
server IDs. The owned service stops before `profile-report.json` is built,
so raw files can be flushed. `request-events/` retains the original events.
The report selects this run and pairs stage intervals within each PID,
preserving separate worker timelines and requests without correlated events.

This is diagnostic collection, not an isolated calibration or a readiness
acknowledgement from every worker. Event pairs do not prove complete stage
coverage or GPU service time; the report explicitly leaves `calibration_ready`
false. Omni and audio-speech requests support per-request reports when server
IDs are available. Profiling overhead may affect performance:
keep the same setting across comparisons and validate final capacity without
profiling. The default is off.

To compare explicit candidates, use:

```bash
python -m benchmarks.benchmarker.restage_campaign --spec campaign.json --output campaign-results
```

The campaign spec contains `configs` (candidate key to YAML path), `baseline`
(one of those keys), `rates`, `repeats`, `arrival_seed`, and `trial_options`.
`trial_options` contains the single-trial fields above except `config_path`,
`rate`, and `arrival_seed`. All candidates share those options. Each repeat
uses the same seeded arrival sequence across candidates. Configurations are
copied into the results directory before measurement; use absolute model
paths so the copies resolve the same checkpoint.

`selection.json` ranks candidates by the highest prefix of the tested rate grid
that passed every repeat, then median goodput at that prefix endpoint.
An isolated passing point above a failed lower rate remains visible as
`best_tested_rate`, but does not raise the `passing_prefix_rate` used for ranking. Exact ties retain the
baseline. It records comparison with the baseline, nonmonotonic observations
and whether the highest tested rate still passed. `recommended.yaml` is
written only when a candidate has a passing rate. These are empirical
comparisons within the measured space, not confidence bounds or global
optimality claims. Failed trial execution stops the campaign and records
`failure.json`; it is not silently converted into a low performance score.

The current campaign measures candidates sequentially and retains its order
in the trial log. It does not yet randomize/interleave candidate order or
reuse a serving process across load points. Prediction-based pruning and
held-out calibration validation remain separate work.

### Resume an interrupted campaign

Set `run_identity` in the original spec to a fixed identifier for the source
revision, runtime image, GPU hardware and model/remote-asset snapshots. The
caller must verify those resources still match when admitting the next run;
this string does not discover or verify the environment automatically.
Keep input assets immutable during measurement. Then resume with:

```bash
python -m benchmarks.benchmarker.restage_campaign --spec campaign.json --output campaign-results --resume
```

Resume compares the recorded identity, complete trial options, workload/SLO,
candidate contents and saved candidate snapshots, local reference-audio
hashes, and quality-config contents. A mismatch stops before any trial is
launched. Campaigns created without an identity cannot be resumed by adding
one afterward. Only one process may write a results directory at a time.

`completed-trials.json` is replaced atomically after each completed trial;
resume skips those cells and rebuilds `trials.jsonl` from that checkpoint.
The checkpoint includes completed but infeasible evaluations. An unfinished
cell is rerun in an `-attempt-00001` directory, preserving its previous raw
files. A trial that finished but was interrupted before checkpointing may be
rerun; partial measurements are never promoted to completed evaluations.
`failure.json` retains the last execution failure as history even after a
later successful resume produces `selection.json`.

### ASR campaigns

Use the same campaign command with `"task": "asr"` in the spec (`tts` is
the default). ASR `trial_options` supplies `model_path`, `port`, `lang`,
`max_wer`, `samples`, and `slo`; optional fields include `stream`, `warmup`
and the startup/request timeouts. It does not need `asr_config_path` or a
second quality model. Each sample's `ref_audio` is the input clip and
`ref_text` is its ground-truth transcript; `target_text` is unused for ASR.

For example, an ASR SLO can use `{"max_latency_s": 5, "max_rtf": 1,
"min_good_fraction": 0.99}`. Choose thresholds for the intended workload;
these example values are not validated model limits. Per-request WER must
also meet `max_wer`. Failed requests remain failed even if their text matches.
Scoring uses the existing ASR normalizer and preserves normalized transcripts
and edit counts in `quality-detail.json`.

ASR supports latency and RTF constraints. Audio TTFA and playback-underrun
constraints are rejected because the response is text. Streaming text TTFT
is recorded by the sender but is not yet a selection constraint. This adapter
has CPU contract coverage; real-model ASR candidate measurements remain pending.

## Source method and remaining work

This integration follows `yl3469/sglang-omni` branch
`lisa/restage-planner-polish` at
`ca4c87ab93ad68d63320b4b199308c7364cb8b0e`: the workload law, fixed-budget
enumeration, process replication and provenance distinction are inherited
from that work. Runtime configuration uses current SGLang-Omni process and
placement compilation.

Historical service-law constants retain their original hardware and stack
labels. They are not calibration for a new system. Per-process capacity,
sharing interference, SLO-constrained measurements, SM binding and model
coverage remain necessary before recommending a topology.


## Fit isolated service measurements

`fit_service_law` in `calibration.py` fits the inherited four-term service
model using nonnegative least squares. Supply accepted isolated measurements
with request IDs, context tokens, audio seconds and service seconds. The
context and duration probes must identify all four terms; constant or
confounded workloads are rejected. Supply independent validation requests to
retain per-request predictions and a held-out error separate from fit error.
The returned dataclass can be serialized with `dataclasses.asdict`.

The caller supplies provenance and separately measured stall parameters.
This API does not collect model measurements, verify provenance against a
running server or infer saturated serving capacity from serial latency.
Those integrations remain necessary before prediction can rank new-hardware
placements.
