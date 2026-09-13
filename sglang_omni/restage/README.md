# Restage integration

Restage plans stage placement, process replicas and resource configuration,
then measures the exported candidates. `sgl-omni autotune plan` exports
**unmeasured candidates**; `sgl-omni autotune run` measures them and selects a
winner within the tested configurations and load grid, subject to quality and
SLO checks.

## Generate candidates

Use a serving YAML that names the model's `config_cls` and `model_path`, then:

```bash
sgl-omni autotune plan \
  --config pipeline.yaml \
  --search-space examples/configs/restage_qwen3_tts_search.json \
  --output restage-plan \
  --max-candidates 256
```

Use device IDs in the same visible-device namespace as the serving process.
`replica_counts` lists the replica choices for each GPU process, either as one
list applied to every process or as a mapping from process name to its choices.
`examples/configs/restage_moss_td_pd_search.json` uses the mapping form to vary
MOSS-TD decode replicas over 1/2/3 on two GPUs while Prefill stays single.

Each named dimension contains alternative groups of ordinary serving CLI
overrides. Related changes, such as `thinker.tp_size` and `thinker.gpu`, belong
in the same group; different dimensions are combined. Only use TP sizes
supported by the model; schema acceptance alone does not establish runtime
support.

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

## Measure candidates

```bash
sgl-omni autotune run --spec campaign.json --output campaign-results
```

The campaign spec contains `configs` (candidate key to YAML path), `baseline`
(one of those keys), `rates`, `repeats`, `arrival_seed`, and `trial_options`.
A spec may instead use `"plan_directory": "plan"` with `baseline` naming an
accepted candidate file stem such as `candidate-00000`; the directory is
relative to the campaign JSON and every accepted candidate is measured, with
the baseline first. All candidates share one workload, SLO and load grid, and
each repeat uses the same seeded arrival sequence across candidates.
Configurations are copied into the results directory before measurement; use
absolute model paths so the copies resolve the same checkpoint.

`selection.json` ranks candidates by the highest prefix of the tested rate grid
that passed every repeat, then median goodput at that prefix endpoint. An
isolated passing point above a failed lower rate remains visible as
`best_tested_rate`, but does not raise the `passing_prefix_rate` used for
ranking. Exact ties retain the baseline. `recommended.yaml` is written only
when a candidate has a passing rate. These are empirical comparisons within the
measured space, not confidence bounds or global optimality claims. Failed trial
execution stops the campaign and records `failure.json`; it is not silently
converted into a low performance score.

Candidates are measured sequentially in the recorded order. The campaign does
not randomize/interleave candidate order or reuse a serving process across load
points.

### Adapt the measured load grid

Add an optional `adaptive_search` object to the campaign spec:

```json
{
  "max_rate": 256,
  "growth_factor": 2,
  "target_arrival_duration_s": 120
}
```

The campaign measures every candidate at each initial `rates` entry, then
increases the common rate while at least one candidate passes every repeat. It
stops when every candidate fails the latest rate or `max_rate` is reached.
Every candidate retains the same measured grid and paired arrival seeds. There
is no binary refinement or inference about rates between measured points.

`target_arrival_duration_s` increases `corpus_repeats` to supply at least
`rate * duration` requests, rounded to a whole corpus. This targets the nominal
arrival window; Poisson arrivals and final draining change actual duration.
Repeated inputs are not new independent data and may change cache behavior.

Each `rate-*` directory is a regular campaign with its own trial evidence. The
parent records `adaptive-campaign.json`, `adaptive-search.json` with the stop
reason, and the aggregate `selection.json`. Execution errors stop the run
rather than becoming performance failures. A failed SLO point is
not a mathematical upper bound on capacity, and a passing maximum rate leaves
the boundary unmeasured.

### Resume an interrupted campaign

Set `run_identity` in the original spec to a fixed identifier for the source
revision, runtime image, GPU hardware and model snapshots. The caller must
verify those resources still match when admitting the next run; this string
does not discover or verify the environment. Then resume with:

```bash
sgl-omni autotune run --spec campaign.json --output campaign-results --resume
```

Resume compares the recorded campaign identity and skips trials already listed
in `completed-trials.json`, which is replaced atomically after each completed
trial. An unfinished cell reruns in a new `-attempt-00001` directory,
preserving its previous raw files. Partial measurements are never promoted to
completed evaluations.

### Workload options

`trial_options` contains the single-trial fields below except `config_path`,
`rate`, and `arrival_seed`.

`warmup_sample` is a separate sample object repeated for the configured
`warmup` count before timing starts; omitting it keeps the existing
first-sample warmup, and a zero count disables warmup. For new-audio ASR
comparisons, choose a warmup waveform outside the measured set. Warmup results
are excluded from request records, quality evaluation and the SLO denominator.

`corpus_repeats` (default `1`) sends the supplied corpus in order that many
times. Repeated requests receive unique IDs; `workload.json` records their
original sample IDs in `request_sources`. This increases request count, not
distinct input count, and does not clear caches.

For Qwen3-ASR repeated-audio measurements, `asr.factory.pre_lm_cache_size_bytes: 0`
disables storage of completed encoder results while preserving the encoder
worker and batching. In-flight requests with the same audio fingerprint can
still share one encoding, and this does not disable radix KV reuse or
preprocessing caches.

### Explore MPS client limits

`examples/configs/restage_qwen3_tts_mps_search.json` uses the existing native
MPS runtime and stage environment defaults. It puts preprocessing, the TTS
engine and vocoder in separate processes, and compares MPS off, MPS on at
100/100, and client percentages of 50/50, 75/25 and 25/75. Keeping an uncapped
MPS-on control separates MPS scheduling effects from the effect of caps.

The example sets both `tts_engine.gpu_memory_fraction` and
`tts_engine.engine.mem_fraction_static`. The former declares a placement
budget; the latter reaches SGLang through `server_args_overrides`, which
otherwise keeps its 0.85 default. These settings are not hard memory
isolation; verify actual allocation before treating colocated candidates as
feasible.

Use `mps=on` for capped trials: auto mode can leave single-client GPUs without
MPS. Stage env values are defaults, so remove an inherited
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` when testing these choices.
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` limits available client execution
resources; it does not reserve an exclusive SM partition. Keep daemon limits
and per-context partition settings fixed across comparisons, and record them.

## Measure a single TTS or Omni trial

After allocating the visible GPUs, run one candidate with a JSON trial spec:

```bash
python -m benchmarks.benchmarker.restage_tts --spec trial.json --output trial-results
```

The spec supplies `config_path`, `model_path`, `asr_config_path`,
`asr_model_path`, `port`, `rate`, `lang`, `max_wer`, `slo`, and `samples`.
Each sample has `sample_id`, `ref_text`, `ref_audio`, and `target_text`.
Config paths are relative to the spec; audio paths should be absolute.
Optional `sender_options` are the existing TTS sender options, such as
`stream`, `voice`, `task_type`, and `no_ref_audio`. `warmup` defaults to one
request.

The trial launches TTS, records an open-loop arrival cohort, stops TTS, then
launches the configured ASR model on the same allocated devices and port. ASR
does not run during the timed TTS cohort, and both services clean up only their
own process group. The caller must keep the GPU allocation available throughout
the trial.

Outputs include the workload and sender settings, incrementally saved raw
request records, service logs, the ASR quality protocol, per-request audio/WER
evidence and joint SLO results. Missing or failed transcription is not quality
success. WER measures transcript agreement; speaker similarity and perceptual
quality require separate evaluation.

For an Omni model's read-aloud workload, set `api` to `chat` in the trial spec
or campaign `trial_options`. This reuses the existing Omni SeedTTS sender at
`/v1/chat/completions` with its required `sender_options`: `voice_clone`,
`speaker`, `max_tokens`, `temperature`, and `stream` (`system_prompt` is
optional). The audio must agree with the full `target_text` under the same WER
check. This covers speech generation through the chat API, not arbitrary
multimodal conversation. The chat sender records first-audio time, per-chunk
PCM duration and maximum playback underrun for streaming responses.

The default `api` is `speech`, preserving `/v1/audio/speech` trials. For either
API, `max_ttfa_s` and `max_underrun_s` require `sender_options.stream=true`,
because non-streaming responses provide no first-audio timestamp. Fewer than
two chunks leaves continuity unmeasured; such a request does not pass an
explicit playback constraint.

## ASR campaigns

Use the same campaign command with `"task": "asr"` in the spec (`tts` is the
default). ASR `trial_options` supplies `model_path`, `port`, `lang`,
`max_wer`, `samples`, and `slo`; optional fields include `stream`, `warmup`
and the startup/request timeouts. It does not need `asr_config_path` or a
second quality model. Each sample's `ref_audio` is the input clip and
`ref_text` is its ground-truth transcript; `target_text` is unused for ASR.

For example, an ASR SLO can use `{"max_latency_s": 5, "max_rtf": 1,
"min_good_fraction": 0.99}`. Choose thresholds for the intended workload;
these example values are not validated model limits. Per-request WER must also
meet `max_wer`. Failed requests remain failed even if their text matches.
Scoring uses the existing ASR normalizer and preserves normalized transcripts
and edit counts in `quality-detail.json`.

ASR supports latency and RTF constraints. Audio TTFA and playback-underrun
constraints are rejected because the response is text. Streaming text TTFT is
recorded by the sender but is not yet a selection constraint.

## Installed command

The wheel includes the shared `benchmarks` package used by Restage, so both
subcommands are available after installing SGLang-Omni and preparing the model
assets. Local reference-audio and configuration paths are resolved relative to
the spec; model identifiers and supported HTTP/data/file media references
remain unchanged. Required local audio and ASR configuration files are checked
before launching a model. Hardware admission remains the caller's
responsibility.

## Source method

This integration follows `yl3469/sglang-omni` branch `lisa/restage-planner-polish`
at `ca4c87ab93ad68d63320b4b199308c7364cb8b0e`: the fixed-budget enumeration,
process replication and provenance distinction are inherited from that work.
Runtime configuration uses current SGLang-Omni process and placement
compilation. Per-process capacity prediction, sharing interference and
SM binding remain outside this integration.
