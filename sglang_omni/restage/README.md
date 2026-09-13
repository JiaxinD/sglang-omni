# Restage integration

Restage plans stage placement, process replicas and resource configuration.
The current integration exports **unmeasured candidates**. It does not yet
run calibration, benchmark candidates, enforce performance SLOs or select a
validated winner.

## Generate candidates

Use a serving YAML that names the model's `config_cls` and `model_path`, then:

```bash
sgl-omni autotune plan \
  --config pipeline.yaml \
  --search-space examples/configs/restage_qwen3_tts_search.json \
  --output restage-plan \
  --max-candidates 256
```

The example varies Qwen3-TTS engine memory budgets on two visible GPUs. Its
values are search choices, not recommended memory settings. Use device IDs
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
resume interrupted runs. Prediction-based pruning and held-out calibration
validation remain separate work.

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
