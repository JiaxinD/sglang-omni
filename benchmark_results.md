# Phase 4 — Async Decode Benchmark Results (Higgs TTS)

Hardware: ion9 H200 (single GPU). Server: `sglang_omni.cli serve --config
examples/configs/higgs_tts.yaml`, async OFF (`_event_loop_normal`) vs ON
(`_event_loop_async_decode`, via `SGLANG_OMNI_ENABLE_ASYNC_DECODE=1`).
Driver: `scripts/benchmark_async.py`, 20–24 requests/config, `max_new_tokens=128`,
production sampling. Metric: end-to-end `/v1/audio/speech` latency (prefill +
decode + vocoder + transport). `query_hit` = fraction of decode steps where
`execute_resolve` found the CUDA event already done (overlap engaged) vs had to
block — captured via `scripts/bench_inject`.

## Results

bs=4 now runs (was a crash before the launch-first output_ids fix, commit
`fbeb5b0`). Two independent runs per batch size; production sampling's variable
output lengths give the OFF baseline run-to-run spread, so a range is shown.

| config | mean ms | p50 ms | p99 ms | req/s | query_hit | async Δlatency / throughput |
|---|---|---|---|---|---|---|
| Higgs bs=1 OFF | 554–564 | ~570 | ~800 | 1.77–1.81 | — (sync) | — |
| Higgs bs=1 ON  | 555–561 | ~585 | ~820 | 1.78–1.80 | **100%** | **+0.5% (neutral), 1.005x** |
| Higgs bs=4 OFF | 888–976 | ~900 | ~1380 | 3.83–4.27 | — (sync) | — |
| Higgs bs=4 ON  | 771–778 | ~820 | ~1200 | 4.77–4.81 | **100%** | **+13–20% faster, 1.13–1.25x** |

## Honest assessment

**The async overlap is latency-neutral at bs=1 but a real win at bs>1.**

- `query_hit = 100%` at both batch sizes (every step, `execute_resolve` finds the
  event already done — the host collect ran concurrently with the next forward,
  exactly as launch-first intends).
- **bs=1: neutral** (+0.5%, within noise). The per-step host collect for a single
  request is tiny, so hiding it behind the forward recovers ~nothing. This matches
  the earlier bs=1 finding and PR #572 (batching the per-step D2H 3→1 was also
  neutral).
- **bs=4: ~13–20% faster, ~1.13–1.25× throughput.** The per-step collect scales
  with the number of requests (the Python collect loop, per-req sampler-state
  scatter, D2H of a bigger snapshot). At bs=4 that work is substantial, and
  launch-first overlaps step N-1's collect behind step N's forward, so the saving
  is real and grows with batch size. **This is the regime that matters for
  throughput**, and the feature delivers there.

### Why the gain is bs-dependent — reconciled with the nsys profile

`stall_analysis.md` (decode-isolated nsys, async ON) shows the GPU is ~98% idle
during decode with a ~3.8 ms serial gap per step *regardless of batch size*. That
residual gap is the **next step's** `get_next_batch_to_run` + `prepare_for_decode`
CPU work, which happens before the next forward is even enqueued and so cannot be
hidden behind the current forward by any stream trick. What launch-first *does*
hide is the **previous step's collect**; that collect is small at bs=1 (neutral)
and large at bs=4 (the win). The two measurements agree:

- Overlap-able work (collect) → scales with bs → the bs=4 speedup.
- Non-overlap-able work (next-step prepare, ~3.8 ms) → the residual GPU idle that
  remains even with async ON.

## Plan B (multi-stream) implication

`stall_analysis.md` answer is **(b): not worth doing.** The residual ~3.8 ms gap
is CPU per-step scheduler/prepare work, not the D2H that Plan B's alt-stream would
overlap — and Plan A already overlaps the D2H/collect (`query_hit=100%`). Plan B's
incremental target on Higgs is ≈0. The large remaining opportunity is the CPU
per-step path itself (reduce Python per-step work / capture more into CUDA Graph /
adopt the upstream overlap scheduler), which is a different work item.

## Could not / did not measure (honest gaps)

- **Llama-3-8B (bs=1/8, olen=512) — N/A by design.** This optimization lives in
  the omni `ModelRunner` + `OmniScheduler`; plain Llama-3-8B is served by upstream
  sglang directly (its own FutureMap overlap scheduler) and never flows through
  this code path, so the flag is inert for it. Reported as N/A, not fabricated.
- **Run count.** Two runs per config (not a large sample); production sampling
  makes per-run output length vary, which is why OFF spreads 888–976 ms at bs=4.
  The ON numbers are tighter (771–778) and the gain direction/magnitude is
  consistent across both runs and mechanistically explained, but this is a
  ballpark, not a tightly-bounded measurement.
- Higher concurrency (bs=16/32) latency not run end-to-end here; the nsys profile
  (`stall_analysis.md`) covers bs up to 32 and shows the same per-step structure,
  so the collect-overlap gain is expected to persist/grow, but that's inferred,
  not an e2e latency number.

## Bottom line

Mechanism: ✅ correct (output_codes bit-identical OFF vs ON, **bs=1 and bs=4**,
100/100 each) and engaging (100% query_hit). Wall-time: **neutral at bs=1, ~13–20%
faster (≈1.2× throughput) at bs=4** — a genuine throughput win in the batched
regime, not just structural cleanup. Plan B would not add to this on Higgs.
