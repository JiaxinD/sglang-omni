# Phase 4 — Async Decode Benchmark Results (Higgs TTS)

Hardware: ion9 H200 (single GPU). Server: `sglang_omni.cli serve --config
examples/configs/higgs_tts.yaml`, async OFF (`_event_loop_normal`) vs ON
(`_event_loop_async_decode`, via `SGLANG_OMNI_ENABLE_ASYNC_DECODE=1`).
Driver: `scripts/benchmark_async.py`, 20 requests/config, `max_new_tokens=128`,
production sampling. Metric: end-to-end `/v1/audio/speech` latency (prefill +
decode + vocoder + transport). `query_hit` = fraction of decode steps where
`execute_resolve` found the CUDA event already done (overlap engaged) vs had to
block — captured via `scripts/bench_inject`.

## Results

| config | mean ms | p50 ms | p99 ms | req/s | query_hit |
|---|---|---|---|---|---|
| Higgs bs=1 OFF | 554 | 560 | 795 | 1.81 | — (sync) |
| Higgs bs=1 ON  | 555 | 538 | 804 | 1.80 | **100% (2038/2038)** |
| Higgs bs=4 OFF | 905 | 901 | 1456 | 4.06 | — (sync) |
| Higgs bs=4 ON  | — | — | — | — | **crashed** (see below) |

(bs=1 OFF/ON across two runs: OFF 554–564 ms, ON 548–555 ms — the OFF↔ON gap is
within run-to-run noise from production sampling's variable output lengths.)

## Honest assessment

**The overlap mechanism works, but e2e latency is neutral at bs=1.**
- `query_hit = 100%` (2038 decode steps, 0 fallbacks): every step, when
  `execute_resolve` consumes step N-1, the CUDA event is already done — the
  CPU collect/output work IS running concurrently with step N's GPU forward,
  exactly as designed (launch-first, single-stream, `event.query()`).
- Yet bs=1 e2e latency is unchanged (555 vs 554 ms, within noise). **The ~1.1ms
  per-step "CPU bubble" the profile flagged (cuda_runtime > kernel gap) is CPU
  time that was already overlapping GPU compute in the synchronous path** (CUDA
  async kernel dispatch + CUDA-graph replay already let the CPU run ahead). It
  was not a serial wall-time stall, so hiding it explicitly behind the forward
  recovers ~no wall time. This matches the PR #572 finding that batching the
  per-step D2H 3→1 was also **latency-neutral** — the gap is CPU-side overhead,
  not GPU idle.

**Applicability bound.** This optimization helps only when the per-step CPU
work is a *serial* wall-time stall longer than what CUDA async dispatch already
hides — i.e. when `forward` time is short relative to the CPU bubble AND the CPU
genuinely stalls the GPU. On Higgs decode with CUDA Graph (forward ≈ 3.72 ms,
CPU already overlapped), that condition does not hold, so the gain is ≈0. The
task's projected ~22% (small model, bs=1) assumed the CPU bubble was a serial
stall; measured `query_hit=100%` + neutral latency shows it was not, here.

**Could not measure (honest gaps):**
- **bs=4 ON crashed**: `replay_prepare` size mismatch (tensor 3 vs 4) when a
  request finishes mid-batch under the launch-first lag — a remaining bs>1
  (concurrent-request) batch-composition bug. async **OFF** bs=4 is fine (4.06
  req/s); the bit-identical correctness gate is bs=1 only. bs>1 needs more work
  (see phase3_summary.md). So no async bs=4 number.
- **Llama-3-8B (bs=1/8, olen=512) — N/A by design**: this optimization lives in
  the omni `ModelRunner` + `OmniScheduler`; plain Llama-3-8B is served by
  upstream sglang directly (which already has its own FutureMap overlap
  scheduler) and never flows through this code path, so the flag is inert for
  it. Reported as N/A rather than fabricated.
- **No isolated decode-only / nsys re-profile**: latency here is end-to-end
  (prefill + vocoder dominate part of it). `query_hit=100%` is the direct
  evidence the overlap engages; a decode-isolated torch.profiler/nsys trace to
  quantify recovered GPU-idle time per step was not run.

## Bottom line

Mechanism: ✅ correct and engaging (100% overlap, output_codes bit-identical
OFF vs ON over 200 prompt×run pairs). Wall-time payoff on Higgs bs=1: ≈0 — the
targeted CPU bubble was already hidden by CUDA async execution. Recommend: (1)
fix the bs>1 path and (2) before investing further, run a decode-isolated nsys
profile to confirm whether ANY config has a serial CPU stall this can recover —
if not, this is best framed (like PR #572) as a correctness/structure cleanup
that *enables* the multi-stream follow-up (Plan B), not a standalone speedup.
