# Phase 4 — Async Decode Benchmark Results (Higgs TTS)

Hardware: ion9 H200 (single GPU). Server: `sglang_omni.cli serve --config
examples/configs/higgs_tts.yaml`, async OFF (`_event_loop_normal`) vs ON
(`_event_loop_async_decode`, via `SGLANG_OMNI_ENABLE_ASYNC_DECODE=1`).
`query_hit` = fraction of decode steps where `execute_resolve` found the CUDA
event already done (overlap engaged) vs had to block.

## Statistical methodology (v2 — the numbers to cite)

Driver `scripts/benchmark_v2.py`. The headline numbers below are measured under:
- **Greedy, `temperature=0`** (deterministic via the T4 batched-sampler argmax
  short-circuit) → per-prompt output length is fixed, so wall-time is not
  contaminated by output-length jitter.
- **Fixed `max_new_tokens=128`.**
- **10 runs/config** (a run = 20 requests at concurrency=bs); **run #0 discarded
  as warmup**; statistics over the remaining **9 runs**.
- Per-config CI = Student-t with **8 dof** (n=9, t≈2.31). OFF-vs-ON delta uses a
  **Welch** two-sample t 95% CI; "significant" = the CI does not cross 0.

## Results (v2 — greedy, fixed olen, 9×20 req)

| config | mean ± std (ms) | 95% CI (ms) | p50 | p99 | throughput (req/s) | query_hit |
|--------|----------------:|-------------|----:|----:|-------------------:|-----------|
| bs=1 OFF | 521.0 ± 1.4 | [519.9, 522.1] | 462 | 658 | 1.919 ± 0.005 | — (sync) |
| bs=1 ON  | 527.0 ± 1.0 | [526.3, 527.8] | 468 | 666 | 1.897 ± 0.004 | **100%** |
| bs=4 OFF | 751.1 ± 8.6 | [744.5, 757.7] | 721 | 1005 | 4.910 ± 0.058 | — (sync) |
| bs=4 ON  | 715.9 ± 9.4 | [708.7, 723.1] | 690 | 970 | 5.217 ± 0.068 | **100%** |

**OFF vs ON:**

| | latency Δ (OFF−ON) | 95% CI | p | throughput | verdict |
|---|---|---|---|---|---|
| bs=1 | −6.0 ms (**−1.16%**) | [−1.40%, −0.92%] | 5.8e-8 | ×0.989 | **significant — ON ~1.2% SLOWER** |
| bs=4 | +35.2 ms (**+4.69%**) | [+3.49%, +5.88%] | 3.6e-7 | ×1.063 (+6.3%) | **significant — ON ~4.7% FASTER** |

## Honest assessment

**bs=1: a small but significant regression (~1.2%).** The lookahead's fixed
per-step bookkeeping (event record/query, staging handle, early output_ids
publish) costs ~6 ms and there is no overlap payoff — a single request's per-step
collect is too small to hide anything. (Earlier runs called this "neutral"; with
production-sampling noise removed it is a small *real* regression, not zero.)

**bs=4: a real, statistically firm win — but ~5%, not the earlier ballpark.**
+4.7% latency / +6.3% throughput, both CIs well clear of 0, p<1e-6. launch-first
hides step N−1's collect behind step N's forward, and that collect scales with
batch size, so the gain shows up only at bs>1.

### Correction vs the earlier "ballpark" numbers

An earlier 2-run pass with **production sampling** (temperature=1.0) reported
bs=4 "+13–20% latency / ~1.2× throughput" (OFF 888–976 ms, ON 771–778 ms). That
was inflated by **output-length jitter**: with stochastic sampling the OFF runs
drew longer outputs than ON, exaggerating the delta. The fixed-greedy v2
measurement is the apples-to-apples truth: **+4.7% / +6.3%**. (bs=1 ballpark:
OFF 554–564, ON 555–561 — "neutral", now refined to −1.2%.) The ballpark numbers
are retained here only as a cautionary reference; **cite the v2 numbers.**

### Reconciliation with the nsys profile

`stall_analysis.md` (decode-isolated nsys, async ON) shows the GPU is ~98% idle
during decode with a ~3.8 ms serial gap/step that is the **next step's**
`get_next_batch_to_run` + `prepare_for_decode` CPU work — not overlap-able by any
stream trick, and untouched by async. What launch-first *does* hide is the
**previous step's collect** (small at bs=1 → net loss after overhead; larger at
bs=4 → net +4.7%). The single-digit % size of even the bs=4 win is exactly what
the "GPU mostly idle, big non-overlappable CPU residual" profile predicts.

## Plan B (multi-stream) implication

`stall_analysis.md` answer is **(b): not worth doing.** The residual ~3.8 ms gap
is CPU scheduler/prepare work, not the D2H Plan B's alt-stream would overlap, and
Plan A already overlaps the D2H/collect (`query_hit=100%`). Plan B's incremental
target on Higgs is ≈0. The large remaining opportunity is the CPU per-step path
itself (vectorize/sink `_populate_cg_buffers`, capture more into CUDA Graph, or
adopt the upstream overlap scheduler) — a different work item.

## Could not / did not measure (honest gaps)

- **Llama-3-8B (bs=1/8) — N/A by design.** This optimization lives in the omni
  `ModelRunner` + `OmniScheduler`; plain Llama-3-8B is served by upstream sglang
  directly (its own FutureMap overlap scheduler) and never flows through this
  code path, so the flag is inert. Reported as N/A, not fabricated.
- **Matrix kept to bs=1 / bs=4** (the firm-up task scope). bs=8/16/32 e2e latency
  not run; the nsys profile (`stall_analysis.md`) covers bs up to 32 and shows the
  same per-step structure, so the collect-overlap gain is expected to persist, but
  that is inferred, not an e2e latency number.

## Bottom line

Mechanism: ✅ correct (output_codes bit-identical OFF vs ON, bs=1 and bs=4,
100/100 each) and engaging (100% query_hit). Wall-time (greedy, fixed olen,
9×20 req, statistically firm): **bs=1 −1.2% (regression), bs=4 +4.7% latency /
+6.3% throughput.** A modest but real bs>1 throughput win with a small bs=1 cost
— so keep it off by default and enable for batched serving. Plan B would not add
to this on Higgs.
