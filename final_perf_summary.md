# Final perf summary — async decode (one-step lookahead) on Higgs TTS

Terminal numbers for the PR. Methodology: greedy (`temperature=0`, deterministic
via the batched-sampler argmax short-circuit), fixed `max_new_tokens=128`, 9 kept
runs/config × 20 req (warmup discarded), Student-t per-config CIs + Welch
OFF-vs-ON delta. Off by default (`--enable-async-decode` /
`SGLANG_OMNI_ENABLE_ASYNC_DECODE=1`); the fast path keeps bs=1 on the sync path
(`async_decode_min_batch_size`, default 2).

## bs=4 (the win) — statistically firm

| | mean ± std (ms) | 95% CI | throughput | query_hit |
|---|---|---|---|---|
| OFF | 751.1 ± 8.6 | [744.5, 757.7] | 4.910 ± 0.058 req/s | — |
| ON | 715.9 ± 9.4 | [708.7, 723.1] | 5.217 ± 0.068 req/s | 100% |

**ON is −4.7% latency (95% CI [−5.9%, −3.5%], p≈4e-7) and +6.3% throughput**
(per-config CIs disjoint). Unaffected by the fast path / fix1 / fix2 (bs≥2 always
uses the lookahead path). This is the headline.

## bs=1 — ≈ neutral, below the measurement noise floor

The fast path routes bs=1 to the plain synchronous step (`run_batch`, identical
to async-OFF). Structurally its latency therefore equals OFF up to the event-loop
branch (tens of µs). Empirically the OFF-vs-ON delta is **dominated by
between-server noise (~±1.5%)** on the shared node — the same bs=1 code measured
−0.57% and −2.16% across two sessions — so a precise <0.5% figure is **not
reliably resolvable** with separate-server benchmarking here. Best statement:
**bs=1 ≈ break-even (no meaningful regression), off by default regardless.**
(Without the fast path, bs=1 lookahead showed a small ~1.1% regression; the fast
path removes the lookahead overhead so bs=1 falls back to the OFF path.)

## Correctness

`output_codes` **bit-identical OFF vs ON, bs=1 and bs=4, 100/100 each** (greedy).
bs=4 verify also exercises the fast-path drain transitions (requests finishing
mid-batch drop bs to 1 intermittently). 41 unit tests pass.

## Plan B (multi-stream)

Not worth doing on Higgs — `stall_analysis.md` (decode-isolated nsys) shows decode
is CPU-dispatch-bound (GPU ~1.3% busy, ~3.8 ms/step non-overlappable next-step
CPU prepare), which a multi-stream D2H cannot recover (Plan A already overlaps the
collect). The next real lever is the per-step CPU path, a separate effort.

## PR perf claim (final wording)

> Async decode (one-step lookahead), off by default. At **bs=4** it reduces
> end-to-end latency by **~4.7%** (95% CI [3.5%, 5.9%]) and improves throughput by
> **~6%**; **bs=1** is break-even (a fast path routes single-request decode to the
> plain synchronous step, so it carries no lookahead overhead). output_codes are
> bit-identical OFF vs ON at both batch sizes. The gain scales with batch size
> because the overlapped per-step host collect grows with the number of requests.
> A decode-isolated nsys profile shows the residual per-step GPU idle is CPU
> scheduler/prepare work, not D2H, so a multi-stream follow-up would not add to
> this on Higgs.
