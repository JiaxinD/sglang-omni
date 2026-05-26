# Benchmark v2 — async decode, statistically firm (greedy, fixed olen)

Driver `scripts/benchmark_v2.py`, ion9 H200 single GPU, Higgs TTS. **Greedy
(`temperature=0`, deterministic via the T4 argmax short-circuit), fixed
`max_new_tokens=128`** → per-prompt output length is fixed, so wall-time isn't
contaminated by length jitter (the flaw in the old production-sampling
benchmark). 10 runs/config (run = 20 requests at concurrency=bs); **run #0
discarded as warmup**, stats over the remaining **9 runs**. CI = Student-t, **8
dof** (n=9; note the spec said "9 dof" but 9 samples → dof=8, t≈2.31 — using the
correct value). OFF-vs-ON delta uses a Welch two-sample t CI; "significant" =
the 95% CI does not cross 0. Raw per-run data + JSON: `/tmp/v2_results.json`.

## Final numbers (per config)

| config | latency mean ± std (ms) | 95% CI (ms) | p50 | p99 | throughput (req/s) | query_hit |
|--------|------------------------:|-------------|----:|----:|-------------------:|-----------|
| bs=1 OFF | **521.0 ± 1.4** | [519.9, 522.1] | 462 | 658 | 1.919 ± 0.005 | — |
| bs=1 ON  | **527.0 ± 1.0** | [526.3, 527.8] | 468 | 666 | 1.897 ± 0.004 | 100% (20030/20030) |
| bs=4 OFF | **751.1 ± 8.6** | [744.5, 757.7] | 721 | 1005 | 4.910 ± 0.058 | — |
| bs=4 ON  | **715.9 ± 9.4** | [708.7, 723.1] | 690 | 970 | 5.217 ± 0.068 | 100% (5777/5777) |

Raw per-run latency means (ms), runs 1–9 (warmup excluded):
- bs1 OFF: 522.0, 520.3, 524.0, 521.7, 521.3, 520.5, 519.1, 519.9, 520.2
- bs1 ON : 528.8, 528.1, 527.3, 527.6, 526.8, 526.4, 525.9, 525.9, 526.5
- bs4 OFF: 742.2, 759.5, 766.6, 753.0, 755.0, 753.0, 743.0, 742.8, 744.7
- bs4 ON : 720.9, 733.9, 721.7, 719.9, 704.4, 704.1, 712.2, 712.1, 714.0

No outliers excluded (none present — variance is tiny and runs are tightly
clustered; the only notably slow run was #0, the warmup, discarded by protocol).

## OFF vs ON (the headline)

| | latency Δ (OFF−ON) | 95% CI | p | throughput ratio (ON/OFF) | verdict |
|---|---|---|---|---|---|
| **bs=1** | **−6.0 ms (−1.16%)** | [−1.40%, −0.92%] | 5.8e-8 | 0.989 (−1.1%) | **significant — ON is ~1.2% SLOWER** |
| **bs=4** | **+35.2 ms (+4.69%)** | [+3.49%, +5.88%] | 3.6e-7 | 1.063 (+6.3%) | **significant — ON is ~4.7% FASTER** |

(Throughput: per-config 95% CIs are disjoint at bs=4 — OFF [4.87, 4.96] vs ON
[5.17, 5.27] — so the +6.3% throughput gain is significant too.)

## One-line conclusions

- **bs=1: significantly SLOWER by 1.2%** (CI [−1.4%, −0.9%], p≈6e-8). The
  lookahead's fixed per-step bookkeeping (event record/query, staging handle,
  publishing output_ids early) costs ~6 ms and there's no overlap payoff — the
  single-request collect is too small to hide anything. (This refines the earlier
  "neutral": with the noise removed it's a small, real regression, not zero.)
- **bs=4: significantly FASTER by 4.7% latency / +6.3% throughput** (CI
  [+3.5%, +5.9%], p≈4e-7). launch-first hides step N−1's collect behind step N's
  forward, and that collect scales with batch size, so the gain appears at bs>1.

## Does this support a PR claim?

**Yes, but the honest number is ~5%, not the old ballpark.** The PR can state:

> "At bs=4, async decode reduces end-to-end latency by **4.7%** (95% CI
> [3.5%, 5.9%]) and improves throughput by **~6%** (greedy, olen=128, 9×20
> requests); bs=1 regresses ~1.2%."

It should **NOT** repeat the earlier "+13–20% / ~1.2× throughput" — that was a
production-sampling artifact: with temperature=1.0 the OFF runs happened to draw
longer outputs (888–976 ms spread) than ON, inflating the apparent delta. Under
fixed greedy length the apples-to-apples gain is +4.7% / +6.3%, and it is
statistically firm (tiny variance, p<1e-6).

## Caveat on framing

The feature is now a **small but real bs>1 throughput win with a small bs=1
regression** — so it genuinely wants to stay **off by default** (current
behavior) and be enabled for batched/throughput-oriented serving. This is
consistent with `stall_analysis.md`: decode is CPU-dispatch-bound (GPU ~1.3%
busy), the overlap recovers only the part of the per-step CPU work that scales
with batch (the collect), and the larger residual (~3.8 ms/step next-step
prepare) is untouched — which is why even the bs=4 win is single-digit %.
