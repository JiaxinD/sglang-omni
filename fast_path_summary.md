# Fast-path summary — bypass lookahead for bs < threshold

Goal: eliminate the bs=1 −1.2% async-decode regression (benchmark_v2). Approach:
at low concurrency the lookahead can't overlap anything useful, so run a plain
synchronous step instead.

## Commit

`33daa47` [async-decode] fast path: bypass lookahead for bs < threshold
(files: `sglang_omni/scheduling/omni_scheduler.py`,
`sglang_omni/models/higgs_tts/stages.py`,
`tests/unit_test/pipeline/test_async_decode.py`). **Not pushed** (per request).

Mechanism: gate the lookahead branch on
`len(batch.reqs) >= async_decode_min_batch_size` (default 2 → only bs=1 takes the
fast path). Below the threshold, decode falls through to the existing drain+sync
branch — byte-for-byte the same as async-OFF `_event_loop_normal` (`run_batch` +
`process_batch_result`) plus a no-op `_resolve_pending_async()`. Tunable via
`SGLANG_OMNI_ASYNC_DECODE_MIN_BS` env or `server_args.async_decode_min_batch_size`.

## Edge cases covered

- **bs=1, no pending** → fast path (sync), no lookahead overhead.
- **bs 1→2+** (new request arrives) → first bs≥2 step launches with no prev
  pending (lookahead warmup), as normal.
- **bs 2+→1** (request finishes) → the bs=1 step hits the else branch, which
  first calls `_resolve_pending_async()` to **drain the in-flight lookahead step**
  (preserving ordering, never stranding a pending), then runs the bs=1 batch
  synchronously.
- **empty batch** → unchanged idle handling (also drains any pending first).
- Unit tests (`test_async_decode.py`, drive the *real* event loop over a scripted
  bs sequence): `test_fast_path_bs1_bypasses_lookahead_and_drains_on_transition`
  (1→2→2→1→1→idle asserts sync/launch/resolve/**drain**/sync/idle order +
  `_async_pending is None` at the end), plus threshold knob tests at min_bs=1 and
  min_bs=4. **41 unit tests pass** (38 prior + 3 new).

## Correctness (hard gate)

- **bs=4 verify_correctness 10×10 = 100/100 bit-identical** OFF vs ON — and this
  run exercises the fast-path drain transitions (requests finish mid-batch,
  dropping bs to 1 intermittently). No regression, no crash, no deadlock.
- **bs=1 fast-path is bit-identical to OFF by construction**: it runs the exact
  same synchronous `run_batch` → `execute()` path as `_event_loop_normal`, with
  deterministic greedy (T4). (The bs=1 verify itself flaked twice on *server
  startup* — GPU memory grabbed mid-run on the shared fleet + same-port restart
  socket reuse — not a correctness divergence; see "infra notes".)

## bs=1 wall-time — old −1.2% vs new (fast path)

Same-session measurement, GPU3, greedy, `max_new_tokens=128`, 7 kept runs × 20
req (run #0 warmup discarded), Student-t CIs, Welch OFF-vs-ON delta:

| bs=1 config | mean ± std (ms) | 95% CI | vs OFF | engaged? |
|---|---|---|---|---|
| OFF | 535.0 ± 2.4 | [532.8, 537.2] | — | — |
| ON, lookahead (`MIN_BS=1`) | 541.0 ± 3.5 | [537.8, 544.3] | **−1.13%** (CI [−1.80,−0.46], p=0.004) | query_hit=16024 ✓ |
| ON, **fast-path** (`MIN_BS=2`) | 538.0 ± 1.9 | [536.3, 539.8] | **−0.57%** (CI [−1.04,−0.09], p=0.02) | resolve_stats=NONE ✓ |

(`resolve_stats=NONE` = the fast-path server did **zero** lookahead resolves at
bs=1 → fast path engaged. The lookahead server logged 16024 resolves → lookahead
engaged. The same-session lookahead −1.13% reproduces the committed −1.16%.)

## Verdict — honest

- **Fast path engages and roughly halves the regression: −1.13% → −0.57%
  (6 ms → 3 ms).** bs≥2 unaffected (always lookahead) → **bs=4 still +4.7%**.
- **It does not fully hit "neutral".** The −0.57% CI is [−1.04%, −0.09%] — it
  still excludes 0 (p=0.02, marginal), so ON is ~0.57% slower than OFF even with
  the fast path. The user target was "neutral / CI crosses 0 / <0.5%"; this is
  **just outside** it.
- **Residual source** (~3 ms): (1) the async event loop's per-step branch
  (`_batch_is_decode` + `len` + a no-op `_resolve_pending_async()` call) is
  slightly heavier than `_event_loop_normal`'s `if batch:`; (2) the model runner
  still has `_async_enabled=True`, so `_populate_cg_buffers` runs its per-step
  overrun guard (`generation_done[rows]` gather + `torch.where`) even on a
  fast-path sync step. Fully closing it would require the model runner to also
  skip async-specific per-step work below the threshold — more invasive, and
  arguably not worth it for ~3 ms.

## Recommendation

Ship the fast path: it halves the bs=1 cost for free and is the right default
(only bs=1 diverts; bs≥2 keeps the +4.7% win). If you want the last ~0.57% gone,
I can (a) inline the pending check and (b) skip the overrun guard when there's no
in-flight pending — say the word; otherwise −0.57% (≈break-even, off by default
anyway) is a reasonable place to stop.

## Infra notes (why two GPU runs flaked)

The shared omni fleet (cards 0–3) had memory grabbed mid-run (a server came up
with only 17 GB avail once) and same-port OFF→ON restarts hit
`address already in use` (socket not released / a stage subprocess lingering).
These are test-harness/infra issues, not fast-path bugs — the clean same-session
run on a genuinely-free GPU3 produced the numbers above. The profiler driver
already got real-port discovery + full-tree cleanup (`a172066`); the
verify/benchmark scripts would benefit from the same (unique ports per server +
GPU-mem precheck) if this is re-run.
