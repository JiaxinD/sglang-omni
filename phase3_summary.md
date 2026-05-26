# Phase 3 Summary — Async Decode (one-step lookahead, single-stream)

Branch `feat/async-decode-lookahead` (off `feat/higgs-cg-batch-d2h`). Default
**off**; opt in via `SGLANG_OMNI_ENABLE_ASYNC_DECODE=1`. The sampling-params
cache was split into its own PR (`sampling_cache_pr_plan.md`).

## Commits

**Planned 6 (the async PR core):**
| # | sha | what | key file:line |
|---|-----|------|---------------|
| 1 | `4d84510` | `_PendingStep` + pinned ping-pong staging buffers (data only) | `base.py` `_PendingStep`, `_next_host_staging` |
| 2 | `d3ec351` | split `execute()` → `execute_launch`/`execute_resolve` + `post_decode_launch`/`post_decode_resolve` hooks; shared `_build_forward_batch`/`_prepare_and_forward`/`_finalize` | `base.py:execute*` |
| 3 | `c848d3a` | migrate Higgs collect to split hooks (`_decode_pack_gpu` + `_decode_collect_host`) | `higgs_tts/model_runner.py` |
| 4 | `ed554a9` | `_event_loop_async_decode` (launch-first) + `_run_batch_launch/resolve` + abort reaches in-flight step | `omni_scheduler.py` |
| 5 | `dab5900` | wire `enable_async_decode` flag (OmniScheduler param + Higgs stages env opt-in) | `omni_scheduler.py`, `higgs_tts/stages.py` |
| 6 | `dd7a447` | unit tests for the lookahead state machine | `tests/unit_test/pipeline/test_async_decode.py` |

**3 follow-on commits (the GPU verify gate caught real launch-first bugs — disclosed honestly):**
| sha | what |
|-----|------|
| `33f45b8` | launch-first correctness: caller-owned pending handle (single slot clobbered N-1); GPU-sourced `output_ids` at launch (input_ids None on step 2); resolved-step `next_token_ids`; overrun KV double-free; length-finish overrun token leak |
| `4e2f595` | EOC-finish overrun: keep the req finishing *at* this step (was filtered pre-emit → hang); skip overrun reqs in stream emit; GPU-side guard routing `generation_done` rows to the padding row (the wasted forward on a done row tripped a device-side gather assert) |
| `42fccad` | `scripts/verify_correctness.py` gate + `verify_inject` + `benchmark_async.py` + `bench_inject` |
| `b000038` | drop overrun reqs via `batch.reqs` trim (not `filter_batch` — `copy()` omits seq_lens); bench-stats only written by the runner-holding stage process |

**3 follow-up commits (bs>1 + sampler determinism — this round):**
| sha | what |
|-----|------|
| `fbeb5b0` | **bs>1 fix**: `execute_resolve` passes `_finalize(set_output_ids=False)` so the lagged resolve stops re-stamping a stale-length output_ids on the live running batch (the replay size mismatch). + regression test. |
| `12dbdfb` | **sampler determinism**: `_sample_independent_batched` short-circuits greedy to argmax (branchless, graph-safe) → reproducible `temperature=0`. + 3 unit tests. |
| `9b70ca9` | tooling: `verify_correctness --concurrency` (bs>1, group by prompt-hash), drop argmax injector, + nsys profiler (`profile_async.py` / `profile_inject`). |

## Tests
- `tests/unit_test/higgs_tts/` + `pipeline/test_scheduler.py` + `pipeline/test_async_decode.py`: **38 pass** (sync path unchanged; async state machine + 3 new greedy-determinism tests + the resolve-output_ids regression assertion).
- `tests/unit_test/pipeline/test_ipc.py`: **7 failures — PRE-EXISTING**, confirmed by running the base-commit (`be93a97`) version (7 fail there too). They concern the mp_runner/launcher IPC, unrelated to async decode; not introduced by this work.
- **verify_correctness (PR merge gate), production greedy (no argmax injector after `12dbdfb`), output_codes bit-identical OFF vs ON:**
  - **bs=1, 10 prompts × 10 runs (max=128): PASS, 100/100 bit-identical.**
  - **bs=4 (concurrency=4), 10 prompts × 10 runs (max=128): PASS, 100/100 bit-identical.** Per-prompt codes are invariant to batch composition AND to async timing (each round of a prompt is identical despite timing-varied batches), and OFF == ON.
  - The bs=1 "audio sha differ" note is benign cross-process vocoder float nondeterminism (cuDNN/cuBLAS algo selection per process); the **output_codes** (what async decode controls) are bit-identical. (bs>1 audio sha isn't compared per-prompt — codes are the gate.)

## Q1 / Q2 (resolved — see `phase3_clarifications.md`)
- **Q1**: lookahead = **1 wasted step**, not 2 (flag staleness is 2 but the finishing step isn't wasted). Invariant 3 unchanged. No loop reorder (resolve-first kills overlap).
- **Q2**: `_FAILED_BATCH_RESULT` is an existing sentinel (`omni_scheduler.py:36`); the async loop reuses it.

## Honest uncertainties / things to flag
1. ~~**CG decode is non-deterministic at `temperature=0`**~~ **FIXED (commit `12dbdfb`).** `_sample_independent_batched` now short-circuits greedy to `argmax` (branchlessly, graph-safe), mirroring the per-row `_sample_independent`. Production `temperature=0` is reproducible, so the gate no longer injects an argmax patch (`verify_inject` keeps only the per-step code dump). New unit tests in `test_batched_step.py` cover temp=0 / top_k=1 / mixed rows.
2. **EOC overrun guard adds a per-step GPU gather** (`generation_done[rows]` + `torch.where`) in `_populate_cg_buffers` when async is on — GPU-side, no host sync. Cheap, but it's extra work each decode step; an alternative is to suppress the launch entirely when all reqs are done (needs the resolve, which lags).
3. ~~**bs>1 (concurrent requests) is NOT yet working in async mode**~~ **FIXED (commit `fbeb5b0`).** Root cause was NOT a batch-trim problem (filter_batch trims correctly): `_finalize` re-published `schedule_batch.output_ids` during the *resolve*, but under launch-first the resolve lags one step and runs on the LIVE running batch whose output_ids the current launch already set at the right length — so it stamped a stale-length output_ids that the next `prepare_for_decode` turned into an input_ids mismatching seq_lens, tripping `replay_prepare` (`input_buffers.py` copy_, "tensor a (2) vs b (3)") once a req finished mid-batch. Fix: `execute_resolve` calls `_finalize(..., set_output_ids=False)`; the launch is the sole output_ids publisher. **Verified: bs=4 verify_correctness 10×10 = 100/100 bit-identical OFF vs ON** (and bs=1 still 100/100 — no regression). The bit-identical claim is now bs=1 AND bs=4.
4. **async-off byte-identical** rests on `execute()` being a pure extraction over shared sub-steps (verified by inspection + 35 tests), not a diff against the pre-refactor binary.
5. **abort during an in-flight step**: the launched step is always resolved (never stranded); the aborted req shares Req objects with `running_batch`; `_async_pending_batch()` is in the abort cleanup tuples. Covered by reasoning + existing abort tests, not exercised end-to-end.

## Scope creep
Production-code changes beyond the 6 planned commits were all **bug fixes** the
GPU gate surfaced (`33f45b8`, `4e2f595`), not opportunistic edits. The
argmax-forcing + code-dump live only under `scripts/` (verify PYTHONPATH), never
imported by the server in normal operation.

## Phase 4 note (applicability)
This optimization lives in the **omni** `ModelRunner` + `OmniScheduler`. Plain
**Llama-3-8B is served by upstream sglang directly** (which already has its own
FutureMap overlap scheduler) and does **not** flow through our code path — so
`--enable-async-decode` is inert for it and the Llama-3-8B benchmark rows
**cannot exercise this change**. The benchmark reports the meaningful configs
(Higgs TTS bs=1, bs=4) and explains the Llama N/A — see `benchmark_results.md`.
