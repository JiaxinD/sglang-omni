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

## Tests
- `tests/unit_test/higgs_tts/` + `pipeline/test_scheduler.py` + `pipeline/test_async_decode.py`: **35 pass** (sync path unchanged; async state machine covered).
- `tests/unit_test/pipeline/test_ipc.py`: **7 failures — PRE-EXISTING**, confirmed by running the base-commit (`be93a97`) version (7 fail there too). They concern the mp_runner/launcher IPC, unrelated to async decode; not introduced by this work.
- **verify_correctness (PR merge gate), greedy, output_codes bit-identical OFF vs ON:**
  - length-finish 10 prompts × 10 runs (max=32): **PASS, 100/100 bit-identical**, audio OK.
  - EOC-finish 2 × 2 (max=256): **PASS, bit-identical**, audio OK.
  - EOC-finish 10 × 10 (max=256): *(running — result appended to verdict)*.
  - The 100/100 "audio sha differ" note is benign cross-process vocoder float nondeterminism (cuDNN/cuBLAS algo selection per process); the **output_codes** (what async decode controls) are bit-identical.

## Q1 / Q2 (resolved — see `phase3_clarifications.md`)
- **Q1**: lookahead = **1 wasted step**, not 2 (flag staleness is 2 but the finishing step isn't wasted). Invariant 3 unchanged. No loop reorder (resolve-first kills overlap).
- **Q2**: `_FAILED_BATCH_RESULT` is an existing sentinel (`omni_scheduler.py:36`); the async loop reuses it.

## Honest uncertainties / things to flag
1. **CG decode is non-deterministic at `temperature=0`** (PRE-EXISTING, not from this PR): `_sample_independent_batched` (sampler.py:214) uses `multinomial` and — unlike the per-row `_sample_independent` (sampler.py:124) — does NOT short-circuit greedy to `argmax`, so near-tie logits diverge run-to-run. The gate forces argmax via a non-invasive `scripts/verify_inject/sitecustomize.py` (patched before CUDA-graph capture). **Recommend a separate one-line fix** making the batched sampler short-circuit greedy like the per-row one — it would make production greedy reproducible.
2. **EOC overrun guard adds a per-step GPU gather** (`generation_done[rows]` + `torch.where`) in `_populate_cg_buffers` when async is on — GPU-side, no host sync. Cheap, but it's extra work each decode step; an alternative is to suppress the launch entirely when all reqs are done (needs the resolve, which lags).
3. **bs>1 (concurrent requests) is NOT yet working in async mode** — a real open bug. The bit-identical gate is bs=1 (concurrency=1, sequential). At bs=4 async ON, when one of several requests finishes mid-batch the next launched batch hits a `cuda_graph_runner.replay_prepare` size mismatch (tensor 3 vs 4) — the launch-first lag leaves the running batch's tensors inconsistent with its (reduced) req count. async **OFF** bs=4 is fine. **bs>1 needs additional work before merge**; bs=1 is verified correct. (See benchmark_results.md.)
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
