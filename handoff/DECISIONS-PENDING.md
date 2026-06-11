# DECISIONS-PENDING — PR #745 (feat/moss-local-state-pool)

Items that need **your** call. No code was changed for any of these; they are
investigation + options only. Last updated against PR head `7ee050b` +
local `2f33b06` (pushed) + `2024b80` (Phase 2, staged on fork, **not** on the
live PR yet).

---

## D0. Push Phase 2 (GPU radix hash) to the live PR? — needs go/no-go

Commit `2024b80` ("perf: capture-safe GPU radix hash …") is implemented,
CPU-tested green, and documented, but I **staged it on the fork branch only**
(`origin/claude/friendly-keller-tv8t5e`) instead of pushing to the live PR,
because it is the largest/most-scrutinized change and you've been supervising
closely. Push access to `feat/moss-local-state-pool` is confirmed (2f33b06
landed).

- **Option A (recommended):** say "go" → I `git push upstream
  2024b80:feat/moss-local-state-pool` and add the 11.b draft reply to
  gaoyang07's thread to `REVIEW-REPLIES.md` (you post it — footer constraint).
- **Option B:** hold until shuaills' review lands / you eyeball the diff
  (`handoff/patches/0002-*.patch`).

The GPU output-layer (S0 bit-identity) rerun is **PENDING-GPU** either way.

---

## D1. c1 — chunked-prefill advances `generation_steps` (Ratish, real issue)

Thread: PR #745 `#discussion_r3398358018` (model_runner.py:277).
Status: **investigated, NOT acted on — your call.**

### The bug (confirmed by code read)

`_collect_frame` samples a frame and publishes `result.next_token_ids` /
`schedule_batch.output_ids` for **all** rows including `is_chunked > 0`
(model_runner.py:270-272), then filters `emit_indices` only for the
feedback-scatter + journal. But the shared `ModelRunner._finalize`
(`base.py:343`) does `data.generation_steps += 1` for **every** request not in
`skip_rids` — including chunked rows (and even for `is_prefill_only` batches).

Why it matters: sampling is **positional/stateless** —
`multinomial_with_seed(probs, seed, position)` with
`position = generation_steps * num_channels + channel` (moss_tts/model_runner.py
:324-325, 357-358). So there is **no RNG stream to "consume"**; the *only* way a
mid-prefill chunk perturbs final output is by advancing `generation_steps`. A
K-chunk prefill advances it K times where the no-chunk path advances once, so
the **final chunk's first real frame samples at a shifted position** → not
bit-identical to the no-chunk path.

Severity: latent/defensive today — the PR notes chunked prefill is
"structurally unreachable on ≥80GB cards" for v1.5 default config. But Ratish is
right that the guard is incomplete, and #734/#736 lookahead could make it
reachable.

### Fix target

Do **not** advance `generation_steps` for `is_chunked > 0` (non-final) chunks;
the final chunk (`is_chunked == 0`) advances once, matching the no-chunk path.
The set is exactly `_is_chunked_request`.

### Approach A — "skip before frame production" (in `_collect_frame`)

Exclude chunked rows from the micro-decode itself: gather the non-chunked
sub-batch, sample on it, scatter back into batch-aligned `next_token_ids` with
placeholders for chunked positions.

- **Files:** `model_runner.py` (`_collect_frame`, significant restructure).
- **Tensor shapes:** **affected / risky.** `decode_frame_graphed` is
  CUDA-graphed at fixed batch sizes (`frame_graph_max_bs`); feeding it a
  variable sub-batch (chunked excluded) breaks graph-replay shape assumptions →
  needs a separate graph or eager fallback for mixed batches. `next_token_ids` /
  `output_ids` must still be full-batch length → placeholder ids for chunked
  positions.
- **Does NOT by itself stop the `generation_steps` advance** — `_finalize`
  still loops the full `scheduler_output.requests`. So Approach A *also* needs
  the skip-set piece below. Strictly more work, higher blast radius, touches the
  capture path and risks the S0 gate.

### Approach B — `_finalize` skip-set (reuse existing mechanism)

`_finalize` **already** takes `skip_rids` (`base.py:304, 338-341`) and the
async-resolve path already builds one (`base.py:190`, for finished reqs). Feed
the chunked rids in so their `generation_steps += 1` is suppressed. No sampling
restructure; chunked rows are still sampled (as today) but their step isn't
advanced and (already) their feedback/journal are skipped. **No tensor-shape /
graph-capture interaction.**

Two plumbing variants:
- **B1 (recommended): subclass hook.** Add `def finalize_skip_rids(self,
  scheduler_output) -> set[str]` to `ModelRunner` (default empty), called at
  *both* `_finalize` sites (`base.py:118` and `:202`) and unioned into
  `skip_rids`. MOSS-TTS Local overrides it to return the chunked rids. Robust:
  fires even for `is_prefill_only` chunked batches (where `post_prefill` returns
  early and never runs `_collect_frame`). Base change is additive + behaviour-
  neutral for every other model (default empty).
- **B2: stash on `batch_result`.** Set `result.skip_finalize_rids = {chunked
  rids}` in `_collect_frame`, union `getattr(batch_result,
  "skip_finalize_rids", set())` in `_finalize`. Smaller, but **has a gap**: a
  `is_prefill_only` chunked batch skips `_collect_frame`, so the set is never
  set and `generation_steps` still over-advances. B1 closes that gap.

- **Files (B1):** `base.py` (new hook + 2 union lines) + `model_runner.py`
  (override returning chunked rids). **Tensor shapes: unaffected.**

### Recommendation

**Approach B1.** Lowest blast radius, reuses the existing `skip_rids`
mechanism, no graph/shape interaction, and surgically removes exactly the
spurious `generation_steps` advance Ratish flagged. Approach A is strictly more
invasive and still needs the skip-set anyway.

### Test construction (CPU, fake-model style like test_state_pool.py)

1. **`generation_steps` not advanced for chunked rows:** drive the finalize
   path with a batch of one `is_chunked=1` + one `is_chunked=0` request; assert
   chunked `data.generation_steps` unchanged and normal advanced by 1.
2. **Bit-identical base point:** run (a) no-chunk: prefill completes in one shot;
   (b) chunked: same request split into K chunks (K-1 mid + 1 final). Stub
   `_sample_tokens` / `multinomial_with_seed` to record its `positions` arg;
   assert the `positions` for the first real decoded frame are **equal** between
   the two runs (so the sampled tokens, hence audio, are bit-identical). Since
   sampling is positional, equal `positions` ⇒ equal output — no GPU needed.

### One thing to confirm before coding

`is_chunked` semantics: I read `req.is_chunked > 0` (via `_is_chunked_request`)
as "non-final chunk, more to come" and `== 0` as the final/only chunk. The fix
assumes that. Worth a 1-line confirm against the scheduler's chunk bookkeeping
before implementing.

---

## D2. c2 — remove `scripts/gate_s0_sync_parity.py`? (Ratish)

Thread: `#discussion_r3398085976` ("seems like a local script file which
shouldn't be here"). Also `#discussion_r3398238552` on
`tests/unit_test/moss_tts_local/test_parity_harness.py` ("should be removed as
well").

Tension: this script + the parity harness are the PR's **S0 bit-identity gate**,
cited in the PR body ("Verification") and now the **output layer** of the
two-layer rubric for the GPU-hash change (`docs/design/gpu_radix_hash.md`).
Removing them deletes the behaviour-neutrality evidence.

Options (your call):
- Keep as-is.
- **Relocate** the standalone script under `tests/` (or `tests/manual/`) and/or
  convert the gate into a `@pytest.mark.gpu` test so it stops looking like a
  stray local script while preserving the verification.
- Remove (only if you've decided the S0 gate's job is done and CI's sentinel
  test suffices).

Draft reply: `REVIEW-REPLIES.md → c2`. Not posted.

---

## D3. c3 — rename `MossTTSLocalDecodeJournal`? (Ratish)

Thread: `#discussion_r3398513076` (state_pool.py:159) — suggests
`MossTTSLocalDecodeFrameResult` / `MossTTSLocalDecodeStepOutput`, asks @JiaxinD.

Tension: the journal name is part of the **#736 interface contract** (PR body
"For #736" + the now-`#734` docstring). A rename is an API-surface change that
touches #734/#736 consumers, so per the watch rules this is your call, not a
mechanical nit.

Options: keep `…Journal`; or rename to one of Ratish's suggestions (then grep
`MossTTSLocalDecodeJournal` across the branch + coordinate with #734/#736).
Draft reply: `REVIEW-REPLIES.md → c3`. Not posted.

---

## D4. AkazaAkane — vectorize `_collect_frame` into [P] pool tensors (follow-up)

Thread: `#discussion_r3398199811` (model_runner.py:173). Directional design
input (orchestration into `[P]` tensors / `gen_steps = pool.generation_steps
[row_t]`). Per the brief this is the *second half* of the pooling work, a
follow-up after this PR + #734, pairing with #736. Draft ack:
`REVIEW-REPLIES.md → 11c`. Not posted.
