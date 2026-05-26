# Mini-plan: separate PR — cache static decode sampling params (drop 3 D2H/step)

**Independent of async decode** (orthogonal); ship as its own small PR off `main`.

## Problem
`HiggsTTSModelRunner._extract_decode_sampling_params` (`model_runner.py:101`) calls
`_flat_sampling_attr` (`model.py:48-58`) once each for `temperatures` / `top_ps` /
`top_ks` on **every decode step**. Each does `.detach().cpu().flatten().tolist()`
— a D2H sync. So every decode step pays **~3 redundant D2H** to re-read values that
are static per request for the whole generation (`investigation.md` §2.1, §8).

## Change (one atomic commit)
- In `_populate_cg_buffers` / `prepare_decode`, extract per-request temperature /
  top_p / top_k **once** (first decode step for a request) and cache host-side,
  keyed by request id (e.g. a dict on the runner or on `data`). Subsequent steps
  read the cached Python values — no D2H.
- Invalidate/drop the cache entry on request finish/abort (reuse the existing
  `acquire_row` / row lifecycle, or clear in the same place rows are released).
- Keep the padding-row defaults (`[1.0]`, `K_MAX`) unchanged.

Scope: `sglang_omni/models/higgs_tts/model_runner.py` (+ maybe a tiny field on the
runner). ~30-40 LOC. No scheduler / no base-runner changes.

## Verify
- Unit: cached values equal the per-step-extracted values for a few configs.
- A/B: decode tokens/sec before/after on Higgs bs=1 (expect a small steady-state
  win from removing 3 syncs/step; `gpu-perf-ab` skill). Honest note: like #564,
  the win may be modest since these syncs hit an otherwise-idle GPU — but they are
  pure waste and the cache is trivially correct.

## Relationship to async PR
Stacks cleanly either way. If async lands first, this still helps the *prepare*
half (which async runs in `execute_launch`); if this lands first, async inherits a
cheaper prepare. No conflict in the touched code.
