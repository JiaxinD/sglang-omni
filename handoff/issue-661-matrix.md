## TTS optimization coverage matrix (detailed)

_Status verified against GitHub on 2026-06-21._

Legend: ✅ merged · 🟡 open PR (in progress) · ◐ partial (merged, scope-limited) · ❌ not done · n/a not applicable

| Optimization | Fish S2-Pro | Qwen3-TTS | Voxtral | Higgs v3 | MOSS-Local |
|---|---|---|---|---|---|
| V1 staged onboarding | ✅ #142/#334 | ✅ #451 | ✅ #248/#451 | ✅ #428 | ✅ #728 |
| Shared scheduler infra ¹ | ❌ | ✅ #451 | ✅ #451 | ✅ #476 | ✅ #728 |
| AR decode CUDA graph | ✅ #153 | ✅ #527 | ✅ #527 | ✅ #503 | ✅ #728 |
| Async decode (1-step) | ❌ | ❌ | ❌ | ✅ #590/#638 | ✅ #758 |
| Batched vocoder decode | ✅ #614 ² | ✅ #451 | ❌ #718 ⁵ | ✅ #574 | ✅ #728 |
| Streaming vocoder CUDA graph | ❌ | ❌ | ❌ | 🟡 #729 | ✅ #798 |
| Ref-audio encode cache | 🟡 #809 ³ | ◐ #662 ⁴ | n/a ⁶ | ✅ #605/#713 | ✅ #748 |
| Encoder compile/batch | ❌ | ❌ | n/a ⁶ | ✅ #612 | ✅ #728 |
| Streaming output | ✅ #157/#374 | 🟡 #704 | 🟡 #697 | ✅ #655 | ✅ #753 |
| Colocated / single-GPU budget | 🟡 #409/#391 ⁷ | ✅ #570 | ✅ #570 | ✅ #428/#430 | ✅ #810 |
| Unified seed | ✅ #824 | ✅ #824 | ✅ #824 | ✅ #824 | ✅ #728/#824 |

### Notes

1. **Shared scheduler infra** = runs on the shared scheduler stack rather than a bespoke per-model scheduler. Qwen3-TTS / Voxtral run as staged pipelines on `SimpleScheduler` (#451); Higgs migrated off its custom `HiggsScheduler` onto `OmniScheduler` (#476); MOSS-Local uses the shared stack (#728). Fish S2-Pro still runs its own `FishScheduler`.
2. **Fish batched vocoder (#614)** — attribution to confirm: #614 is "Add streaming TTS schedulers"; Fish's batched *codebook* decode actually landed with the CUDA-graph work (#153). May want a more precise ref (or `✅ #153`).
3. **Fish ref-audio cache (🟡 #809)** — open: reusable reference-encoder refactor (tracks #802).
4. **Qwen3 ref-audio cache (◐ #662, partial)** — #662 ("uploaded voice APIs", merged) added a shared *speaker-artifact* cache for **named/uploaded** voices and wired Qwen3 ref-audio + x-vector through it. It does **not** cache arbitrary per-request `ref_audio` encodes the way Higgs #605/#713 does, so general per-request ref-encode caching for Qwen3 is still open.
5. **Voxtral batched vocoder (❌ #718)** — mirrored Higgs #574 but was closed without merging.
6. **Voxtral n/a** — no reference-audio / voice-cloning input path (preset-voice only), so ref-audio cache and encoder compile/batch don't apply.
7. **Fish colocated budget (🟡 #409/#391)** — two open-but-stalled PRs for the same 24 GB consumer-GPU OOM (#359); neither merged.

cc @zhaochenyang20 @<MELODY-HANDLE>
