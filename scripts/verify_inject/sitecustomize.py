"""Auto-imported (PYTHONPATH) injector for verify_correctness.py.

Two non-invasive patches, applied at interpreter startup (before the server
builds / captures CUDA graphs, so the graph captures the patched kernels):

1. Force TRUE greedy: replace the Higgs batched sampler with argmax. The
   production CG sampler (_sample_independent_batched) uses multinomial even at
   temperature~0 (unlike the per-row _sample_independent which short-circuits
   to argmax), so decode is otherwise non-deterministic run-to-run. argmax is
   RNG-free and batch-shape-invariant -> deterministic, so async-on vs
   async-off output_codes can be compared bit-for-bit.

2. If SGLANG_OMNI_VERIFY_DUMP is set, tee each request's per-step appended
   codes to that JSONL file (the shared _decode_collect_host is the single
   collect point for both the sync and async paths).

Production code is untouched; this lives only on the verify script's PYTHONPATH.
"""
import os


def _install():
    import torch

    from sglang_omni.models.higgs_tts import model_runner as _mr
    from sglang_omni.models.higgs_tts import sampler as _samp

    def _argmax_sampler(logits_BNV, *, temperature=None, top_p=None, top_k_buf=None):
        return logits_BNV.argmax(dim=-1).to(torch.long)

    _samp._sample_independent_batched = _argmax_sampler

    dump_path = os.environ.get("SGLANG_OMNI_VERIFY_DUMP")
    if dump_path:
        import json

        _orig = _mr.HiggsTTSModelRunner._decode_collect_host
        _fh = open(dump_path, "a", buffering=1)

        def _patched(self, combined_cpu, result, requests):
            before = [len(sr.data.output_codes) for sr in requests]
            _orig(self, combined_cpu, result, requests)
            for sr, n0 in zip(requests, before):
                oc = sr.data.output_codes
                if len(oc) > n0:
                    _fh.write(
                        json.dumps(
                            {"rid": str(sr.request_id), "codes": oc[-1].tolist()}
                        )
                        + "\n"
                    )

        _mr.HiggsTTSModelRunner._decode_collect_host = _patched


try:
    _install()
    print("[verify_inject] greedy/argmax + code-dump patches installed")
except Exception as exc:  # never break the server if the injector misfires
    print(f"[verify_inject] WARNING: install failed: {exc!r}")
