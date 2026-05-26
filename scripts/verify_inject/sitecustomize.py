"""Auto-imported (PYTHONPATH) code-dump injector for verify_correctness.py.

ONE non-invasive patch, applied at interpreter startup: if
SGLANG_OMNI_VERIFY_DUMP is set, tee each request's per-step appended codes to
that JSONL file (the shared ``_decode_collect_host`` is the single collect point
for both the sync and async paths). Each line also carries a stable ``pid`` (a
hash of the request's prompt token ids) so concurrent (bs>1) runs can be grouped
back per prompt for the bit-identical comparison.

Note: this NO LONGER forces argmax. Since the batched sampler now short-circuits
greedy to argmax (sampler.py ``_sample_independent_batched``), production
``temperature=0`` is itself deterministic, so the gate compares the real
production sampler path OFF vs ON. Production code is untouched; this lives only
on the verify script's PYTHONPATH.
"""

import hashlib
import os


def _prompt_id(data) -> str:
    """Stable short hash of the prompt token ids — identical across server runs
    and batch sizes, unlike the server-assigned random rid."""
    ids = getattr(data, "input_ids", None)
    if ids is None:
        return "?"
    try:
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    except Exception:
        return "?"
    return hashlib.blake2b(repr(seq).encode(), digest_size=8).hexdigest()


def _install():
    from sglang_omni.models.higgs_tts import model_runner as _mr

    dump_path = os.environ.get("SGLANG_OMNI_VERIFY_DUMP")
    if not dump_path:
        return

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
                        {
                            "rid": str(sr.request_id),
                            "pid": _prompt_id(sr.data),
                            "codes": oc[-1].tolist(),
                        }
                    )
                    + "\n"
                )

    _mr.HiggsTTSModelRunner._decode_collect_host = _patched


try:
    _install()
    print("[verify_inject] per-step code-dump patch installed (no argmax force)")
except Exception as exc:  # never break the server if the injector misfires
    print(f"[verify_inject] WARNING: install failed: {exc!r}")
