"""Auto-imported (PYTHONPATH) injector for benchmark_async.py.

Captures async-decode overlap stats WITHOUT changing the math or timing:
- production sampling is left intact (no argmax patch, unlike verify_inject),
- on SIGTERM (how the benchmark stops the server) it dumps the model runner's
  _async_query_hit / _async_query_miss (how often resolve found the event
  already done = overlap worked vs had to block) + total resolved steps to
  $SGLANG_OMNI_BENCH_STATS.

Production code is untouched; this lives only on the benchmark's PYTHONPATH.
"""

import json
import os
import signal

_runners = []  # ModelRunner instances seen


def _install():
    from sglang_omni.model_runner import base as _b

    _orig_init = _b.ModelRunner.__init__

    def _init(self, *a, **k):
        _orig_init(self, *a, **k)
        _runners.append(self)

    _b.ModelRunner.__init__ = _init

    path = os.environ.get("SGLANG_OMNI_BENCH_STATS")
    if not path:
        return

    def _dump(*_):
        hit = sum(getattr(r, "_async_query_hit", 0) for r in _runners)
        miss = sum(getattr(r, "_async_query_miss", 0) for r in _runners)
        # Only the stage process that actually holds a model runner with
        # resolve activity writes — otherwise sibling stage processes (which
        # share the env path) would race and overwrite with zeros.
        try:
            if hit + miss > 0:
                with open(path, "w") as fh:
                    json.dump({"query_hit": hit, "query_miss": miss}, fh)
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _dump)


try:
    _install()
    print("[bench_inject] overlap-stats SIGTERM dumper installed")
except Exception as exc:
    print(f"[bench_inject] WARNING: install failed: {exc!r}")
