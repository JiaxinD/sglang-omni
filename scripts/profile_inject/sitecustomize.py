"""Auto-imported (PYTHONPATH) NVTX + capture-range injector for the T2
decode-isolated nsys profile. ZERO production change — wraps the async
ModelRunner.execute_launch / execute_resolve at import time.

- Pushes NVTX ranges "decode_launch" / "decode_resolve" so the GPU timeline is
  readable per step (launch = forward+sample+D2H enqueue; resolve = host
  collect of the previous step).
- Drives the nsys capture window via the CUDA Profiler API: after
  SGLANG_OMNI_PROFILE_WARMUP decode steps it calls cudaProfilerStart, and after
  a further SGLANG_OMNI_PROFILE_CAPTURE steps it calls cudaProfilerStop. Run
  nsys with `--capture-range=cudaProfilerApi` so only the steady-state decode
  window lands in the trace.

Lives only on the profiler's PYTHONPATH; the server in normal operation never
imports it.
"""
import os


def _install():
    import torch

    from sglang_omni.model_runner import base as _b

    nvtx = torch.cuda.nvtx
    cudart = torch.cuda.cudart()
    warmup = int(os.environ.get("SGLANG_OMNI_PROFILE_WARMUP", "40"))
    capture = int(os.environ.get("SGLANG_OMNI_PROFILE_CAPTURE", "80"))
    st = {"step": 0, "started": False, "stopped": False}

    _orig_launch = _b.ModelRunner.execute_launch
    _orig_resolve = _b.ModelRunner.execute_resolve

    def execute_launch(self, scheduler_output):
        if st["step"] == warmup and not st["started"]:
            cudart.cudaProfilerStart()
            st["started"] = True
            print(f"[profile_inject] cudaProfilerStart @ decode step {st['step']}",
                  flush=True)
        nvtx.range_push("decode_launch")
        try:
            return _orig_launch(self, scheduler_output)
        finally:
            nvtx.range_pop()
            st["step"] += 1
            if (st["started"] and not st["stopped"]
                    and st["step"] >= warmup + capture):
                cudart.cudaProfilerStop()
                st["stopped"] = True
                print(f"[profile_inject] cudaProfilerStop @ decode step {st['step']}",
                      flush=True)

    def execute_resolve(self, pending):
        nvtx.range_push("decode_resolve")
        try:
            return _orig_resolve(self, pending)
        finally:
            nvtx.range_pop()

    _b.ModelRunner.execute_launch = execute_launch
    _b.ModelRunner.execute_resolve = execute_resolve


try:
    _install()
    print("[profile_inject] NVTX + capture-range wrappers installed")
except Exception as exc:
    print(f"[profile_inject] WARNING: install failed: {exc!r}")
