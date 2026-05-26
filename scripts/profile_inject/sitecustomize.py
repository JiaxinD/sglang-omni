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


def _patch_mem():
    """Optional (env-gated) cap on the tts_engine KV/static footprint so the
    profile fits alongside other jobs on a shared card. Decode profiling needs
    KV only for the active batch (bs<=32 x olen<=1024 ~ tens of thousands of
    tokens), not the default 0.85-mem-fraction pool. Profiling-only."""
    mf = os.environ.get("SGLANG_OMNI_PROFILE_MEM_FRAC")
    if not mf:
        return
    from sglang_omni.models.higgs_tts import stages as _st

    _orig = _st.create_sglang_tts_engine_executor

    def _patched(
        model_path, *, device="cuda:0", max_new_tokens=2048, server_args_overrides=None
    ):
        ov = dict(server_args_overrides or {})
        ov["mem_fraction_static"] = float(mf)
        ov["max_running_requests"] = int(
            os.environ.get("SGLANG_OMNI_PROFILE_MAXRUN", "34")
        )
        mtt = os.environ.get("SGLANG_OMNI_PROFILE_MAXTOK")
        if mtt:
            ov["max_total_tokens"] = int(mtt)
        print(f"[profile_inject] tts_engine mem overrides: {ov}", flush=True)
        return _orig(
            model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            server_args_overrides=ov,
        )

    _st.create_sglang_tts_engine_executor = _patched


def _wrap_nvtx(func, label, nvtx):
    """Wrap ``func`` so each call is bracketed by an NVTX push/pop ``label``."""
    def wrapped(*args, **kwargs):
        nvtx.range_push(label)
        try:
            return func(*args, **kwargs)
        finally:
            nvtx.range_pop()
    wrapped.__name__ = getattr(func, "__name__", "wrapped")
    return wrapped


def _patch_method(cls, name, label, nvtx):
    """Monkeypatch ``cls.name`` to push NVTX range ``label``. Handles plain
    instance methods and staticmethods. No-op (logged) if the attribute is not
    defined directly on ``cls``."""
    raw = cls.__dict__.get(name)
    if raw is None:
        print(f"[profile_inject] FINE skip {cls.__name__}.{name} (not on class)",
              flush=True)
        return
    if isinstance(raw, staticmethod):
        setattr(cls, name, staticmethod(_wrap_nvtx(raw.__func__, label, nvtx)))
    else:
        setattr(cls, name, _wrap_nvtx(raw, label, nvtx))
    print(f"[profile_inject] FINE wrapped {cls.__name__}.{name} -> {label}",
          flush=True)


def _install_collect_probe(higgs_cls, nvtx):
    """Replace ``_decode_collect_host`` with a faithful copy that brackets its
    three parts (host flag reads / per-request python loop / next_token_ids H2D)
    in NVTX sub-ranges. Profiling-only; logic is identical to the original so
    correctness is unchanged. Falls back to a plain wrap if anything is off."""
    import torch

    orig = higgs_cls.__dict__.get("_decode_collect_host")
    if orig is None:
        print("[profile_inject] FINE skip collect probe (method absent)", flush=True)
        return

    def probed(self, combined_cpu, result, requests):
        nvtx.range_push("resolve.collect_host")
        try:
            model = self.model
            num_codebooks = model._cg_codes_BN.shape[1]
            codes_BN_cpu = combined_cpu[:, :num_codebooks]
            nvtx.range_push("collect.read_flags")
            was_done_cpu = combined_cpu[:, num_codebooks].bool().tolist()
            gen_done_after_cpu = combined_cpu[:, num_codebooks + 1].bool().tolist()
            nvtx.range_pop()
            cb0_per_row = []
            nvtx.range_push("collect.pyloop")
            for b, sched_req in enumerate(requests):
                data = sched_req.data
                req = data.req
                if req.is_chunked > 0:
                    cb0_per_row.append(0)
                    continue
                if req.finished():
                    cb0_per_row.append(0)
                    continue
                if was_done_cpu[b]:
                    cb0_per_row.append(0)
                    continue
                codes_N = codes_BN_cpu[b]
                data.output_codes.append(codes_N.to(torch.long))
                data.generation_done = bool(gen_done_after_cpu[b])
                self._mark_sampler_finished(req, data.generation_done)
                cb0_per_row.append(int(codes_N[0].item()))
            nvtx.range_pop()
            nvtx.range_push("collect.next_ids_h2d")
            result.next_token_ids = torch.tensor(
                cb0_per_row, dtype=torch.long,
                device=result.logits_output.next_token_logits.device,
            )
            nvtx.range_pop()
        finally:
            nvtx.range_pop()

    higgs_cls._decode_collect_host = probed
    print("[profile_inject] FINE collect probe installed (resolve.collect_host "
          "split into read_flags/pyloop/next_ids_h2d)", flush=True)


def _patch_fine(nvtx):
    """Env-gated finer-grained NVTX ranges INSIDE the decode launch/resolve path
    so the ~3.8ms per-step CPU gap (stall_analysis.md) can be attributed to its
    individual sub-steps. Profiling-only; nests under decode_launch /
    decode_resolve. Each wrap is independent — one missing symbol can't disarm
    the rest."""
    if os.environ.get("SGLANG_OMNI_PROFILE_FINE") != "1":
        return
    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    # scheduler-level (sibling of decode_launch in the event loop)
    _patch_method(OmniScheduler, "get_next_batch_to_run", "step.get_next_batch", nvtx)
    # launch path (nested under decode_launch)
    _patch_method(ModelRunner, "_build_forward_batch", "launch.build_forward_batch", nvtx)
    _patch_method(HiggsTTSModelRunner, "_populate_cg_buffers",
                  "launch.populate_cg_buffers", nvtx)
    _patch_method(HiggsTTSModelRunner, "_extract_decode_sampling_params",
                  "launch.extract_sampling_params", nvtx)
    _patch_method(ModelWorker, "forward_batch_generation", "launch.forward", nvtx)
    _patch_method(ModelRunner, "_sample_next_token_ids", "launch.sample", nvtx)
    _patch_method(HiggsTTSModelRunner, "post_decode_launch",
                  "launch.post_decode_launch", nvtx)
    _patch_method(HiggsTTSModelRunner, "_decode_pack_gpu", "launch.pack_gpu", nvtx)
    # resolve path (nested under decode_resolve). Replace with a sub-probed
    # copy (NOT just a wrap) so the 3.4ms collect_host can be split into its
    # host-read / python-loop / next_ids-H2D parts — the H2D is a blocking
    # cudaMemcpy on the single decode stream and is the prime sync suspect.
    _install_collect_probe(HiggsTTSModelRunner, nvtx)
    # CUDA graph replay (nested under launch.forward) — optional: only present
    # when decode runs under a captured graph.
    try:
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
        _patch_method(CudaGraphRunner, "replay", "launch.forward_replay", nvtx)
    except Exception as exc:  # pragma: no cover - import shape varies upstream
        print(f"[profile_inject] FINE skip CudaGraphRunner.replay ({exc!r})",
              flush=True)


def _install():
    import torch

    _patch_mem()

    from sglang_omni.model_runner import base as _b

    nvtx = torch.cuda.nvtx
    _patch_fine(nvtx)
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
            print(
                f"[profile_inject] cudaProfilerStart @ decode step {st['step']}",
                flush=True,
            )
        nvtx.range_push("decode_launch")
        try:
            return _orig_launch(self, scheduler_output)
        finally:
            nvtx.range_pop()
            st["step"] += 1
            if st["started"] and not st["stopped"] and st["step"] >= warmup + capture:
                cudart.cudaProfilerStop()
                st["stopped"] = True
                print(
                    f"[profile_inject] cudaProfilerStop @ decode step {st['step']}",
                    flush=True,
                )

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
