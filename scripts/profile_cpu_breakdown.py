#!/usr/bin/env python
"""CPU path breakdown for Higgs async decode (finer-grained T2 follow-up).

stall_analysis.md established that decode is GPU-idle ~99% with a ~3.8ms/step
CPU prepare gap. This script attributes that gap to its individual CPU
sub-steps. It reuses profile_async's server launch / health / load-drive
plumbing, but with SGLANG_OMNI_PROFILE_FINE=1 so scripts/profile_inject pushes
nested NVTX ranges (launch.populate_cg_buffers, launch.extract_sampling_params,
launch.forward[_replay], launch.post_decode_launch, resolve.collect_host, ...).

It then parses the NVTX push/pop trace (nsys stats --report nvtx_pushpop_trace)
and reports, per range, the steady-state CPU duration distribution plus the
per-step period (median gap between consecutive decode_launch starts).

NO production change: instrumentation lives only on the profiler PYTHONPATH and
is gated by SGLANG_OMNI_PROFILE_FINE.

Usage:
    python scripts/profile_cpu_breakdown.py --gpu 3 --conc 4 --olen 128 \
        --warmup 20 --capture 60
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import statistics as stats
import subprocess
import sys
import time

REPO = "/data/sglang-omni"


def _percentile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _analyze_nvtx(rep_path, parent="decode_launch"):
    """Parse the NVTX push/pop trace; per range name return the steady-state
    CPU duration distribution (us) plus the per-step period from consecutive
    `parent` starts."""
    rep = rep_path + ".nsys-rep"
    if not os.path.exists(rep):
        return {"error": f"no report at {rep}"}
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_pushpop_trace", "--format", "csv",
         "--force-export=true", rep],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return {"error": f"nsys stats failed: {out.stderr[-400:]}"}
    lines = out.stdout.splitlines()
    hdr_i = next((i for i, ln in enumerate(lines)
                  if ln.startswith("Start")), None)
    if hdr_i is None:
        return {"error": f"no CSV header: {lines[:4]}"}
    reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_i:])))
    fns = reader.fieldnames or []
    start_key = next((f for f in fns if f.lower().startswith("start")), None)
    dur_key = next((f for f in fns if f.lower().startswith("duration")
                    or f.lower().startswith("durs")), None)
    # exclusive (self) time, if this nsys version emits it
    excl_key = next((f for f in fns if "nonchild" in f.lower().replace(" ", "")
                     or "durnonchild" in f.lower().replace(" ", "")), None)
    name_key = next((f for f in fns if f.lower() == "name"), None)
    if not (start_key and dur_key and name_key):
        return {"error": f"unexpected columns: {fns}"}

    def _ns(v):
        return float(v.strip().replace(",", ""))

    incl = {}   # name -> [inclusive us]
    excl = {}   # name -> [exclusive us]
    starts = {}  # name -> [start ns]
    for row in reader:
        try:
            name = row[name_key].strip()
            d = _ns(row[dur_key])
            s = _ns(row[start_key])
        except (ValueError, KeyError, TypeError):
            continue
        incl.setdefault(name, []).append(d * 1e-3)  # ns -> us
        starts.setdefault(name, []).append(s)
        if excl_key and row.get(excl_key, "").strip():
            try:
                excl.setdefault(name, []).append(_ns(row[excl_key]) * 1e-3)
            except ValueError:
                pass

    # per-step period: median delta between consecutive parent starts
    p_starts = sorted(starts.get(parent, []))
    deltas = [(p_starts[i + 1] - p_starts[i]) * 1e-3
              for i in range(len(p_starts) - 1)]
    period_us = stats.median(deltas) if deltas else 0.0

    ranges = {}
    for name, xs in incl.items():
        e = excl.get(name, [])
        ranges[name] = {
            "count": len(xs),
            "mean_us": round(stats.mean(xs), 3),
            "median_us": round(stats.median(xs), 3),
            "p90_us": round(_percentile(xs, 0.90), 3),
            "min_us": round(min(xs), 3),
            "max_us": round(max(xs), 3),
            "std_us": round(stats.pstdev(xs), 3) if len(xs) > 1 else 0.0,
            "mean_excl_us": round(stats.mean(e), 3) if e else None,
        }
    return {
        "parent": parent,
        "step_count": len(p_starts),
        "period_us": round(period_us, 3),
        "period_p90_us": round(_percentile(deltas, 0.90), 3) if deltas else 0.0,
        "ranges": ranges,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--port", type=int, default=8330)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--olen", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--capture", type=int, default=60)
    ap.add_argument("--outdir", default="/tmp/nsys_cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"cpu_bs{args.conc}_olen{args.olen}"
    os.makedirs(args.outdir, exist_ok=True)

    # Arm the finer-grained injector BEFORE importing profile_async (its _launch
    # copies os.environ into the child server's env).
    os.environ["SGLANG_OMNI_ENABLE_ASYNC_DECODE"] = "1"
    os.environ["SGLANG_OMNI_PROFILE_FINE"] = "1"

    sys.path.insert(0, REPO)
    from scripts import profile_async as P
    from benchmarks.dataset.seedtts import load_seedtts_samples

    sample = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", 1,
                                  split="en")[0]

    rep = os.path.join(args.outdir, tag)
    log = f"/tmp/profile_{tag}.log"
    print(f"\n===== {tag} (conc={args.conc} olen={args.olen} "
          f"warmup={args.warmup} capture={args.capture} gpu={args.gpu} "
          f"port={args.port}) =====", flush=True)
    P._kill_servers()
    proc, _ = P._launch(tag, args.gpu, args.port, args.warmup, args.capture,
                        rep, log)
    try:
        P._wait_ready(log, proc)
        real_port = P._actual_port(log, args.port)
        if not P._health(real_port):
            raise RuntimeError(f"server not reachable on port {real_port}")
        print(f"  server ready on port {real_port}; driving load...", flush=True)
        P._drive(real_port, sample, args.conc, args.olen)
        ok = P._wait_capture_done(log, proc)
        print(f"  capture done={ok}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(8)
        P._kill_servers()

    res = _analyze_nvtx(rep)
    out_json = os.path.join(args.outdir, f"{tag}_nvtx.json")
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\nwrote {out_json}")

    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return
    print(f"\n  steady-state decode steps captured: {res['step_count']}")
    print(f"  per-step period: median {res['period_us']:.1f} us  "
          f"(p90 {res['period_p90_us']:.1f} us)")
    print(f"\n  {'range':<34}{'n':>5}{'mean us':>10}{'median':>9}"
          f"{'p90':>9}{'self us':>9}{'% step':>8}")
    period = res["period_us"] or 1.0
    order = ["step.get_next_batch", "decode_launch", "launch.build_forward_batch",
             "launch.populate_cg_buffers", "launch.extract_sampling_params",
             "launch.forward", "launch.forward_replay", "launch.sample",
             "launch.post_decode_launch", "launch.pack_gpu",
             "decode_resolve", "resolve.collect_host"]
    seen = set()
    for name in order + sorted(res["ranges"]):
        if name in seen or name not in res["ranges"]:
            continue
        seen.add(name)
        r = res["ranges"][name]
        self_us = r["mean_excl_us"]
        self_s = f"{self_us:>9.1f}" if self_us is not None else f"{'-':>9}"
        print(f"  {name:<34}{r['count']:>5}{r['mean_us']:>10.2f}"
              f"{r['median_us']:>9.2f}{r['p90_us']:>9.2f}{self_s}"
              f"{100.0 * r['mean_us'] / period:>7.1f}%")


if __name__ == "__main__":
    main()
