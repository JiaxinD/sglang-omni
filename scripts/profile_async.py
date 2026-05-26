#!/usr/bin/env python
"""T2: decode-isolated nsys profile of Higgs TTS async decode.

For each config it runs the server under `nsys profile` with the CUDA
Profiler API capture range (scripts/profile_inject arms Start/Stop around a
steady-state decode window), drives a constant-bs load (C identical requests so
the batch stays bs=C through the window), then parses the GPU kernel trace to
answer: is there a serial GPU-idle gap that single-stream (Plan A) leaves on the
table and multi-stream (Plan B) could recover?

Metric per config: window span, GPU-busy %, total idle, max inter-kernel gap,
and the count of gaps >= 0.5 ms (the Plan-B-recoverable threshold).

Usage:
    python scripts/profile_async.py --gpu 2            # all 6 configs
    python scripts/profile_async.py --gpu 2 --only bs4 # one config
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = "/data/sglang-omni"
MODEL = "boson-sglang/higgs-audio-v3-tts-4b-base"
INJECT = f"{REPO}/scripts/profile_inject"

# (tag, concurrency, output_len, warmup_steps, capture_steps)
# Windows are kept well under the EOC-bound natural decode length (~the prompts
# stop on EOC before max_new_tokens), so cudaProfilerStop always fires inside
# the run. A ~30-step steady-state window is ample to characterise per-step gaps.
CONFIGS = [
    ("bs1_olen256", 1, 256, 10, 30),
    ("bs4_olen256", 4, 256, 10, 30),
    ("bs16_olen256", 16, 256, 10, 30),
    ("bs32_olen256", 32, 256, 10, 30),
    ("bs4_olen64", 4, 64, 8, 24),
    ("bs4_olen1024", 4, 1024, 10, 30),
]


def _kill_servers():
    # Kill the whole pipeline: the cli, the spawned stage subprocesses (which
    # actually bind the API port and otherwise linger -> the next server falls
    # back to an ephemeral port), and the nsys wrapper.
    pat = "[s]glang_omni.cli serve|[n]sys profile|[s]tage_process|multiprocessing.spawn"
    subprocess.run(
        f"ps aux | grep -E '{pat}' | awk '{{print $2}}' | xargs -r kill -9", shell=True
    )
    time.sleep(5)


def _actual_port(log_path, default):
    """The omni server falls back to an ephemeral port if the requested one is
    taken; read the real port it bound from its log."""
    import re

    try:
        m = re.findall(r"Uvicorn running on http://[\d.]+:(\d+)", open(log_path).read())
        if m:
            return int(m[-1])
    except FileNotFoundError:
        pass
    return default


def _health(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            return True
        except Exception:
            try:
                requests.get(f"http://127.0.0.1:{port}/", timeout=3)
                return True
            except Exception:
                time.sleep(2)
    return False


def _launch(tag, gpu, port, warmup, capture, rep_path, log_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SGLANG_OMNI_ENABLE_ASYNC_DECODE"] = "1"
    env["SGLANG_OMNI_PROFILE_WARMUP"] = str(warmup)
    env["SGLANG_OMNI_PROFILE_CAPTURE"] = str(capture)
    # Cap the KV/static footprint so the profile fits alongside other jobs on a
    # shared card (decode needs KV only for the active batch). Overridable.
    env.setdefault(
        "SGLANG_OMNI_PROFILE_MEM_FRAC", os.environ.get("PROF_MEM_FRAC", "0.12")
    )
    env.setdefault("SGLANG_OMNI_PROFILE_MAXRUN", "34")
    env.setdefault("SGLANG_OMNI_PROFILE_MAXTOK", "49152")
    env["PYTHONPATH"] = f"{INJECT}:{REPO}"
    log = open(log_path, "w")
    cmd = [
        "nsys",
        "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "-o",
        rep_path,
        "--force-overwrite=true",
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        "examples/configs/higgs_tts.yaml",
        "--port",
        str(port),
    ]
    return (
        subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ),
        log,
    )


def _wait_ready(log_path, proc, timeout=480):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server died at startup; see {log_path}")
        try:
            if "Application startup complete" in open(log_path).read():
                return
        except FileNotFoundError:
            pass
        time.sleep(3)
    raise TimeoutError(log_path)


def _drive(port, sample, conc, olen):
    url = f"http://127.0.0.1:{port}/v1/audio/speech"

    def _send(_):
        payload = {
            "model": MODEL,
            "input": sample.target_text,
            "ref_audio": sample.ref_audio,
            "ref_text": sample.ref_text,
            "max_new_tokens": olen,
            "temperature": 0.0,
        }
        try:
            requests.post(url, json=payload, timeout=600)
        except Exception as exc:
            print(f"  [warn] request err: {str(exc)[:80]}", flush=True)

    # C identical requests in flight -> constant bs=C through the capture window
    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(_send, range(conc)))


def _wait_capture_done(log_path, proc, timeout=900):
    """Block until the inject prints cudaProfilerStop (capture window done) or
    the (stop-shutdown) process exits."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return True
        try:
            if "cudaProfilerStop" in open(log_path).read():
                return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    return False


def _analyze(rep_path):
    """Parse the GPU kernel trace; return gap stats over the capture window."""
    rep = rep_path + ".nsys-rep"
    if not os.path.exists(rep):
        return {"error": f"no report at {rep}"}
    out = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_trace",
            "--format",
            "csv",
            "--force-export=true",
            rep,
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return {"error": f"nsys stats failed: {out.stderr[:200]}"}
    # nsys prints preamble lines ("Generating SQLite...", "Processing...") before
    # the real CSV header. Find the header line (starts with "Start (ns)").
    lines = out.stdout.splitlines()
    hdr_i = next((i for i, ln in enumerate(lines) if ln.startswith("Start (ns)")), None)
    if hdr_i is None:
        return {"error": f"no CSV header in nsys output: {lines[:3]}"}
    rows = []
    reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_i:])))
    start_key = dur_key = None
    for fn in reader.fieldnames or []:
        lf = fn.lower()
        if lf.startswith("start"):
            start_key = fn
        elif lf.startswith("duration") or lf.startswith("durs"):
            dur_key = fn
    if not start_key or not dur_key:
        return {"error": f"unexpected columns: {reader.fieldnames}"}

    def _ns(v):
        v = v.strip().replace(",", "")
        return float(v)

    for row in reader:
        try:
            s = _ns(row[start_key])
            d = _ns(row[dur_key])
        except (ValueError, KeyError, TypeError):
            continue
        rows.append((s, s + d))
    if len(rows) < 2:
        return {"error": f"too few kernels ({len(rows)})"}
    rows.sort()
    window = rows[-1][1] - rows[0][0]
    busy = sum(e - s for s, e in rows)
    # Merge overlaps for a correct busy figure, then gaps on the merged timeline.
    merged = []
    for s, e in rows:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    busy_merged = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    gaps = [g for g in gaps if g > 0]
    ns = 1e-6  # ns -> ms
    return {
        "kernels": len(rows),
        "window_ms": window * ns,
        "busy_pct": 100.0 * busy_merged / window if window else 0.0,
        "idle_ms": (window - busy_merged) * ns,
        "max_gap_ms": (max(gaps) * ns) if gaps else 0.0,
        "n_gaps_ge_0p5ms": sum(1 for g in gaps if g * ns >= 0.5),
        "mean_gap_us": (sum(gaps) / len(gaps) * 1e-3) if gaps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--port", type=int, default=8230)
    ap.add_argument("--only", default="", help="run only the config with this tag")
    ap.add_argument("--outdir", default="/tmp/nsys")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sys.path.insert(0, REPO)
    from benchmarks.dataset.seedtts import load_seedtts_samples

    sample = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", 1, split="en")[
        0
    ]

    configs = [c for c in CONFIGS if (not args.only or c[0] == args.only)]
    results = {}
    for ci, (tag, conc, olen, warmup, capture) in enumerate(configs):
        port = args.port + ci  # unique per config to avoid stale-port fallback
        print(
            f"\n===== {tag} (conc={conc} olen={olen} "
            f"warmup={warmup} capture={capture} port={port}) =====",
            flush=True,
        )
        _kill_servers()
        rep = os.path.join(args.outdir, tag)
        log = f"/tmp/profile_{tag}.log"
        proc, _ = _launch(tag, args.gpu, port, warmup, capture, rep, log)
        try:
            _wait_ready(log, proc)
            real_port = _actual_port(log, port)
            if not _health(real_port):
                raise RuntimeError(f"server not reachable on port {real_port}")
            print(f"  server ready on port {real_port}; driving load...", flush=True)
            _drive(real_port, sample, conc, olen)
            ok = _wait_capture_done(log, proc)
            print(f"  capture done={ok}", flush=True)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            time.sleep(8)
            _kill_servers()
        stats = _analyze(rep)
        results[tag] = stats
        print(f"  {stats}", flush=True)

    print("\n" + "=" * 92)
    print(
        f"{'config':<16}{'kernels':>9}{'window ms':>11}{'busy %':>9}"
        f"{'idle ms':>9}{'maxgap ms':>11}{'gaps>=0.5ms':>13}"
    )
    for tag, st in results.items():
        if "error" in st:
            print(f"{tag:<16}  ERROR: {st['error']}")
            continue
        print(
            f"{tag:<16}{st['kernels']:>9}{st['window_ms']:>11.1f}"
            f"{st['busy_pct']:>9.1f}{st['idle_ms']:>9.2f}"
            f"{st['max_gap_ms']:>11.3f}{st['n_gaps_ge_0p5ms']:>13}"
        )
    print("=" * 92)
    import json

    json.dump(
        results, open(os.path.join(args.outdir, "stall_stats.json"), "w"), indent=2
    )
    print(f"wrote {args.outdir}/stall_stats.json")


if __name__ == "__main__":
    main()
