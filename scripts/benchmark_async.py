#!/usr/bin/env python
"""Phase 4 benchmark: async decode OFF vs ON on Higgs TTS.

Measures end-to-end request latency + throughput at a given concurrency, with
PRODUCTION sampling (no argmax patch). For async ON it also reports the overlap
query-hit rate (resolve found the event already done) and derives per-step time
from total decode steps (= query_hit + query_miss).

Configs (the only ones that exercise this code path — see notes):
    Higgs TTS bs=1, Higgs TTS bs=4.

Usage:
    python scripts/benchmark_async.py --gpu 1 --requests 20 --max-new-tokens 128
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = "/data/sglang-omni"
MODEL = "boson-sglang/higgs-audio-v3-tts-4b-base"
INJECT = f"{REPO}/scripts/bench_inject"


def _samples(n):
    sys.path.insert(0, REPO)
    from benchmarks.dataset.seedtts import load_seedtts_samples

    return load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", n, split="en")


def _launch(port, gpu, async_on, stats_path, log_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SGLANG_OMNI_ENABLE_ASYNC_DECODE"] = "1" if async_on else "0"
    env["SGLANG_OMNI_BENCH_STATS"] = stats_path
    env["PYTHONPATH"] = f"{INJECT}:{REPO}"
    if os.path.exists(stats_path):
        os.remove(stats_path)
    log = open(log_path, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "sglang_omni.cli", "serve",
         "--config", "examples/configs/higgs_tts.yaml", "--port", str(port)],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def _wait(log_path, proc, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server died; see {log_path}")
        try:
            if "Application startup complete" in open(log_path).read():
                return
        except FileNotFoundError:
            pass
        time.sleep(3)
    raise TimeoutError(log_path)


def _kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _one(url, s, max_new_tokens):
    payload = {"model": MODEL, "input": s.target_text, "ref_audio": s.ref_audio,
               "ref_text": s.ref_text, "max_new_tokens": max_new_tokens}
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=300)
    dt = time.time() - t0
    r.raise_for_status()
    return dt


def _run(port, samples, n, conc, max_new_tokens):
    url = f"http://127.0.0.1:{port}/v1/audio/speech"
    # warmup
    _one(url, samples[0], max_new_tokens)
    reqs = [samples[i % len(samples)] for i in range(n)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        lat = list(ex.map(lambda s: _one(url, s, max_new_tokens), reqs))
    wall = time.time() - t0
    return lat, wall


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--configs", default="1,4", help="comma-sep concurrencies")
    args = ap.parse_args()

    concs = [int(x) for x in args.configs.split(",")]
    samples = _samples(8)
    results = {}
    for conc in concs:
        for async_on in (False, True):
            tag = "ON" if async_on else "OFF"
            stats = f"/tmp/bench_stats_bs{conc}_{tag}.json"
            log = f"/tmp/bench_bs{conc}_{tag}.log"
            print(f"=== bs={conc} async={tag} (GPU{args.gpu}) ===", flush=True)
            proc = _launch(args.port, args.gpu, async_on, stats, log)
            try:
                _wait(log, proc)
                lat, wall = _run(args.port, samples, args.requests, conc,
                                 args.max_new_tokens)
            finally:
                _kill(proc)
                time.sleep(5)
            qstats = {}
            if os.path.exists(stats):
                qstats = json.load(open(stats))
            steps = qstats.get("query_hit", 0) + qstats.get("query_miss", 0)
            results[(conc, tag)] = {
                "lat_mean": statistics.mean(lat), "lat_p50": _pct(lat, 50),
                "lat_p99": _pct(lat, 99), "wall": wall, "n": len(lat),
                "throughput": len(lat) / wall, "steps": steps, **qstats,
            }
            print(f"  mean={statistics.mean(lat)*1000:.0f}ms "
                  f"p50={_pct(lat,50)*1000:.0f}ms p99={_pct(lat,99)*1000:.0f}ms "
                  f"thrpt={len(lat)/wall:.2f}req/s steps={steps} "
                  f"qhit={qstats.get('query_hit','?')}/{qstats.get('query_miss','?')}",
                  flush=True)

    print("\n" + "=" * 70)
    print(f"{'config':<16}{'mean ms':>10}{'p50 ms':>9}{'p99 ms':>9}"
          f"{'req/s':>8}{'qhit%':>8}")
    for conc in concs:
        for tag in ("OFF", "ON"):
            r = results.get((conc, tag))
            if not r:
                continue
            qh = r.get("query_hit", 0)
            qm = r.get("query_miss", 0)
            qpct = (100 * qh / (qh + qm)) if (qh + qm) else 0
            print(f"bs={conc:<2} {tag:<10}{r['lat_mean']*1000:>10.0f}"
                  f"{r['lat_p50']*1000:>9.0f}{r['lat_p99']*1000:>9.0f}"
                  f"{r['throughput']:>8.2f}{qpct:>8.1f}")
        off, on = results.get((conc, "OFF")), results.get((conc, "ON"))
        if off and on:
            d = (off["lat_mean"] - on["lat_mean"]) / off["lat_mean"] * 100
            print(f"  -> bs={conc} async latency delta: {d:+.1f}% "
                  f"(neg=faster); throughput {on['throughput']/off['throughput']:.3f}x")
    print("=" * 70)
    json.dump({f"{c}_{t}": v for (c, t), v in results.items()},
              open("/tmp/bench_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
