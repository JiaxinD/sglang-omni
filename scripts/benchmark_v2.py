#!/usr/bin/env python
"""Async decode benchmark v2 — statistically firm bs=1 / bs=4 numbers.

Differences from benchmark_async.py (which used production sampling, variable
output length, 2 runs -> ballpark):
  - GREEDY (temperature=0): with the T4 batched-sampler argmax short-circuit,
    production greedy is deterministic, so per-prompt output length is FIXED ->
    no length-jitter contaminating wall time.
  - FIXED max_new_tokens=128.
  - 10 runs/config (run = 20 requests at concurrency=bs); run #1 = warmup,
    discarded; stats over the remaining 9 runs.
  - mean +/- std + 95% CI (Student-t), and an OFF-vs-ON delta with a Welch CI so
    "significant?" = does the CI cross zero.

Reuses benchmark_async's server launch/teardown + bench_inject query-hit dumper.
Matrix is fixed (bs=1, bs=4); do NOT extend here.

Usage:
    python scripts/benchmark_v2.py --gpu 1 --port 8270
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from scipy import stats

REPO = "/data/sglang-omni"
sys.path.insert(0, f"{REPO}/scripts")
from benchmark_async import MODEL, _kill, _launch, _samples, _wait  # noqa: E402

RUNS = 10            # runs per config; run #1 discarded as warmup
REQS_PER_RUN = 20
MNT = 128            # fixed output cap


def _one(url, s):
    payload = {"model": MODEL, "input": s.target_text, "ref_audio": s.ref_audio,
               "ref_text": s.ref_text, "max_new_tokens": MNT,
               "temperature": 0.0}  # greedy -> deterministic, fixed length
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=300)
    dt = time.time() - t0
    r.raise_for_status()
    return dt


def _run_once(url, samples, conc):
    reqs = [samples[i % len(samples)] for i in range(REQS_PER_RUN)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        lat = list(ex.map(lambda s: _one(url, s), reqs))
    wall = time.time() - t0
    return statistics.mean(lat) * 1000, REQS_PER_RUN / wall, lat  # ms, req/s, raw


def _ci95(xs):
    xs = np.asarray(xs, float)
    n = len(xs)
    m = xs.mean()
    sd = xs.std(ddof=1)
    se = sd / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, sd, (m - h, m + h)


def _delta_ci(off, on):
    """OFF - ON (positive => ON faster/lower). Welch two-sample 95% CI + p."""
    off, on = np.asarray(off, float), np.asarray(on, float)
    d = off.mean() - on.mean()
    va, vb, na, nb = off.var(ddof=1), on.var(ddof=1), len(off), len(on)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return d, (d, d), float("inf"), 1.0
    dof = se**4 / ((va / na)**2 / (na - 1) + (vb / nb)**2 / (nb - 1))
    h = stats.t.ppf(0.975, dof) * se
    _, p = stats.ttest_ind(off, on, equal_var=False)
    return d, (d - h, d + h), dof, p


def _bench_config(gpu, port, conc, async_on, samples):
    tag = "ON" if async_on else "OFF"
    stats_path = f"/tmp/v2_stats_bs{conc}_{tag}.json"
    log = f"/tmp/v2_bs{conc}_{tag}.log"
    print(f"\n=== bs={conc} async={tag} (GPU{gpu}, greedy, olen={MNT}) ===",
          flush=True)
    proc = _launch(port, gpu, async_on, stats_path, log)
    lat_means, thrpts, all_lat = [], [], []
    try:
        _wait(log, proc)
        url = f"http://127.0.0.1:{port}/v1/audio/speech"
        for r in range(RUNS):
            m, tp, raw = _run_once(url, samples, conc)
            warm = " (warmup, discarded)" if r == 0 else ""
            print(f"  run {r:2d}: mean_lat={m:7.1f}ms thrpt={tp:5.2f}req/s{warm}",
                  flush=True)
            if r > 0:
                lat_means.append(m)
                thrpts.append(tp)
                all_lat.extend(raw)
    finally:
        _kill(proc)
        time.sleep(5)
    qstats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    return {"lat_means": lat_means, "thrpts": thrpts,
            "p50": float(np.percentile([x * 1000 for x in all_lat], 50)),
            "p99": float(np.percentile([x * 1000 for x in all_lat], 99)),
            "n_req": len(all_lat), **qstats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--port", type=int, default=8270)
    args = ap.parse_args()

    samples = _samples(8)
    res = {}
    for conc in (1, 4):
        for async_on in (False, True):
            res[(conc, "ON" if async_on else "OFF")] = _bench_config(
                args.gpu, args.port, conc, async_on, samples)

    out = {"runs_kept": RUNS - 1, "reqs_per_run": REQS_PER_RUN, "olen": MNT,
           "sampling": "greedy(temperature=0)", "configs": {}}
    print("\n" + "=" * 78)
    for conc in (1, 4):
        for tag in ("OFF", "ON"):
            r = res[(conc, tag)]
            m, sd, ci = _ci95(r["lat_means"])
            tm, tsd, tci = _ci95(r["thrpts"])
            qh, qm = r.get("query_hit", 0), r.get("query_miss", 0)
            qpct = 100 * qh / (qh + qm) if (qh + qm) else 0
            print(f"bs={conc} {tag:<3} lat={m:6.1f}+/-{sd:4.1f}ms "
                  f"CI95[{ci[0]:.1f},{ci[1]:.1f}] p50={r['p50']:.0f} p99={r['p99']:.0f} "
                  f"thrpt={tm:.2f}+/-{tsd:.2f} qhit={qpct:.0f}%({qh}/{qh+qm})",
                  flush=True)
            out["configs"][f"bs{conc}_{tag}"] = {
                "lat_means_ms": r["lat_means"], "thrpts": r["thrpts"],
                "lat_mean": m, "lat_std": sd, "lat_ci95": list(ci),
                "p50_ms": r["p50"], "p99_ms": r["p99"],
                "thrpt_mean": tm, "thrpt_std": tsd, "thrpt_ci95": list(tci),
                "query_hit": qh, "query_miss": qm, "query_hit_pct": qpct,
                "n_req": r["n_req"]}
    print("-" * 78)
    for conc in (1, 4):
        off, on = res[(conc, "OFF")], res[(conc, "ON")]
        d, ci, dof, p = _delta_ci(off["lat_means"], on["lat_means"])
        offm = np.mean(off["lat_means"])
        dpct, cipct = 100 * d / offm, (100 * ci[0] / offm, 100 * ci[1] / offm)
        td, tci, _, tp = _delta_ci(on["thrpts"], off["thrpts"])  # ON-OFF for thrpt
        sig = "SIGNIFICANT" if (ci[0] > 0 or ci[1] < 0) else "NOT significant (CI crosses 0)"
        print(f"bs={conc} OFF-ON latency delta = {d:+.1f}ms ({dpct:+.1f}%), "
              f"95%CI[{cipct[0]:+.1f}%,{cipct[1]:+.1f}%] p={p:.4f} -> {sig}", flush=True)
        out["configs"][f"bs{conc}_delta"] = {
            "lat_delta_ms": d, "lat_delta_pct": dpct,
            "lat_delta_ci95_pct": list(cipct), "welch_dof": dof, "p_value": p,
            "significant": bool(ci[0] > 0 or ci[1] < 0),
            "thrpt_ratio": float(np.mean(on["thrpts"]) / np.mean(off["thrpts"]))}
    print("=" * 78)
    json.dump(out, open("/tmp/v2_results.json", "w"), indent=2)
    print("wrote /tmp/v2_results.json")


if __name__ == "__main__":
    main()
