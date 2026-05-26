#!/usr/bin/env python
"""Async-decode correctness gate (PR merge hard gate).

Runs Higgs TTS decode with async decode OFF and ON, greedy (true argmax, made
deterministic via scripts/verify_inject), over N prompts x M runs, and asserts
the per-request output_codes are **bit-identical** between the two modes. On
divergence it prints the first mismatching (prompt, run, step, codebook token).

The async lookahead must not change *what* the model produces — only *when* the
host consumes it — so under deterministic sampling OFF and ON must match exactly.

Usage:
    python scripts/verify_correctness.py --gpu 3 --prompts 10 --runs 10 \
        --max-new-tokens 128

Exit code 0 = all bit-identical (PASS); 1 = divergence (FAIL).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import requests

REPO = "/data/sglang-omni"
MODEL = "boson-sglang/higgs-audio-v3-tts-4b-base"
INJECT = f"{REPO}/scripts/verify_inject"


def _load_samples(meta, n, lang):
    sys.path.insert(0, REPO)
    from benchmarks.dataset.seedtts import load_seedtts_samples

    return load_seedtts_samples(meta, n, split=lang)


def _launch_server(port, gpu, async_on, dump_path, log_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SGLANG_OMNI_ENABLE_ASYNC_DECODE"] = "1" if async_on else "0"
    env["SGLANG_OMNI_VERIFY_DUMP"] = dump_path
    env["PYTHONPATH"] = f"{INJECT}:{REPO}"
    open(dump_path, "w").close()  # truncate
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "sglang_omni.cli", "serve",
         "--config", "examples/configs/higgs_tts.yaml", "--port", str(port)],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def _wait_ready(log_path, proc, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server died during startup; see {log_path}")
        try:
            if "Application startup complete" in open(log_path).read():
                return
        except FileNotFoundError:
            pass
        time.sleep(3)
    raise TimeoutError(f"server not ready in {timeout}s; see {log_path}")


def _kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _collect(port, samples, runs, max_new_tokens, dump_path):
    """Send prompts x runs sequentially; return {(p,r): {"sha":..,"codes":[...]}}."""
    url = f"http://127.0.0.1:{port}/v1/audio/speech"
    out = {}
    off = os.path.getsize(dump_path)  # read dump incrementally
    for p, s in enumerate(samples):
        for r in range(runs):
            payload = {
                "model": MODEL, "input": s.target_text,
                "ref_audio": s.ref_audio, "ref_text": s.ref_text,
                "max_new_tokens": max_new_tokens, "temperature": 0.0,
            }
            # The gate is output_codes (what async decode controls). Codes are
            # dumped DURING decode, before vocoding, so capture them even if the
            # (downstream, decode-independent) audio response errors.
            sha, audio_ok = None, True
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                sha = hashlib.sha256(resp.content).hexdigest()
            except Exception as exc:
                audio_ok = False
                print(f"  [warn] audio failed p{p} r{r}: {str(exc)[:80]}", flush=True)
            with open(dump_path) as fh:
                fh.seek(off)
                new = fh.read()
                off = fh.tell()
            codes = [json.loads(ln)["codes"] for ln in new.splitlines() if ln.strip()]
            out[(p, r)] = {"sha": sha, "codes": codes, "audio_ok": audio_ok}
    return out


def _codes_diff(ca, cb, p, r):
    """First divergence between two per-step code lists, or None if identical."""
    if len(ca) != len(cb):
        return f"prompt {p} run {r}: step count differs OFF={len(ca)} ON={len(cb)}"
    for step, (xa, xb) in enumerate(zip(ca, cb)):
        if xa != xb:
            for k, (ta, tb) in enumerate(zip(xa, xb)):
                if ta != tb:
                    return f"prompt {p} run {r} step {step} codebook {k}: OFF={ta} ON={tb}"
            return f"prompt {p} run {r} step {step}: {xa} vs {xb}"
    return None


def _compare(a, b):
    """Gate on output_codes bit-identity (what async decode controls). Audio
    sha is informational only — it can differ across server *processes* due to
    vocoder cuDNN/cuBLAS float nondeterminism even when codes are identical.
    """
    # Sanity: intra-mode determinism (all runs of a prompt must match).
    for tag, res in (("OFF", a), ("ON", b)):
        by_prompt = {}
        for (p, r), v in res.items():
            by_prompt.setdefault(p, []).append((r, v["codes"]))
        for p, runs in by_prompt.items():
            base = runs[0][1]
            for r, codes in runs[1:]:
                d = _codes_diff(base, codes, p, r)
                if d:
                    return False, f"[{tag}] non-deterministic across runs: {d}"

    # Cross-mode: OFF vs ON output_codes must be bit-identical.
    audio_mismatch = audio_failed = 0
    for key in sorted(a):
        p, r = key
        if not a[key]["codes"] or not b[key]["codes"]:
            return False, f"prompt {p} run {r}: no codes captured (server died?)"
        d = _codes_diff(a[key]["codes"], b[key]["codes"], p, r)
        if d:
            return False, d
        sa, sb = a[key]["sha"], b[key]["sha"]
        if sa is None or sb is None:
            audio_failed += 1
        elif sa != sb:
            audio_mismatch += 1
    note = ""
    if audio_mismatch:
        note += (f"  [note: {audio_mismatch}/{len(a)} audio sha differ despite "
                 f"identical codes — cross-process vocoder float nondeterminism]")
    if audio_failed:
        note += (f"  [note: {audio_failed}/{len(a)} audio responses errored "
                 f"(downstream vocoder; decode codes still captured & compared)]")
    return True, f"all {len(a)} (prompt,run) output_codes bit-identical{note}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    samples = _load_samples(args.meta, args.prompts, args.lang)
    print(f"loaded {len(samples)} prompts; runs={args.runs}; "
          f"max_new_tokens={args.max_new_tokens}", flush=True)

    results = {}
    for async_on in (False, True):
        tag = "ON" if async_on else "OFF"
        dump = f"/tmp/codes_{tag.lower()}.jsonl"
        log = f"/tmp/higgs_{tag.lower()}.log"
        print(f"=== launching server async={tag} on GPU{args.gpu} ===", flush=True)
        proc = _launch_server(args.port, args.gpu, async_on, dump, log)
        try:
            _wait_ready(log, proc)
            print(f"  ready; sending {args.prompts}x{args.runs} requests...", flush=True)
            t0 = time.time()
            results[tag] = _collect(args.port, samples, args.runs,
                                    args.max_new_tokens, dump)
            print(f"  done in {time.time()-t0:.0f}s", flush=True)
        finally:
            _kill(proc)
            time.sleep(5)

    ok, msg = _compare(results["OFF"], results["ON"])
    print("\n" + "=" * 60)
    print(f"VERIFY {'PASS' if ok else 'FAIL'}: {msg}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
