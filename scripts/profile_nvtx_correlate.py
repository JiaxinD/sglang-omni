#!/usr/bin/env python
"""Full-export NVTX breakdown + CUDA-activity correlation for a CPU-breakdown
nsys report. Forces a complete sqlite export (kernels + runtime + memcpy), then
per NVTX range reports duration distribution and the CUDA activity (kernel busy,
runtime API calls) that overlaps it."""
import os, subprocess, sqlite3, statistics as st, sys
from collections import Counter

rep = sys.argv[1] if len(sys.argv) > 1 else "cpu_bs4_olen128_probe"
base = rep[:-9] if rep.endswith(".nsys-rep") else rep
nsysrep = base + ".nsys-rep"
sqlitep = base + ".full.sqlite"
if not os.path.exists(sqlitep):
    if os.path.exists(sqlitep):
        os.remove(sqlitep)
    print("exporting full sqlite ...", flush=True)
    r = subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", sqlitep, nsysrep], capture_output=True, text=True)
    if r.returncode != 0:
        print("export failed:", r.stderr[-400:]); sys.exit(1)

con = sqlite3.connect(sqlitep); cur = con.cursor()
tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
sid = {r[0]: r[1] for r in cur.execute("SELECT id,value FROM StringIds")}


def ranges(name):
    return sorted((s, e) for s, e in cur.execute(
        "SELECT start,end FROM NVTX_EVENTS WHERE text=?", (name,)))


KERN = next((t for t in tabs if t == "CUPTI_ACTIVITY_KIND_KERNEL"), None)
MEMC = next((t for t in tabs if t == "CUPTI_ACTIVITY_KIND_MEMCPY"), None)
RT = next((t for t in tabs if t == "CUPTI_ACTIVITY_KIND_RUNTIME"), None)
print("kernel table:", KERN, "| memcpy:", MEMC, "| runtime:", RT)

# kernel busy total in trace
if KERN:
    kn = cur.execute(f"SELECT COUNT(*),SUM(end-start) FROM {KERN}").fetchone()
    print(f"kernels total: n={kn[0]} busy={ (kn[1] or 0)/1e3:.1f}us")

names = ["decode_launch", "launch.build_forward_batch", "launch.populate_cg_buffers",
         "launch.extract_sampling_params", "launch.forward", "launch.forward_replay",
         "launch.post_decode_launch", "launch.pack_gpu", "decode_resolve",
         "resolve.collect_host", "collect.read_flags", "collect.pyloop",
         "collect.next_ids_h2d"]


def overlap_busy(table, s, e):
    if not table:
        return 0.0, 0
    rows = cur.execute(
        f"SELECT start,end FROM {table} WHERE end>=? AND start<=?", (s, e)).fetchall()
    busy = sum(min(e, en) - max(s, st) for st, en in rows if en > st)
    return busy, len(rows)


def rt_calls(s, e):
    if not RT:
        return Counter()
    c = Counter()
    for st_, en_, nid in cur.execute(
            f"SELECT start,end,nameId FROM {RT} WHERE start>=? AND start<=?", (s, e)):
        c[sid.get(nid, str(nid))] += 1
    return c


print(f"\n{'range':<32}{'n':>4}{'med us':>9}{'mean us':>9}{'kbusy us':>9}{'nkern':>6}  rtcalls(window-median-step)")
# choose a representative median-duration instance per range for activity detail
for nm in names:
    rs = ranges(nm)
    if not rs:
        continue
    durs = sorted((e - s) / 1e3 for s, e in rs)
    med = st.median(durs)
    mean = st.mean(durs)
    # representative instance closest to median dur
    rep_inst = min(rs, key=lambda x: abs((x[1] - x[0]) / 1e3 - med))
    s, e = rep_inst
    kb, nk = overlap_busy(KERN, s, e)
    mb, nm_ = overlap_busy(MEMC, s, e)
    rc = rt_calls(s, e)
    rcs = " ".join(f"{k.split('_v')[0]}:{v}" for k, v in rc.most_common(4))
    print(f"{nm:<32}{len(rs):>4}{med:>9.1f}{mean:>9.1f}{kb/1e3:>9.1f}{nk:>6}  {rcs}")
