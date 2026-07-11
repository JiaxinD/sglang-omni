# Same-GPU Data Parallelism with CUDA MPS

Same-GPU data parallelism runs several complete serving replicas on one GPU and lets
[CUDA MPS](https://docs.nvidia.com/deploy/mps/index.html) share the GPU between them.
It is a **worker placement strategy, not a routing change**: each replica exposes a
normal worker endpoint, so colocated replicas can be registered behind the
[Omni Router](omni_router.md) (or any load balancer) exactly like workers placed on
separate GPUs. The router only sees endpoints and workloads; it does not need to know
that several of them share one physical GPU.

This is a **conditional optimization**. It pays off when a properly tuned single
replica cannot keep the GPU busy; it does little or nothing when one replica already
saturates GPU compute. Measure your own setup before adopting it (see
[Measure your own setup](#measure-your-own-setup)).

## When it helps

Consider same-GPU DP when **all** of the following hold:

* A **properly tuned** single replica (concurrency, admission limits, CPU allocation
  all tuned) still leaves substantial GPU compute idle under your target workload.
  Host-bound or pipeline-bound serving, common for small AR-TTS/ASR models inside a
  multi-stage pipeline, is the typical case.
* The GPU has enough free memory for at least one more full replica (weights + KV
  pool + MPS context).
* You can give each replica its own CPU cores and enough traffic to saturate it.

High-compute GPUs such as H100/H200 are common candidates simply because a small model
leaves more of them idle, but the gate is the measured headroom, not the GPU model.

## When it does not help

* **Compute-bound serving.** A model that already reaches its GPU-compute ceiling with
  one replica gains little or no peak throughput from more replicas; they just split
  the same ceiling (see the MOSS-TTS-Local counterexample in the
  [case study](#h100-higgs-case-study) section).
* **Tight VRAM.** If a second replica does not fit with a workable KV pool, same-GPU
  DP is not available; replicas with starved KV pools crash under load.
* **Low-load or latency-critical serving.** Below saturation a single replica already
  responds faster than a split one.

## Tune a single replica first

The single-replica baseline decides whether same-GPU DP is worth it, and an
under-driven baseline makes DP look better than it is. Before comparing anything:

* Sweep client concurrency well past where throughput stops rising.
* Know the generation stage's admission limit. Higgs serves with
  `max_running_requests=64` and `cuda_graph_max_bs=64` by default; both can be raised
  via `sgl-omni serve --max_running_requests N --cuda_graph_max_bs N` (the CUDA-graph
  capture range must cover the admission limit, and raising it costs capture memory).
* Client concurrency is **not** the active generation batch: requests beyond the
  admission limit wait in the scheduler queue, and requests also spend time in the
  other pipeline stages. Read the scheduler log lines (`#running-req`, `#queue-req`)
  to see the effective regime.
* Give the replica the full CPU core block of the GPU's NUMA node, and keep the load
  generator off those cores.

The tuned single replica's throughput, latency, and GPU utilization are the baseline
every DP configuration has to beat.

## Prerequisites

* NVIDIA CUDA MPS available; GPU compute mode `Default` (check with
  `nvidia-smi --query-gpu=compute_mode --format=csv,noheader`), so a per-user MPS
  daemon needs no root.
* Enough GPU memory for every replica. Each replica is sized with
  `--mem-fraction-static` (for Higgs this works on current `main`); expect a roughly
  fixed per-replica overhead (weights, codec, MPS context) on top of the KV fraction.
* A measured, tuned single-replica baseline that still shows GPU headroom.
* Non-overlapping CPU core blocks, one per replica, on the GPU's NUMA node. Note that
  on SMT machines, logical CPUs `N` and `N + ncores` are often the same physical core;
  check `lscpu -e=CPU,CORE,NODE` before calling core blocks "dedicated".
* Enough offered concurrency to saturate **each** replica, not just the pool.
* Sequential (or staggered) replica launch; concurrent launches can race on memory
  accounting and CUDA-graph capture.

## Select a GPU and NUMA node

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# pick a GPU with util ~0 and only a few MiB used, then find its NUMA node:
GPU_ID=0
NODE=$(cat /sys/class/drm/card$GPU_ID/device/numa_node)
numactl -H | grep "node $NODE cpus"     # CPU cores on that node
```

Pin the replicas to cores on the GPU's NUMA node, and keep memory on the same node
(`numactl --cpunodebind=$NODE --membind=$NODE`).

## Start one MPS daemon per GPU

Run one MPS control daemon per GPU, with a private pipe/log directory. Every replica
on that GPU must see the same `CUDA_MPS_PIPE_DIRECTORY`, or it silently runs without
MPS and time-slices instead.

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # sanity: responds
```

When packing replicas on **several** GPUs of one node, do not share one daemon across
GPUs: in verification a single node-wide daemon spanning multiple GPUs collapsed with
`cudaErrorMpsRpcFailure`. Start one daemon per GPU (daemon started with
`CUDA_VISIBLE_DEVICES=<gpu>`, and that GPU's replicas started with
`CUDA_VISIBLE_DEVICES=0` plus that daemon's pipe directory).

## Launch replicas sequentially

`--mem-fraction-static` sets each replica's generation-stage memory budget. The
fractions are **per-replica requests, not additive shares of the device**: with
several replicas already resident, a later replica profiles less free memory and can
end up with a much smaller KV pool than the first one, even with identical flags
(in one verification run, three sequentially launched `mf=0.27` replicas received
97,503 / 53,149 / 20,961 KV tokens). A starved replica batches less, lags the others,
and can crash under load. **After launch, read each replica's log line
`KV Cache is allocated. #tokens: ...` and confirm every replica got a workable pool**;
if the last replica's pool is far smaller, lower the fractions or drop a replica.

Starting points that fit an 80 GB H100 for Higgs:

| Replicas | `--mem-fraction-static` each | CPU cores each (32-core node) |
|---|---|---|
| 2 | 0.42 | 16 |
| 3 | 0.27 | 10 |

Launch sequentially and wait for `/health` before starting the next replica:

```bash
GPU_ID=0; NODE=0
CORE_BLOCKS=("0-15" "16-31")            # one block per replica, non-overlapping
MF=0.42
i=0
for PORT in 8801 8802; do
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  numactl --cpunodebind=$NODE --membind=$NODE -C "${CORE_BLOCKS[$i]}" \
    sgl-omni serve \
      --model-path bosonai/higgs-tts-3-4b \
      --mem-fraction-static $MF \
      --host 127.0.0.1 --port $PORT --model-name higgs > srv_$PORT.log 2>&1 &
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 127.0.0.1:$PORT/health)" = 200 ]; do sleep 6; done
  i=$((i+1))
done
```

`examples/launch_same_gpu_dp.sh` wraps these steps (MPS daemon, sequential launch,
attach check, safe teardown) behind environment variables, with DP2 as the default.

## Drive every replica to saturation

Aggregate concurrency is not sufficient: **each replica needs enough traffic to reach
its useful batch regime**. With `max_running_requests=64`, a per-replica concurrency
of 16 leaves each replica at a fraction of its admission capacity, and DP can then
look *slower* than a tuned single replica. Drive per-replica concurrency near the
admission limit (one client stream per replica is the simplest way), and confirm the
regime from the scheduler logs (`#running-req` near the cap).

## Verify MPS attachment

Three different things are easy to conflate: "MPS enabled" (env vars set), "MPS daemon
running", and "this replica's processes actually attached as MPS clients". Verify the
last one, per process:

```bash
# an MPS server exists:
echo get_server_list | nvidia-cuda-mps-control
# every replica process (including its stage-worker children) appears as a client:
for SRV in $(echo get_server_list | nvidia-cuda-mps-control); do
  echo "server $SRV clients:"; echo "get_client_list $SRV" | nvidia-cuda-mps-control
done
# and no MPS RPC errors were logged:
grep -c MpsRpc srv_88*.log                        # must total 0 across replica logs
```

Map the client PIDs back to your replica processes (`ps -o pid,cmd -p <pid>`). A
replica that missed the pipe directory falls back to time-slicing without any error,
and the whole comparison is invalid.

## Route traffic

For deployment, register each replica endpoint with the [Omni Router](omni_router.md)
or another load balancer; colocated replicas are ordinary workers from the router's
point of view. Keep the router's `--max-connections` at least as large as the total
offered concurrency so the upstream connection cap does not throttle the pool.

For performance validation, benchmark the replicas **directly** first (one client per
replica) to isolate the same-GPU placement and MPS effects, then add the router and
measure the end-to-end overhead separately.

## Teardown safely

Order matters, and the stage-worker children matter:

1. `SIGTERM` the replicas: `pkill -TERM -f "sgl-omni serve"` (or the launcher PIDs).
2. Wait until the GPU's compute-apps list is empty. Stage workers are
   `multiprocessing-fork` children, not processes named `sgl-omni`; killing only the
   parent orphans them and they keep holding GPU memory, so drain **by GPU
   compute-apps**, not by process name:

   ```bash
   GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU_ID '$1==g{print $2}')
   while nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -q "$GPU_UUID"; do sleep 3; done
   ```

3. Quit the daemon: `echo quit | nvidia-cuda-mps-control`.
4. Only after the daemon is down, `SIGKILL` any stragglers still holding the GPU.

Never `SIGKILL` a live MPS client: it wedges the MPS server with
`cudaErrorMpsRpcFailure` and new clients fail to attach for minutes.

## Memory and stability limits

* **VRAM sets the replica count.** On 80 GB, three Higgs replicas at `mf=0.27` fit; a
  fourth (at `mf≈0.11`) fails to get a workable KV pool ("Colocated GPU budget leaves
  no KV-cache headroom"). Size the count to the VRAM budget, not to ambition.
* **DP3 is operationally tighter than DP2**: smaller per-replica KV pools, less
  balanced per-replica throughput under load, and launch is racy under MPS
  (`MpsRpcFailure` during concurrent CUDA-graph capture — stagger launches and retry).
* **Start with DP2.** In the pinned case study below DP2 captured most of the
  available gain with lower latency and simpler operation; move to DP3 only if a
  sweep on your own setup shows a clearly repeatable further gain.

## H100 Higgs case study

Setup (pin this to reproduce): one H100 80 GB (driver 580.126.20 / CUDA 13), sglang-omni
`a78de4cb`, sglang `0.5.12.post1`, model `bosonai/higgs-tts-3-4b` (snapshot `7556c17e`),
`/v1/audio/speech`, seed-tts-eval EN, 300 samples per client,
`max_running_requests=64` / `cuda_graph_max_bs=64` (defaults), server cores 0-31 of the
GPU's NUMA node (split per replica for DP), one benchmark client per replica on logical
CPUs 64-95 (SMT siblings of the server cores on this host), fresh servers per run,
3 interleaved repetitions per configuration.

| Configuration | Per-replica concurrency | Aggregate qps (clean runs) | p95 latency | vs single (default 64/64) |
|---|---|---|---|---|
| Single replica (mf 0.85, 32 cores) | 48 | 21.5 (21.2–21.9, n=3) | ~3.2 s | — |
| Single replica (mf 0.85, 32 cores) | 96 | 21.9 (21.7–22.1, n=4) | ~5.4 s | baseline |
| DP2 + MPS (2 × mf 0.42, 16 cores each) | 64 | 34.8 (31.5–37.7, n=3 clean; 2 further runs failed under host-load spikes) | ~4.8 s | ~1.4–1.7x |
| DP3 + MPS (3 × mf 0.27, 10 cores each) | 64 | 39.9–46.9 (n=2 clean; 1 collapsed and 1 failed launch reported) | ~6 s | ~1.8–2.1x |

The direction is unanimous — every cleanly completed DP run beat the single baseline —
but the magnitude drifts with host conditions: roughly **1.4–2.1x in our interleaved
repetitions**, and 2.0x / 2.16x (DP2 / DP3) in the reviewer's independent one-shot
sweep of the same pinned SHA (single 29.9 / DP2 59.7 / DP3 64.5). Treat same-GPU DP
here as roughly a doubling opportunity whose realized size depends on the host, and
measure it on your own setup rather than quoting a single ratio.

Notes:

* The baseline is the default `64/64` single configuration driven to its plateau, the
  best stable single point of this sweep. A cap sanity check (64/64, 96/96, 128/128 at
  matching concurrency, CUDA-graph capture lists confirmed per point) left throughput
  unchanged (20.4-21.8 qps) with `#queue-req` always 0, so the admission cap was not
  the single's wall in this runtime. Whether the cap binds is runtime-dependent: the
  reviewer's runtime showed the same single config pinned at 64 running with 32
  queued. The single
  replica at c96 runs against its admission cap (`#running-req` pinned at 64 with a
  standing queue), so DP configurations also have more aggregate admission capacity
  than the single; part of the gain is that capacity, not only MPS overlap.
* DP runs are fragile in ways single runs were not: in our repetitions, DP rounds that
  coincided with host-load spikes failed mid-run with `cudaErrorMpsRpcFailure` or
  collapsed in throughput, while the core-pinned single replica stayed within a few
  percent even at host load 70+. Failed rounds are reported, not discarded.
* DP3 was not clearly better than DP2 here (his one-shot: +8%; our clean runs overlap
  with DP2 once drift is accounted). With a starved third KV pool and extra launch
  fragility, DP2 remains the recommended starting point.
* Absolute qps on a shared host drifts with box load; rely on interleaved
  back-to-back ratios, and re-measure on your own hardware.
* Counterexample from the same verification effort: MOSS-TTS-Local on the same GPU
  reached a ~13 qps compute ceiling with one replica (~81% util), and DP2/DP3 all
  converged to the same ~13 qps — no peak-throughput gain for a compute-bound model.
  These are setup-specific case studies, not properties of the model families.

## Measure your own setup

Before adopting same-GPU DP, confirm one tuned replica is below GPU saturation under
your real workload:

```bash
# coarse: watch utilization while a saturating client runs
nvidia-smi dmon -i $GPU_ID -s um -d 5
# precise: device-level SM-active over a steady window
nsys profile --gpu-metrics-devices $GPU_ID --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63
```

If SM activity sits well below 100% at the tuned single replica's peak, that idle is
what same-GPU DP can reclaim; if it is already near the ceiling, stop here. This
recipe is young — measurements from other models, GPUs, and workloads are welcome to
harden it.
