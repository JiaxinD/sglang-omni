# Same-GPU Data Parallelism with CUDA MPS

Same-GPU data parallelism runs several complete serving replicas on one GPU and lets
[CUDA MPS](https://docs.nvidia.com/deploy/mps/index.html) share the GPU between them.
It is a **worker placement strategy, not a routing change**: each replica exposes a
normal worker endpoint, so colocated replicas can be registered behind the
[Omni Router](omni_router.md) (or any load balancer) exactly like workers placed on
separate GPUs. The router only sees endpoints and workloads; it does not need to know
that several of them share one physical GPU.

This is a **conditional optimization**. It can pay off when a properly tuned single
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
  DP is not available. A very small later-replica KV pool reduces batching headroom
  and may contribute to imbalance or instability; inspect each replica's actual
  allocation before driving load.
* **Low-load or latency-critical serving.** Additional replicas may add overhead or
  worsen latency below saturation.

## Tune a single replica first

The single-replica baseline decides whether same-GPU DP is worth it, and an
under-driven baseline makes DP look better than it is. Before comparing anything:

* Sweep client concurrency until throughput plateaus, and inspect scheduler state at
  each step rather than assuming a good operating point.
* Know the generation stage's admission limit. Higgs serves with
  `max_running_requests=64` and `cuda_graph_max_bs=64` by default; both can be raised
  via `sgl-omni serve --max_running_requests N --cuda_graph_max_bs N` (the CUDA-graph
  capture range must cover the admission limit, and raising it costs capture memory).
  Whether the default cap binds depends on the runtime — check, do not assume (see the
  case-study notes).
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
* Sequential (or staggered) replica launch; `MpsRpcFailure` was observed during some
  concurrent colocated launch attempts, and staggered launch reduced exposure in the
  tested setup.

## Select a GPU and NUMA node

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# pick a GPU with util ~0 and only a few MiB used, then find its NUMA node
# via the PCI bus id (drm card ordinals do not always match nvidia-smi ordinals):
GPU_ID=0
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID)
BUS=${BUS,,}; BUS=${BUS:4}
NODE=$(cat /sys/bus/pci/devices/$BUS/numa_node)
numactl -H | grep "node $NODE cpus"     # CPU cores on that node
```

Pin the replicas to cores on the GPU's NUMA node, and keep memory on the same node
(`numactl --cpunodebind=$NODE --membind=$NODE`).

## Start a private MPS daemon

The tested deployment pattern isolates each GPU's replica pool behind its own MPS
**pipe directory** and control daemon; replicas select the physical GPU with
`CUDA_VISIBLE_DEVICES=$GPU_ID` and inherit the pipe directory. The daemon itself is
not device-restricted in this pattern — the pipe directory is the isolation boundary,
and every replica must see the same one or it silently runs without MPS and
time-slices instead.

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # sanity: responds
```

Scaling out across several GPUs of one node was **not validated by this guide's case
study**. In the reviewer's separate verification, one daemon shared across GPUs
collapsed with `cudaErrorMpsRpcFailure`, and one daemon per GPU (daemon started with
`CUDA_VISIBLE_DEVICES=<gpu>`, that GPU's replicas started with `CUDA_VISIBLE_DEVICES=0`
in the daemon's namespace) scaled linearly. Validate daemon isolation separately
before scaling this recipe across several GPUs.

## Launch replicas sequentially

`--mem-fraction-static` sets each replica's generation-stage memory budget. The
fractions are **per-replica requests and are not safely interpreted as additive shares
of total device memory**: with several replicas already resident, a later replica can
end up with a much smaller KV pool than the first one, even with identical flags
(in one verification run, three sequentially launched `mf=0.27` replicas received
97,503 / 53,149 / 20,961 KV tokens). A very small pool reduces batching headroom and
may contribute to imbalance or instability. **After launch, read each replica's log
line `KV Cache is allocated. #tokens: ...` and confirm every replica got a workable
pool**; if the last replica's pool is far smaller, lower the fractions or drop a
replica.

Tested starting points on the pinned 80 GB H100 Higgs configuration below:

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
      --host 127.0.0.1 --port $PORT --model-name higgs > "replica_$i.log" 2>&1 &
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 127.0.0.1:$PORT/health)" = 200 ]; do sleep 6; done
  i=$((i+1))
done
```

`examples/launch_same_gpu_dp.sh` wraps these steps behind a per-run state directory:
it records replica PIDs/process groups, ports, and log paths in a manifest, launches
sequentially behind a bounded health gate, surfaces each replica's KV pool, compares
expected replica processes against the MPS client list, and tears down only the
processes recorded for that run. It is a tested example, not a production process
supervisor.

## Drive every replica to saturation

Aggregate concurrency is not sufficient: **each replica needs enough traffic to reach
its useful batch regime**. With `max_running_requests=64`, a per-replica concurrency
of 16 leaves each replica at a fraction of its admission capacity — this under-drive
was the major confound behind an early "single beats DP3" result. Sweep per-replica
concurrency until aggregate throughput plateaus (one client stream per replica is the
simplest way) and inspect scheduler state (`#running-req`, `#queue-req`) rather than
assuming any fixed concurrency is optimal.

## Verify MPS attachment

Four different things are easy to conflate: MPS environment variables are set; the
daemon is running; an MPS server exists; and **the replica processes you launched are
actually attached as MPS clients**. Only the last one makes the comparison valid, and
a replica that missed the pipe directory falls back to time-slicing without any error.

The launcher compares the two sets for you: it collects each replica's process-group
members, reads `get_server_list` / `get_client_list` from the run's control daemon,
fails with a non-zero exit if any replica has no process in the client list, and
writes the mapping to `mps_attach.txt` in the run's state directory for inspection.

To inspect and confirm manually:

```bash
echo get_server_list | nvidia-cuda-mps-control
for SRV in $(echo get_server_list | nvidia-cuda-mps-control); do
  echo "server $SRV clients:"; echo "get_client_list $SRV" | nvidia-cuda-mps-control
done
ps -o pid,pgid,cmd -p <client pids>     # confirm they are your replica processes
grep -c MpsRpc <your replica logs>      # additional check: must total 0
```

## Route traffic

For deployment, register each replica endpoint with the [Omni Router](omni_router.md)
or another load balancer; colocated replicas are ordinary workers from the router's
point of view. Keep the router's `--max-connections` at least as large as the total
offered concurrency so the upstream connection cap does not throttle the pool.

For performance validation, benchmark the replicas **directly** first (one client per
replica) to isolate the same-GPU placement and MPS effects, then add the router and
measure the end-to-end overhead separately.

## Teardown safely

On a shared host, teardown must only touch processes you launched — never signal PIDs
that are not yours, and never treat "the GPU is empty" as the success condition, since
other tenants' processes may be on the same GPU. Track your replica PIDs (and process
groups) at launch; the stage workers are `multiprocessing-fork` children of each
replica, so signalling the replica's **process group** covers them.

1. Stop new traffic.
2. `SIGTERM` each tracked replica process group (`kill -TERM -- -<pgid>`).
3. Wait until the tracked processes have exited (`pgrep -g <pgid>` empty per replica).
4. Confirm none of your PIDs remain in the MPS client list
   (`get_client_list <server>`); live MPS clients must be gone before the daemon
   quits, or the MPS server can enter an RPC-failure state that outlasts your run.
5. Quit the daemon: `echo quit | nvidia-cuda-mps-control` (with your pipe directory).
6. Only as a last resort, `SIGKILL` tracked process groups that survived the drain —
   and only PIDs recorded in your own launch state.

`examples/launch_same_gpu_dp.sh down` implements exactly this against its recorded
state, and refuses to act when no state is found instead of guessing.

## Memory and stability limits

* **VRAM sets the replica count.** In the pinned Higgs/runtime configuration tested
  here, three replicas at `mf=0.27` fit an 80 GB H100 and a fourth (at `mf≈0.11`)
  failed to get a workable KV pool ("Colocated GPU budget leaves no KV-cache
  headroom"). Treat replica counts and fractions as tested starting points for this
  configuration, not hardware rules.
* **DP3 is operationally tighter than DP2**: smaller per-replica KV pools, less
  balanced per-replica throughput under load, and `MpsRpcFailure` was observed during
  some colocated launch attempts; staggered launch reduced exposure in the tested
  setup.
* **Start with DP2.** In the case study below DP2 captured most of the gain seen in
  nominal runs with lower latency and simpler operation; treat DP3 as something to
  validate locally, not a default.

## H100 Higgs case study

Setup (pin this to reproduce): one H100 80 GB (driver 580.126.20 / CUDA 13), sglang-omni
`a78de4cb3edae4da1c5c49278d77aaa41d01e5b4`, sglang `0.5.12.post1`, model
`bosonai/higgs-tts-3-4b` (snapshot `7556c17e05201fccd9c8cc120bc216dcc7b5d561`),
`/v1/audio/speech`, seed-tts-eval EN, 300 samples per client, default
`max_running_requests=64` / `cuda_graph_max_bs=64`, server cores 0-31 of the GPU's
NUMA node (split per replica for DP), one benchmark client per replica on logical CPUs
64-95 (SMT siblings of the server cores on this host), fresh servers per run,
interleaved rounds on a shared host. Every attempted run is reported.

| Config | Attempted | Started | Completed | Request-error failures | Degraded completed | qps of all completed runs |
|---|---:|---:|---:|---:|---:|---|
| Single c48 (mf 0.85, 32 cores) | 3 | 3 | 3 | 0 | 0 | 21.2–21.9 |
| Single c96 (mf 0.85, 32 cores) | 4 | 4 | 4 | 0 | 0 | 21.7–22.1 (baseline 21.9) |
| DP2 + MPS, 2×c64 (2 × mf 0.42, 16 cores each) | 5 | 4 | 3 | 1 | 0 | 31.5–37.7 |
| DP3 + MPS, 3×c64 (3 × mf 0.27, 10 cores each) | 4 | 3 | 3 | 0 | 1 | 13.3–46.9 (nominal 39.9–46.9) |

Failure and degradation detail: one DP2 run failed mid-benchmark with
`cudaErrorMpsRpcFailure`; one DP2 and one DP3 run had a replica fail to start; these
failures coincided with host-load spikes. One DP3 run completed all requests with zero
request errors but at roughly one third of nominal throughput; it is counted as a
degraded completed run above, not excluded.

Nominal completed DP runs showed substantial throughput gains over the default-cap
single baseline (DP2 1.4–1.7x, DP3 1.8–2.1x; the reviewer's independent one-shot sweep
of the same pinned SHA saw larger ratios), but DP configurations also showed meaningful
launch and runtime fragility on the shared host, while the core-pinned single stayed
within a few percent across all runs. DP3 was not clearly repeatably better than DP2.

Notes:

* **Admission cap, two runtimes.** In our repeated runtime, the default 64/64 single
  was not admission-bound: `#queue-req` remained zero, and raising both limits to
  96/96 or 128/128 did not increase throughput. In the reviewer's separate one-shot
  runtime, the c96 single reached 64 running requests with a standing queue; that
  one-shot DP ratio may therefore include an aggregate-admission advantage in addition
  to MPS overlap. Check which regime your runtime is in before interpreting ratios.
* Absolute qps on a shared host drifts with box load; rely on interleaved
  back-to-back comparisons, and re-measure on your own hardware.
* Counterexample from the same verification effort: MOSS-TTS-Local on the same GPU
  reached a ~13 qps compute ceiling with one replica (~81% util), and DP2/DP3 all
  converged to the same ~13 qps — no peak-throughput gain for a compute-bound model.
  These are setup-specific case studies, not properties of the model families.
* H200, multi-GPU scaling, and production stability are outside this case study's
  validated scope.

## Measure your own setup

Before adopting same-GPU DP, check whether one tuned replica is below GPU saturation
under your real workload:

```bash
# coarse: watch utilization while a saturating client runs
nvidia-smi dmon -i $GPU_ID -s um -d 5
# precise: device-level SM-active over a steady window
nsys profile --gpu-metrics-devices $GPU_ID --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63
```

Low SM activity at the tuned single replica's peak may indicate reclaimable compute
headroom; confirm it with a controlled DP comparison on your own setup before relying
on it. If SM activity is already near the ceiling, stop here. This recipe is young —
measurements from other models, GPUs, and workloads are welcome to harden it.
