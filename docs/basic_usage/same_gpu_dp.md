# Same-GPU Data Parallelism with CUDA MPS

> TL;DR: Use CUDA MPS to colocate several full serving replicas on one GPU and reclaim host-dispatch idle. This boost performance of H100 to +134% on TTS models.

We tend to believe the best DP pratice is to use one GPU per replica. But as we found in [issue 907](https://github.com/sgl-project/sglang-omni/issues/907), using multiple replicas on the same GPU can achieve even better throughput.

Same-GPU data parallelism runs several complete serving replicas on one GPU and lets [CUDA MPS](https://docs.nvidia.com/deploy/mps/index.html) share the GPU between them. It is a worker placement strategy, not a routing change: each still replica exposes a normal worker endpoint, so colocated replicas can be registered behind our [Omni Router](omni_router.md) or any load balancer exactly like workers on separate GPUs. The router only sees endpoints; it does not need to know that several of them share one physical GPU.

This is a conditional and ongoing optimization. It can pay off when a properly tuned single replica cannot keep the GPU busy (say on H100, H200 etc), and it does little or nothing when one replica already saturates GPU compute (possibly on 4090, 5090 etc).

In this sense, we share this conclusion with the SGLang Omni community, and make temperate adjustments in this recipe to provide best practices in powering up same-GPU DP for the best ever throughput. We are still diving deep into router and SGLang Omni Scheduler refactor to further explore the best practices of prodcuting serving of TTS models. Please reach out to us if you are also interested in this topic.

## Deploy

The steps below are one continuous flow. `examples/launch_same_gpu_dp.sh` wraps them behind a per-run state directory (`up/verify/down/list`): it records replica PIDs, process groups, ports, and logs, refuses to start when ports are taken or another run is recorded on the same GPU, health-gates sequential startup, checks each replica's KV pool, compares expected replica processes against the MPS client list, and tears down only the processes it recorded. Due to the complexity of hardware variance, `examples/launch_same_gpu_dp.sh` is only tested example. Detailed instructions are as follows:

1. **Choose the GPU and NUMA node.**

Pick a GPU that is idle, then find its NUMA node from the PCI bus id (drm card ordinals do not always match nvidia-smi ordinals), and pin replicas and memory to that node.

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
GPU_ID=0
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID)
BUS=${BUS,,}; BUS=${BUS:4}
NODE=$(cat /sys/bus/pci/devices/$BUS/numa_node)   # if -1, set the node explicitly
numactl -H | grep "node $NODE cpus"
```

【这里能不能提到多卡的扩展，因为基本描述的都是单卡操作】

2. **Start a private MPS daemon.**

Isolate each GPU's replica pool behind its own MPS pipe directory. Every replica must see the same one, or it silently runs without MPS and time-slices instead. Scaling across several GPUs was not validated here; use one daemon per GPU and validate isolation separately before doing so.

【我们其实是验证了多卡的吧，结论就说，每张卡需要单独启动 MPS daemon。】

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER-gpu$GPU_ID/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # sanity: responds
```

3. **Launch replicas sequentially, and check each KV pool.** 

`--mem-fraction-static` is a per-replica request, not an additive share of total memory: with replicas already resident, a later replica can get a much smaller KV pool even with identical flags (in one run, three sequential `mf=0.27` replicas received 97,503 / 53,149 / 20,961 KV tokens). Launch one at a time, wait for `/health`, and confirm each replica's `KV Cache is allocated. #tokens: ...` line is workable. Tested starting points on the 80 GB H100 Higgs configuration are 2 replicas at `mf=0.42` with 16 cores each, or 3 at `mf=0.27` with 10 cores each.

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

【我发现一个非常严肃的问题，就像这里说的，mfs 不是全局 HBM 占比，所以逻辑上，如果我们希望每个 replica 尽量等同的话，先后启动的 replica 的 mfs 应该是递增的。当然，回到另一个问题，如果我们现在直接启用 DP 3，每个都是 0.42 的话，事实上会让这些 replica 的 KV pool 不等价。数学直觉告诉我，最好的 DP 策略就是“ 物理性质等同的 replica”，而不是仅仅功能等同的 replica。一方面，如果我们目前是按顺序启动，且每个 replica 的 mfs 一致，我们需要标注出来，comment 说这不是目前的最优策略，但是即便如此，我们的策略已经带来了恐怖的提升。另一方面，我们应该列个 todo，大概率需要设置一些新的参数，比如 global kv size 之类的，来 overwrite 每个 replica 的 mfs，让其物理等同。这会是一个非常值得研究的事情。或者说，证明最佳的 DP 策略就是“物理等同，每个 replica KV cache 完全一致”。】

4. **Drive every replica to saturation.** Aggregate concurrency is not enough: each replica needs enough traffic to reach its useful batch. With `max_running_requests=64`, a per-replica concurrency of 16 leaves each replica far below its admission capacity, which was the main confound behind an early "single beats DP3" result. Sweep per-replica concurrency, one client stream per replica, until aggregate throughput plateaus, and inspect `#running-req` / `#queue-req`.

5. **Verify MPS attachment.** Four things are easy to conflate: env vars set, daemon running, an MPS server exists, and the replica processes you launched are actually attached as clients. Only the last makes the comparison valid, and a replica that missed the pipe directory falls back to time-slicing without any error. The launcher writes the server-to-client PID mapping to `mps_attach.txt` and fails if any replica has no attached client; to check manually:

   ```bash
   echo get_server_list | nvidia-cuda-mps-control
   for SRV in $(echo get_server_list | nvidia-cuda-mps-control); do
     echo "server $SRV clients:"; echo "get_client_list $SRV" | nvidia-cuda-mps-control
   done
   ps -o pid,pgid,cmd -p <client pids>     # confirm they are your replica processes
   grep -c MpsRpc <your replica logs>      # must total 0
   ```

6. **Route traffic.** For deployment, register each replica endpoint with the [Omni Router](omni_router.md) or another load balancer; colocated replicas are ordinary workers to the router. Keep the router's `--max-connections` at least as large as the total offered concurrency. For performance validation, benchmark the replicas directly first (one client per replica) to isolate the placement and MPS effects, then add the router and measure its overhead separately.

7. **Tear down safely.** On a shared host, only touch processes you launched, and never treat "the GPU is empty" as the success condition. Stop new traffic, then SIGTERM each tracked replica process group (`kill -TERM -- -<pgid>`, which also reaps the `multiprocessing-fork` stage workers) and wait until they exit. Confirm the MPS client list is empty (`get_client_list <server>`): the pipe is private to your run, so any remaining client is outstanding work even when its PID no longer matches a tracked group, and live clients must be gone before the daemon quits, or the MPS server can enter an RPC-failure state that outlasts your run. Only then quit the daemon (`echo quit | nvidia-cuda-mps-control`), and SIGKILL surviving tracked groups only as a last resort. `examples/launch_same_gpu_dp.sh down` follows this order and keeps the state directory whenever cleanup cannot be confirmed.

【好奇，这个相当于是详细的解释如何用对吧，然后 examples/launch_same_gpu_dp.sh 就是傻瓜式的一键启动脚本？我们这里其实没有区分 H100 or H200，我反而觉得是应该写清楚的，上方的例子应该是 H100，H200 需要将 xxx 改为 yyy】


## Why it works: how we found this

We tend to believe the best DP pratice is to use one GPU per replica. But as we found in [issue 907](https://github.com/sgl-project/sglang-omni/issues/907), using multiple replicas on the same GPU can achieve even better throughput, up to a scary +134% throughput gain on H100.

In this sense, we share this conclusion with the SGLang Omni community, and make temperate adjustments in this recipe to provide best practices in powering up same-GPU DP for the best ever throughput. We are still diving deep into router and SGLang Omni Scheduler refactor to further explore the best practices of prodcuting serving of TTS models. Please reach out to us if you are also interested in this topic.

As we said, this recipe grew out of the serving profiling in [#907](https://github.com/sgl-project/sglang-omni/issues/907), which found that several omni serving workloads are limited by host-side dispatch rather than GPU compute. From there we ran same-GPU DP experiments on [Higgs](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html) and [Moss](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html) TTS models.

The evidence for when replication helps is a spectrum, not a single proof:

| Experiment | GPU signal | Controlled observation | Result | Interpretation |
|---|---|---|---|---|
| ASR single replica | GPU timeline 94.3% idle | throughput 0.90x at SM clock 0.455x; 0.31x at host CPU near 0.25x | sensitive to CPU, not to GPU compute | strong host-dispatch-bound causal evidence in this ASR setup |
| Higgs tuned single | SM Active about 29%, GPU idle about 71% | throughput plateaued, worker fully driven | 1.00x normalized | clear reclaimable GPU headroom, but not the full ASR causal closure |
| Higgs DP2 without MPS | SM Active about 37 to 38%, GPU idle about 62 to 63% | added a second same-card server process | about 1.24x normalized | the second process reclaims part of the idle gap; host scheduling and long-tail batching can both contribute |
| Higgs DP with MPS | see the pinned case study in Evaluate | each replica saturated, MPS attachment confirmed | 1.4 to 2.1x nominal, repeated | MPS provided the main additional gain in this setup |

ASR is the strongest host-bound evidence. Higgs started as a gray zone but clearly leaves GPU headroom at a tuned single replica. Running several replicas as separate processes changes host execution, scheduling, and long-tail behavior, and it is not the same as enlarging one replica's batch. Without MPS the CUDA contexts mostly time-slice and recover only part of the idle; MPS lets kernels from different processes run concurrently when resources permit, which is where the main extra gain came from.

## Prepare the baseline

The single-replica baseline decides whether same-GPU DP is worth it, and an under-driven baseline makes DP look better than it is. Tune and measure one replica first, then treat its throughput, latency, and GPU utilization as the number every DP configuration has to beat.

* **Sweep concurrency to the plateau.** Raise client concurrency until throughput stops climbing, and read the scheduler log lines (`#running-req`, `#queue-req`) at each step rather than assuming a good operating point.
* **Know the admission limit.** Higgs serves with `max_running_requests=64` and `cuda_graph_max_bs=64` by default; both can be raised via `sgl-omni serve --max_running_requests N --cuda_graph_max_bs N` (the CUDA-graph capture range must cover the admission limit, and raising it costs capture memory). Whether the default cap binds depends on the runtime, so check the queue, do not assume.
* **Separate client from server.** Client concurrency is not the active generation batch: requests beyond the admission limit wait in the scheduler queue, and requests also spend time in the other pipeline stages.
* **Prerequisites.** NVIDIA CUDA MPS available with GPU compute mode `Default`, so a per-user daemon needs no root; enough GPU memory for every replica, each sized with `--mem-fraction-static` plus a roughly fixed per-replica overhead (weights, codec, MPS context); non-overlapping CPU core blocks, one per replica, on the GPU's NUMA node (on SMT machines logical CPUs `N` and `N + ncores` are often the same physical core, so check `lscpu -e=CPU,CORE,NODE`); and enough offered concurrency to saturate each replica, not just the pool.


## Evaluate

Whether same-GPU DP helps is easy to measure wrong, so hold the comparison to the same discipline for every configuration:

| Control | Why it matters |
|---|---|
| tune the single replica to its throughput plateau | keeps the baseline from being artificially weak |
| hold total GPU and CPU resources fixed | separates replica splitting from simply adding resources |
| give each replica dedicated CPU cores | keeps replicas from contending for host dispatch |
| saturate each replica separately | keeps the DP pool from being under-fed |
| pin software and runtime settings | makes the comparison reproducible |
| report latency and unsuccessful runs | avoids showing only the best throughput |

An early comparison did not drive every configuration equally: each DP replica was left below its saturation regime, so the pool looked slower than one replica. After the environment was pinned and each replica saturated, the gain held up.

**H100 Higgs case study.** One H100 80 GB (driver 580.126.20 / CUDA 13), sglang-omni `a78de4cb`, sglang `0.5.12.post1`, `bosonai/higgs-tts-3-4b` (snapshot `7556c17e`), `/v1/audio/speech`, seed-tts-eval EN, 300 samples per client, default `max_running_requests=64` / `cuda_graph_max_bs=64`, 32 server cores of the GPU's NUMA node split per replica, one client per replica on the SMT-sibling cores, fresh servers per run, interleaved on a shared host. Every attempted run is reported.

| Configuration | Nominal throughput | Relative to single | Run outcome |
|---|---:|---:|---|
| Single c96 | 21.7 to 22.1 qps | 1.0x | 4/4 completed |
| DP2 + MPS, 2 x c64 | 31.5 to 37.7 qps | 1.4 to 1.7x | 3 nominal of 5 attempts |
| DP3 + MPS, 3 x c64 | 39.9 to 46.9 qps | 1.8 to 2.1x | 2 nominal and 1 degraded of 4 attempts |

The failures: one DP2 benchmark run hit `cudaErrorMpsRpcFailure`, and one DP2 and one DP3 replica failed to start, all coinciding with host-load spikes. One DP3 run completed every request but at 13.3 qps, so it is marked degraded rather than excluded. The core-pinned single stayed within a few percent across all runs, and DP3 was not clearly repeatably better than DP2.

The #907 profiling, this repeated case study, and the reviewer verification below are three separate measurement series. They ran on different dates and load, and in some cases different software, so they should not be compared by absolute QPS; the differences between roughly 61, 21, and 29.9 qps are not attributed to a single cause.

> A separate reviewer verification on the same pinned software revision measured 29.9, 59.7, and 64.5 qps for single, DP2, and DP3. Absolute throughput differed between the two runtime environments, including different observed admission behavior, so the two series should not be combined. Both nevertheless showed a clear DP gain once every configuration was saturated.

To measure your own setup, check whether one tuned replica is below GPU saturation under your real workload before adopting DP:

```bash
nvidia-smi dmon -i $GPU_ID -s um -d 5                        # coarse utilization
nsys profile --gpu-metrics-devices $GPU_ID --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63                      # device-level SM-active
```

Low SM activity at the tuned single replica's peak may indicate reclaimable headroom; confirm it with a controlled DP comparison before relying on it. If SM activity is already near the ceiling, stop here.

## Limits and next steps

VRAM sets the replica count. In the pinned Higgs configuration, three replicas at `mf=0.27` fit an 80 GB H100 and a fourth (at `mf` near 0.11) failed to get a workable KV pool ("Colocated GPU budget leaves no KV-cache headroom"); treat replica counts and fractions as tested starting points, not hardware rules. DP3 is operationally tighter than DP2, with smaller per-replica KV pools, less balanced throughput under load, and `MpsRpcFailure` observed during some concurrent launches, so start with DP2 and treat DP3 as something to validate locally.

This is not a universal optimization. A compute-bound model gains little: on the same GPU, MOSS-TTS-Local reached a compute ceiling near 13 qps with one replica (util about 81%), and DP2 and DP3 all converged to the same value with no peak-throughput gain. H200, multi-GPU scaling, and production stability are outside this case study's validated scope, and these are setup-specific results rather than properties of the model families.

Same-GPU DP with MPS is a practical way to recover idle GPU time today, and it also makes clearer where the next work is. Two directions are directly connected. The **router** has to keep every colocated worker fed without becoming the throughput bottleneck, and it needs predictable admission and failure behavior under overload. The **scheduler and runtime** could reduce the serial host work and coordinate several execution streams more directly, so the runtime reclaims that headroom without relying solely on process-level replication. These are ongoing directions, and the recipe stays useful in the meantime. Independent measurements and fixes from other models, GPUs, and workloads are welcome.

