# Same-GPU Data Parallelism

We tend to believe the best DP pratice is to use one GPU per replica. But as we found in [issue 907](https://github.com/sgl-project/sglang-omni/issues/907), using multiple replicas on the same GPU can achieve even better throughput, up to a scary +134% throughput gain on H100.

> Serving on H100 is host / CPU-dispatch-bound, not GPU-bound [issue 907](https://github.com/sgl-project/sglang-omni/issues/907): the GPU sits ~72% idle at single-replica saturation (SM ~28%). Same-GPU data parallelism + CUDA MPS recovers most of that idle (DP3 + MPS = +114% [measured], and +134% in db-ol's independent reproduction on [issue 912](https://github.com/sgl-project/sglang-omni/issues/912)), but a single-process Python router eats most of the gain (via-router DP3 lands below via-router DP2, with the router process pinned at 88 to 96% of one core). This roadmap runs a bridge and a destination in parallel: the bridge is a Python multi-process router (control-plane / data-plane split) that lands as a short win (Phase 1); the destination is a Rust gateway developed alongside it (Phase 2); they converge at switchover (Phase 3), after which the Python relay is retired and possibly reused for RL. Same-GPU DP recipes run as a cross-cutting track.

In this sense, we share this conclusion with the SGLang Omni community, and make temperate adjustments in this recipe to provide best practices in powering up same-GPU DP for the best ever throughput. We are still diving deep into router and SGLang Omni Scheduler refactor to further explore the best practices of prodcuting serving of TTS models. Please reach out to us if you are also interested in this topic.

## When to use this

Run several complete TTS replicas on one GPU behind CUDA MPS to reclaim the GPU that a single replica leaves idle. This is orthogonal to the [Omni
Router](omni_router.md): the router fronts whole workers (roughly one per GPU), while this guide packs several replicas inside one GPU. You can use both together.

【TODO：其实我觉得这里很奇怪，应该说 router 的意义只在于最快的链接各个 worker，逻辑上 router 只需要做好每个同等物理性质的 worker 的分发，而不需要在乎 worker 实际上的 placement，所以这句话应该说，我们默认 router 启动 sglang omni server 时，会让单个 server 占据完整的 GPU，但是逻辑上 router 只关心 worker 的 workload，不需要关心 router 的 placement，所以你仍旧可以用 sglang omni router 来管理这些多卡的 DP worker，而且每个 GPU 上同时存在多个 worker。】

TTS serving is host-bound. Under a saturating client, a single Higgs replica keeps the GPU busy only about 30% of the time (device SM occupancy near 30%, so roughly
70% idle); the limit is host-side request dispatch, not GPU compute. If you launch one replica and see low GPU utilization that does not rise with more concurrency, that idle headroom is reclaimable.

【说实话，从读者的角度来看，这一个段落的两句话都是有道理的，但是似乎都没有解释 when to use this。真正回答 when to use this 的，应该这么说，“你在用 H100/H200 这种单卡算力很高的 GPU 运行 TTS 服务，可以考虑使用 same-GPU DP 来大幅提升 throughput。然后，解释道，我们相信这种 same GPU DP 对于 TTS ASR 这种小模型在高显存和算力显卡上普遍具有显著意义，但是时间有限，我们能够提供的 same-GPU DP 的 recipe 还不够完善，希望社区能够帮助我们完善这个 recipe，让更多的用户能够受益。”】

## How it works

Put more replicas on the same card. Without MPS the replicas time-slice one GPU context and barely help. With CUDA MPS their contexts co-reside on the SMs, so
each replica's kernels fill the others' host-idle gaps and aggregate throughput scales until the card runs out of memory or actually saturates.

【TODO：这里需要解释一下什么是 MPS。】

## Prerequisites

* A free GPU (`util ~0`, a few MiB used).
* GPU compute mode `Default` (the usual setting), so a per-user MPS daemon needs no root: `nvidia-smi --query-gpu=compute_mode --format=csv,noheader`.
* A per-replica KV-pool knob. Each replica is sized with `--mem-fraction-static`.
  For Higgs this requires the companion PR that adds `mem_fraction_role_to_stage`
  (PR #977, merge it first); on a build without it, Higgs pins the AR pool to the
  `0.85` default in `HiggsTtsEngineBuilder.generation_defaults` and rejects the
  flag. MOSS-TTS already accepts `--mem-fraction-static`.

【TODO：我感觉前置条件还有，比如说高算力显卡 etc，然后我们肯定得先把 Higgs 的 PR 合入了，所以不用刻意强调 higgs 的麻烦。】

## Launch Commands

### 1. Pick a card and its NUMA node

【TODO：很多 markdown 排版器会自动给次级标题加序号，所以不要在标题里面有 1.，然后 card 这个词很老中啊，真的有这种用法么😂，用 GPU 不是更好？】

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# pick a card that is util ~0 and memory.used ~4 MiB, then set it and its NUMA node:
CARD=2
NODE=$(cat /sys/class/drm/card$CARD/device/numa_node)
numactl -H | grep "node $NODE cpus"                 # cores on that node
SRV_CORES=0-15                                       # a block of those cores for the replicas
```

The replicas are pinned to cores on the card's NUMA node.

### 2. Start a per-user MPS daemon

Every replica must point at the same `CUDA_MPS_PIPE_DIRECTORY`, or it silently runs without MPS and the comparison is meaningless.

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # sanity: responds
```

### 3. Launch N replicas, one at a time

`--mem-fraction-static` is the fraction of TOTAL device memory each replica reserves for weights plus KV pool. Three replicas at `0.27` (0.81 total) fit an 80 GB card with room for the MPS contexts. Launch them **sequentially**, waiting for each `/health` before starting the next, so memory accounting is deterministic.

```bash
for PORT in 8801 8802 8803; do
  CUDA_VISIBLE_DEVICES=$CARD \
  numactl --cpunodebind=$NODE --membind=$NODE -C $SRV_CORES \
    sgl-omni serve \
      --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
      --mem-fraction-static 0.27 \
      --host 127.0.0.1 --port $PORT --model-name higgs > srv_$PORT.log 2>&1 &
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 127.0.0.1:$PORT/health)" = 200 ]; do sleep 6; done
done
```

The `sgl-omni serve` processes inherit `CUDA_MPS_PIPE_DIRECTORY` exported above, so they attach to the daemon.

`examples/launch_same_gpu_dp.sh` wraps these steps (start MPS, sequential launch, attach check, teardown) behind environment variables.

### 4. Verify MPS attached

A replica that missed the pipe directory falls back to time-slicing. Confirm both:

```bash
echo get_server_list | nvidia-cuda-mps-control    # lists the MPS server
grep -c MpsRpc srv_88*.log                         # must total 0
```

### 5. Serve

Send traffic to `127.0.0.1:8801..8803` directly (one client stream per replica), or front the replicas with the [Omni Router](omni_router.md). Behind the router, keep `--max-connections` at least as large as the total offered concurrency so the upstream connection cap does not throttle the pool.

【这里要不建议用户直接把 8801 ... 都连接到 router 上，然后发给 router。然后备注下，说我们的 router 目前性能不佳，但是很方便，你们也可以自己写 routing 策略，发送到指定的 server 上。】

## Teardown

Order matters. Never `kill -9` a live MPS client: it wedges the MPS server for minutes and new clients fail to attach. Stop the replicas with `SIGTERM`, wait for the card to drain, then quit the daemon.

```bash
pkill -TERM -f "sgl-omni serve"
CARD_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c=$CARD '$1==c{print $2}')
# wait until this card has no compute-apps left, then quit the daemon:
while nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -q "$CARD_UUID"; do sleep 3; done
echo quit | nvidia-cuda-mps-control
```

## Limits

* **The ceiling is memory, not GPU compute.** Three replicas fit an 80 GB card; a fourth does not. On a 141 GB card four replicas fit and still pay off, and the card then approaches real GPU saturation (~95% util). Size the replica count to the VRAM budget.
* **The gain tracks the single-replica idle.** A model that already saturates one replica has no headroom and will not benefit. Measure first (below).
* **maxconn** must cover the pool's concurrency when a router fronts it.

## What to expect

Measured on one node, throughput-only, three repeats per point, one client stream per replica at saturation. The ratios are the stable claim; absolute qps varies with checkpoint and workload.

| Card | DP1 | DP3 + MPS | DP4 + MPS |
|---|---|---|---|
| H100 80 GB | base | +126% | does not fit |
| H200 141 GB | base | +172% | +234% |

MPS is the unlock: without it, same-card replicas time-slice and DP1 to DP3 gains only about +55%. The gain is larger on the bigger card because a single host-bound replica leaves proportionally more of it idle.

## Measure your own idle

Before assuming same-GPU DP helps a model, confirm one replica is below saturation. Launch one replica, drive it with a saturating client, and sample device metrics:

```bash
nsys profile --gpu-metrics-devices $CARD --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63
```

Open `one_replica.nsys-rep` and read the SM-active row. If it sits well below 100% under load, that idle fraction is what same-GPU DP reclaims.
