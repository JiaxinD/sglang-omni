# CPU allocator: topology-aware core planning for colocated serving

Host-bound speech serving degrades sharply when other work shares its CPUs:
past a contention threshold an unprotected server can collapse to ~40% of its
clean throughput while a protected one retains ~97% (see the measurement
tables in the introducing PR). The CPU allocator turns the manual
`numactl`/`taskset` recipes into one flag.

## Usage

```bash
sgl-omni serve --model-path <model> --cpu-allocator static
```

Modes:

| mode | behavior |
| --- | --- |
| `off` (default) | No affinity change; identical to today. |
| `static` | Plan once at startup from the NUMA/SMT topology and pin every stage process. |
| `dynamic` | `static`, plus idle exclusive cores are lent to the shared pool and reclaimed with hysteresis. |

At startup the allocator discovers the CPU/NUMA/SMT topology, anchors every
stage process to its GPU's NUMA node, grants whole physical cores (both SMT
siblings together) exclusively to declared serial dispatch loops, and leaves
everything else on the node's shared pool. The plan and any degradations are
logged as one JSON line (`cpu_alloc plan: ...`).

The universe is `sched_getaffinity`, so a container cpuset or an outer
`taskset` bounds the plan. Exclusivity applies to the processes of this
server; keeping *other* tenants off those cores is the job of the outer
cpuset (cgroup, Docker `--cpuset-cpus`, or Kubernetes CPU manager).

## Model declarations

A model opts in by declaring per-stage host costs
(`PipelineConfig.stage_cpu_costs()`); a model without declarations is a
no-op even when the allocator is enabled. Shipped declarations:

| model | declared stages (exclusive physical cores) |
| --- | --- |
| Higgs TTS | tts_engine 1, vocoder 1 |
| Qwen3-ASR | asr 4 |
| Fun-ASR | asr 5 |
| Whisper | asr 4 |
| MOSS-TTS-Local | tts_engine 1, vocoder 1 |
| Qwen3-TTS | tts_engine 2, vocoder 1 |
| Fish S2-Pro | tts_engine 1, vocoder 1 |
| dots.tts | latent_engine 1, vocoder 1 |

## Capacity planning for colocated deployments

How many services fit on one machine safely: sum the exclusive cores of each
service's declaration plus a shared-pool allowance (2+ physical cores per
service), per NUMA node. The plan CLI computes NUMA/SMT-correct partitions:

```bash
# Replica core blocks for a same-GPU DP pool on GPU 0's NUMA node
python -m sglang_omni.cpu_alloc plan --replicas 3 --gpu-id 0
# Full topology dump for audits
python -m sglang_omni.cpu_alloc topology
```

Give each colocated service its own outer cpuset and enable the allocator
inside it:

```bash
# Docker: one lane per service, allocator partitions within the lane
docker run --cpuset-cpus 0-15,112-127 ... \
  sgl-omni serve --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --cpu-allocator static
```

```yaml
# Kubernetes: static CPU manager gives the pod an exclusive cpuset;
# the allocator partitions inside it (requests==limits, integral CPUs).
resources:
  requests: {cpu: "32", memory: "64Gi"}
  limits: {cpu: "32", memory: "64Gi"}
```

`examples/mps_dp/autodp.sh` uses the same planner for its per-replica
`CORE_BLOCKS` automatically.

## Observing contention in production

`GET /host_contention` reports foreign CPU load on the server's allowed
cpuset (the CI cpuset-contention sampling idea, ported to serving):

```json
{
  "cpuset": "0-15,112-127",
  "foreign_busy_cores_last": 0.1,
  "foreign_busy_cores_window_peak": 11.8,
  "foreign_busy_cores_peak": 11.8,
  "own_busy_cores_last": 4.2
}
```

A latency regression with `foreign_busy_cores_window_peak` near zero is a
real regression; one with a large peak is core theft.
