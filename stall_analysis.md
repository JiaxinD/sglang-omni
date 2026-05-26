# Stall Analysis — Higgs TTS 异步 decode（T2，nsys decode-isolated）

工具：`scripts/profile_async.py` + `scripts/profile_inject`（NVTX + cudaProfilerApi
capture range，零生产代码改动）。硬件：ion9 H200 单卡，async ON，production
greedy（temp=0，已确定性，commit `12dbdfb`）。每个配置发 C 个同 prompt 并发请求
形成 bs=C 的稳态 decode，截取 decode 中段 ~30 步窗口，用 `nsys stats
cuda_gpu_trace` 在合并后的 GPU 时间线上算逐核 gap。

## TL;DR

**(b) — 没有 Plan B（multi-stream）能救的 stall。**

每个配置确实存在巨大的串行 GPU 空闲（每 decode 步约 **3.8ms** 的 gap、占壁钟
~98%），**但这段空闲不是 Plan B 的靶点**。Plan B 是把 D2H/collect 挪到 alt
stream、用 `wait_event` 与 forward 重叠；而这里 (1) forward 本身的 GPU 计算只有
~50µs，(2) D2H/collect 已经被 Plan A 藏在 forward 之后（`query_hit=100%`）。这
3.8ms 的 gap 是**两次 forward 之间的 CPU 每步开销**（`get_next_batch_to_run` +
`prepare_for_decode` + Higgs `_populate_cg_buffers` + 采样参数提取 + Python
collect 循环 + 调度 overhead），multi-stream 一点也碰不到它。**预期 Plan B 能救
的 wall-time ≈ 0。** 这与 benchmark_results.md 的「bs=1 latency-neutral」、PR #572
的「D2H 3→1 也 neutral」、以及 plan_b_prep.md 的 R-INF-2 完全吻合、互相印证。

真正的（且巨大的）优化空间是那 3.8ms 的 **CPU 每步气泡**，但它需要的是另一类工作
（减少每步 Python/调度开销、把更多每步逻辑塞进 CUDA Graph、或采用上游的
FutureMap overlap 调度），**不是 single-stream→multi-stream**。

## Profile 矩阵结果

窗口 = decode 第 10–40 步（bs4_olen64 为 8–32 步），async ON。

| config | conc | olen | window ms | GPU busy % | idle ms | max gap ms | gaps≥0.5ms（=步数） | recoverable by B? |
|--------|------|------|-----------|-----------|---------|-----------|--------------------|-------------------|
| bs1_olen256  | 1  | 256  | 142.9 | **1.45** | 140.9 | 3.60 | 30 | **否** |
| bs4_olen256  | 4  | 256  | 148.2 | **1.39** | 146.1 | 3.78 | 30 | **否** |
| bs16_olen256 | 16 | 256  | 154.0 | **1.31** | 152.0 | 3.89 | 30 | **否** |
| bs32_olen256 | 32 | 256  | 159.7 | **1.28** | 157.6 | 4.00 | 30 | **否** |
| bs4_olen64   | 4  | 64   | 118.2 | **1.38** | 116.5 | 3.77 | 24 | **否** |
| bs4_olen1024 | 4  | 1024 | 147.8 | **1.40** | 145.7 | 3.77 | 30 | **否** |

每步拆解（以 bs4_olen256 为例，~5.15ms/步）：~3.8ms 单段 GPU 空闲（等 CPU）+ ~1.25ms
forward「活跃区」（其中真正 kernel 计算仅 ~50–68µs，其余是 ~59 个小 kernel 之间的
启动延迟）。原始数据 `/tmp/nsys/stall_stats.json` + 每配置 `.nsys-rep`。

## 详细分析

**关键观察：行为对 batch size 和 output len 几乎不变。** bs=1→bs=32 的 busy%
（1.45%→1.28%）和 gap（3.6→4.0ms）几乎一样，kernel 数恒为 ~59/步。原因是
**CUDA Graph padding**：decode forward 总是 replay 一个 padding 到固定 captured bs
的 graph，所以 bs=1 和 bs=32 跑同一个 padded graph、同样的 GPU 时间。这进一步证明
瓶颈不是 GPU 计算量（它对真实 bs 不敏感、且只有几十 µs），而是每步 ~3.8ms 的 CPU
路径。

- **GPU timeline**：每步一个 ~3.8ms 的大空洞 + 一簇 ~59 个 ≤1µs 的小 kernel。单
  stream（全部 Strm 7）。空洞出现在一步的最后一个 kernel 结束、到下一步第一个
  kernel 开始之间——即 launch-first 循环里「resolve(N-1) 收尾 + 下一轮
  get_next_batch_to_run + prepare_for_decode」的 CPU 段。
- **CPU 工作是否藏在 forward 后？** Plan A 设计的那段（host collect / D2H）**是**
  藏住了（`query_hit=100%`，benchmark 已证）。但 forward 太短（~50µs 计算、~1.25ms
  活跃区），根本盖不住 ~3.8ms 的「下一步准备」CPU 工作——这段工作发生在 forward
  *入队之前*，无法被任何流重叠手段藏到当前 forward 背后。
- **Plan B 能救什么？** Plan B 把 `cudaMemcpyAsync`(D2H) 从主流挪到 alt 流，省的是
  「主流上这条 memcpy 的执行不挡下一条主流操作」。在一个 98% 空闲、forward 仅
  ~50µs 的时间线上，这点收益淹没在噪声里。**没有任何配置存在 Plan B 可救的 stall。**

## Plan B 收益预测

**(b) 路径：建议停止 Plan B 投入。** 理由：B 的收益靶点（主流 D2H 入队/执行）在
Higgs decode 上≈0，因为 (1) A 已重叠 collect，(2) 真正吃 wall-time 的是 forward
*之间* 的 CPU 准备，与 stream 拓扑无关。plan_b_prep.md 已说 B「可行且 A 是干净前
置」——技术上成立，但**值不值得做的答案是「不值得」**：它解决的不是这里的瓶颈。

## 副产物（顺手发现，未改动代码）

1. **decode 是彻底的 CPU/dispatch-bound，GPU ~98% 空闲。** 这是 TTS decode 最大的
   单一优化杠杆，方向是「砍每步 CPU 路径」而非任何 GPU 重叠：候选包括把
   `_populate_cg_buffers` / 采样参数提取 / collect 的 Python 循环向量化或下沉进
   graph、把更多每步逻辑纳入 CUDA Graph capture、或接上游 overlap 调度
   （overlap_utils FutureMap，omni 目前 `disable_overlap_schedule=True`）。
2. **forward 活跃区 ~1.25ms 里真实计算仅 ~50µs**——graph replay 内/周边仍有大量
   inter-kernel 启动延迟，说明并非所有每步算子都进了 graph，值得查 capture 覆盖面。

## 诚实标注的局限

1. **nsys 开销**：nsys 逐核 trace 给 CPU 加了开销。非 nsys 的 bs=4 每步 ~4.3ms
   （benchmark 555ms/128≈4.3），nsys 下 ~5.15ms，约 +20%。所以 3.8ms gap 含 ~20%
   nsys 膨胀；但即便打折，GPU 仍 >95% 空闲、瓶颈仍明确在 CPU 侧、非 GPU/非 D2H，
   定性结论稳健。
2. **只覆盖典型 TTS 的中小 KV 段**：窗口取 decode 第 10–40 步（KV≈prompt+几十
   token）。这些 seedtts prompt 在 ~100 步内 EOC，到不了 step 500+ 的大 KV 稳态，
   所以**未刻画长上下文 regime**——超长序列下 attention 可能增加 GPU 计算、相对空
   闲会缩小。但对 TTS 的实际长度（~100–500 token）本 profile 有代表性。
3. olen=64/256/1024 三档因窗口都在早期步，实测差异极小（皆早期小 KV）；olen 主要
   作为「能否跑满窗口」的保证，未触发长序列差异（见局限 2）。
