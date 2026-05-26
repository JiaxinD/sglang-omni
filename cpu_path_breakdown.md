# CPU Path Breakdown — Decode Per-Step Analysis

Higgs TTS 异步 decode 每步 CPU 路径的 finer-grained 分项。`stall_analysis.md`
已证明 decode GPU busy ≈1.3%、每步约 3.8ms 的 CPU 气泡；本报告把那段气泡拆到
单个函数/语句，回答「每段 CPU 工作各花多少、哪些可优化」。**纯 profile，不改生产
代码、不实现优化。**

工具与方法：

- NVTX instrumentation 全部在 `scripts/profile_inject/sitecustomize.py`，由
  `SGLANG_OMNI_PROFILE_FINE=1` gate（profiler-only PYTHONPATH，零生产代码改动）。
  在 launch/resolve 路径上 monkeypatch 加嵌套 NVTX range，并把 `_decode_collect_host`
  替换成逐语句子 range 的等价副本（逻辑一字不差）。
- 驱动脚本 `scripts/profile_cpu_breakdown.py`：复用 `profile_async` 的 launch /
  health / load-drive，单配置 **Higgs bs=4, max_new_tokens=128, greedy(temp=0),
  async ON**，ion9 H200 **GPU 3**（gpu-check：自有 omni 段唯一空闲卡）。
- capture 窗口 warmup=15 / capture=40（decode 第 15–55 步），恰好落在这些 seedtts
  prompt 的自然 EOC（~67 步）之内，使 `cudaProfilerStop` 干净触发、nsys 完整落盘
  CUDA kernel/memcpy/runtime 活动。用 `nsys export --type sqlite` 全量导出后，按
  NVTX range 聚合 CPU 时长（`nvtx_pushpop`）并把 CUDA 活动投影到每个 range 上做交叉
  验证（`scripts/profile_nvtx_correlate.py`）。
- step period = 连续 `decode_launch` 起点的中位间隔。原始数据：
  `/tmp/nsys_cpu/cpu_bs4_olen128_probe.nsys-rep`（+ `.full.sqlite`）、
  `cpu_bs4_olen128_probe_nvtx.json`；无探针对照 run 见 `cpu_bs4_olen128.*`。

---

## TL;DR

Higgs decode bs=4 每步 **~5.05ms**（nsys 下；非 nsys ~4.2ms，见局限 1），GPU 实际
只忙 ~40µs，其余全是 CPU 串行气泡。

**这段气泡的 68% 来自一行代码**：`_decode_collect_host` 末尾的
`result.next_token_ids = torch.tensor(cb0_per_row, ..., device=cuda)`
（`model_runner.py:265`）。它每步触发一次 **阻塞式 H2D（`cudaMemcpyAsync` +
`cudaStreamSynchronize`），中位 3.43ms**，而**此 sync 期间 GPU 99.7% 空闲**（窗口内
kernel 仅 11µs，且上一步最后一个 kernel 在 sync 开始前 400µs 就结束了）。也就是说
`stall_analysis.md` 看到的「每步 3.8ms GPU 空洞」几乎就是这一个 sync。

这行被 **同步路径与异步 resolve 路径共用**（`_collect_step_outputs_cg` 和
`post_decode_resolve` 都调 `_decode_collect_host`），所以 launch/resolve 的 overlap
根本没碰到它——这正解释了 benchmark 里「async bs=1 latency-neutral」。

- **可优化部分上限**：把这行的每步阻塞 H2D 去掉，每步 CPU 上限可降到 ~1.6ms，
  对应 decode throughput 上限提升 **~3×**（**上限**，非承诺值——见下方诚实性说明，
  3.43ms 阻塞在「GPU 空闲」时仍发生，机制未被 cuda/nvtx-only trace 完全解释，落地前
  必须用 A/B 验证实际可回收多少）。
- 其余可优化项都是「锦上添花」量级（合计 ~0.3–0.5ms）：3 个冗余采样参数 D2H
  （~79µs，最易）、`_populate_cg_buffers` 的 Python 行循环 + H2D（~400µs，中等）。
- `forward_replay`（434µs）、`build_forward_batch`（161µs）、`get_next_batch`
  （130µs）都在上游 sglang，难度高、收益有限。

---

## 矩阵结果

bs=4 稳态段（decode 第 15–55 步，n≈40，greedy，async ON，nsys）。时间为**中位数
μs**；`% step` 以 step period 5055µs 为分母。`stall?` = 该 range 执行时 GPU 是否空闲
（CPU 工作未被 GPU 覆盖）。嵌套关系见缩进；父 range 含子 range（避免重复计入，
「可省」只针对叶子/自有时间论证）。

| NVTX range | mean μs | median μs | % step | stall? | 优化难度 | 估算可省 μs |
|---|---|---|---|---|---|---|
| step.get_next_batch（prepare_for_decode，上游） | 131.9 | 129.9 | 2.6% | 是 | 难（上游） | 0–30 |
| **decode_launch（小计）** | 1337.8 | 1301.8 | 25.8% | 是 | — | — |
| ├ launch.build_forward_batch | 177.6 | 161.3 | 3.2% | 是 | 难（上游 get_model_worker_batch） | 0–50 |
| ├ launch.populate_cg_buffers | 489.2 | 475.3 | 9.4% | 是 | 中 | 150–300 |
| │ &nbsp;&nbsp;└ launch.extract_sampling_params | 80.0 | 78.5 | 1.6% | 否（快） | **易**（静态、缓存一次） | ~70 |
| ├ launch.forward | 465.1 | 454.7 | 9.0% | 部分 | 难（上游） | 0–150 |
| │ &nbsp;&nbsp;└ launch.forward_replay（CUDA graph replay 入队） | 445.2 | 434.0 | 8.6% | 部分 | 难（上游 replay_prepare） | 0–150 |
| ├ launch.sample | — | — | 0% | — | — | 0（Higgs 采样在 graph 内，本函数 0 次调用） |
| └ launch.post_decode_launch | 168.2 | 164.9 | 3.3% | 是 | 难 | ~0 |
| &nbsp;&nbsp;&nbsp;&nbsp;└ launch.pack_gpu（GPU scatter+pack + 非阻塞 D2H 入队） | 104.6 | 102.9 | 2.0% | 是 | 难（已是 GPU op） | ~0 |
| **decode_resolve（小计）** | 3525.6 | 3528.1 | 69.8% | 是 | — | — |
| └ resolve.collect_host | 3474.4 | 3477.3 | 68.8% | 是 | — | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ collect.read_flags（`.bool().tolist()` ×2） | 17.0 | 17.0 | 0.3% | 是 | 难（已极小） | ~0 |
| &nbsp;&nbsp;&nbsp;&nbsp;├ collect.pyloop（per-request 收集循环） | 25.2 | 24.9 | 0.5% | 是 | 中（已极小，bs=4） | ~0 |
| &nbsp;&nbsp;&nbsp;&nbsp;└ **collect.next_ids_h2d**（`torch.tensor(cb0, device=cuda)`） | **3424.7** | **3427.5** | **67.8%** | **是（GPU 99.7% 空闲）** | **易–中** | **2800–3400（区间，需 A/B）** |
| **每步总计（period）** | 5094.7 | 5054.5 | 100% | 是（GPU busy ~40µs） | — | — |
| **可优化部分合计** | | | | | | **~3000–3700**（其中 next_ids_h2d 占 ~3.4ms） |

> 一致性校验：`get_next_batch + decode_launch + decode_resolve` 中位合计 = 4960µs，
> 占 period 5055µs 的 **98%**；剩余 ~95µs 是 `recv_requests` / `process_input_requests`
> / event 循环 glue。run1（无探针）独立测得 period 5030µs、collect_host 3451µs，与本
> run 互相印证。

---

## 分项分析

### collect.next_ids_h2d — `torch.tensor(cb0_per_row, device=cuda)`（中位 3427µs，67.8%）★

- **当前实现**（`sglang_omni/models/higgs_tts/model_runner.py:265-269`，`_decode_collect_host`
  尾部）：把 host 上拼好的 `cb0_per_row`（一个长度 = bs 的 **Python int 列表**）用
  `torch.tensor(..., device=cuda)` 直接建成 GPU 张量，写入 `result.next_token_ids`。
- **为什么慢**：`torch.tensor(pylist, device='cuda')` = 先在 **pageable** host 内存建
  CPU 张量，再做一次 H2D。trace 显示它每步恰好对应 **1 个 `cudaMemcpyAsync` + 1 个
  `cudaStreamSynchronize`**；该 sync 中位 **3.43ms**（39/40 步 >1ms，max 3415µs）。
  关键证据：sync 窗口内 GPU kernel 仅 11µs（0.33%），上一步最后一个 kernel 在 sync
  起点前 **400µs** 就结束、下一步第一个 kernel 在 sync 终点后 177µs 才开始——**整个
  3.4ms 期间 GPU 处于空闲**。即这条 sync 不是在等有意义的 GPU 计算，它本身就是
  `stall_analysis.md` 里那个「每步 3.8ms GPU 空洞」。
- **同步/异步都中招**：`_decode_collect_host` 被同步路径 `_collect_step_outputs_cg`
  （`model_runner.py:204`）和异步 `post_decode_resolve`（`model_runner.py:94`）共用。
  异步 lookahead 把 staging 的 D2H 改成了非阻塞、并 overlap 了 collect，但**没动这条
  `next_ids_h2d` 的阻塞 sync**——所以它仍把每步串行化，正好对上 benchmark 的
  「async bs=1 latency-neutral」。
- **优化方向**（设计阶段定，不在本任务实现）：消除「每步从 Python list 建 device 张量
  的阻塞 H2D」。三条互斥/可叠加思路，按从易到难：
  1. **在 CPU 上构造** `next_token_ids`（它在 resolve 路径只喂给 `output_processor`
     做输出上报，且 `_finalize(set_output_ids=False)` 不会把它发布到 `schedule_batch`）
     ——需先确认下游 `output_processor.process` 是否真的需要它在 GPU 上。
  2. 若必须在 GPU 上：改成**预分配 pinned buffer + `copy_(non_blocking=True)`**，
     并把 sync 推迟/省去（值在 `post_decode_launch` 已从 GPU state 设过一份）。
  3. 干脆**不在 collect 里重建**：`post_decode_launch` 已用 `_cg_codes_BN[:,0]` 设好
     `result.next_token_ids`，collect 端只需为「输出上报」提供 host 侧 cb0（已经有
     `cb0_per_row` 这个 list），未必需要再回 GPU。
- **预期可省**：上限 ≈ 该 sync 的 3.4ms（GPU 空闲，没有真实 GPU 等待要「搬走」）。
  但给 **区间 2.8–3.4ms** 并标注不确定（见诚实性 #2）：3.4ms 阻塞在 GPU 空闲时仍发生，
  其精确成因未被 cuda/nvtx-only trace 解释清楚，可能含无法在此 trace 看到的 host/驱动
  侧依赖；实际可回收量**必须用 A/B 实测**（把这行改成 CPU 构造或 pinned 非阻塞，量
  step period 是否从 ~5ms 掉到 ~1.6ms）。

### launch.populate_cg_buffers（中位 475µs，9.4%；自有 ~396µs）

- **当前实现**（`model_runner.py:96-156`）：`acquire_row` 的 **Python for 循环**逐请求
  取 row → `torch.tensor(rows_py, device=cuda)`（H2D）；再 `extract_sampling_params`
  + 把 temps/top_p/top_k 各建一个 `torch.tensor(..., device=cuda)`（**3 次 H2D**）；
  最后 `pool[rows_t]` gather 写 `_cg_active_*`（GPU op）。
- **为什么慢**：trace 显示该 range 含 ~19 `cudaMemcpyAsync` + ~10 `cudaStreamSynchronize`
  + ~8 `cudaLaunchKernel`，但 GPU busy 仅 ~11µs——开销主要是 **Python 循环 + 多次小
  H2D 的入队/同步 CPU 开销**，不是 GPU 计算。
- **优化方向**：把 rows/temps/top_p/top_k 的 4 次独立小 H2D **合并成一次**（先在 host
  拼好再一次拷贝）；`acquire_row` 循环可向量化或缓存 rid→row 映射。属「减少每步小
  H2D 数量 + 去 Python 循环」，中等难度（需小心与 CUDA Graph 的 `_cg_*` 缓冲写入时机
  一致，见 open questions）。
- **预期可省**：**150–300µs**（大不确定：合并 H2D 省多少取决于每次 sync 的固定开销，
  给区间）。

### launch.extract_sampling_params（中位 78.5µs，1.6%）— 最易

- **当前实现**（`model_runner.py:159-183` → `model.py:48` `_flat_sampling_attr`）：每步对
  `sampling_info` 的 temperatures / top_ps / top_ks 各做一次 `.detach().cpu().tolist()`
  ——**3 次 D2H**（trace 实测该 range 含 3 `cudaMemcpyAsync` + 3 `cudaStreamSynchronize`，
  但都很快、GPU 空闲，合计仅 ~79µs）。
- **为什么慢/冗余**：这些采样参数对一个请求是**静态的**（整条 decode 不变），却每步
  重读。`investigation.md` §6 已点名「根本不该每步做，host 侧缓存一次即可消除」。
- **优化方向**：每请求**首步缓存** temps/top_p/top_k 到 host，后续步直接复用。**最易**，
  不依赖任何 async 机制，可作前置小 PR。
- **预期可省**：~70µs（去掉 3 次 D2H 的 CPU 同步开销；占比小但零风险）。

### launch.forward / launch.forward_replay（中位 455 / 434µs，9.0% / 8.6%）

- **当前实现**：`ModelWorker.forward_batch_generation` → `model_runner.forward` →
  `CudaGraphRunner.replay`（上游）。`forward_replay` 含 `replay_prepare`（把 input_ids /
  positions H2D 进 graph 缓冲、设 attn backend）+ graph launch。
- **为什么慢**：trace 显示该 range GPU busy 仅 ~11µs，434µs 几乎全是 **CPU 侧 graph
  replay 入队开销**（`replay_prepare` 的拷贝 + 簿记）。这与 `stall_analysis.md` 的
  「forward 活跃区 ~1.25ms 内真实计算仅 ~50µs」同源。
- **优化方向**：属上游 sglang `cuda_graph_runner`，omni 侧不易动；可查 capture 覆盖面
  （把更多每步算子纳入 graph 以减少 replay_prepare 外的零散 launch）。**难**。
- **预期可省**：0–150µs（高度不确定，多为上游工作）。

### launch.build_forward_batch（中位 161µs，3.2%）/ step.get_next_batch（130µs，2.6%）

- 分别是 `get_model_worker_batch` + `ForwardBatch.init_new`（`base.py:207`）和上游
  `Scheduler.get_next_batch_to_run`（含 `prepare_for_decode` 的 input_ids/positions
  构建 + H2D）。都是上游通用调度/构建路径，**难**动，单项收益小（各 <50µs 可省）。
- 注：`step.get_next_batch` 的 NVTX 计数被 event-loop 空转调用污染（47k 次、中位
  2.6µs）；上表的 130µs 是**关联到真实 decode 步**（紧贴 `decode_launch` 之前那次）的
  中位值，已剔除空转。

### launch.pack_gpu（中位 103µs，2.0%）/ collect.read_flags+pyloop（42µs）

- `pack_gpu`（`model_runner.py:206`）是 scatter + pack 的 GPU op + 非阻塞 D2H 入队，
  collect 的 read_flags / pyloop 是 host 上对已落地 pinned snapshot 的读取与 4 行循环。
  都已经很小（bs=4），**无明显优化空间**。

---

## 候选优化清单（按 ROI 排序）

1. **消除 `next_ids_h2d` 的每步阻塞 H2D**（`_decode_collect_host` 尾，`model_runner.py:265`）
   - 收益：上限 ~3.4ms/步，throughput 上限 ~3×；占整段气泡 68%，是**唯一数量级杠杆**。
   - 难度：易–中（改一行的构造方式；难点在确认下游是否需要 GPU 张量 + 与 lookahead
     的 `result` 消费时机一致）。
   - 风险：`next_token_ids` 同时供同步路径与异步 resolve、且喂 `output_processor`；改 device
     需回归 bs=1/bs>1 的输出正确性。**落地前先做 A/B**（gpu-perf-ab：变体开关 + 量
     step period + profiler trace 背书），确认实际可回收量再定方案。

2. **缓存静态采样参数**（`extract_sampling_params`，`model.py:48`）
   - 收益：~70µs/步（小），但**零风险、最易、与 async 解耦**，适合作前置 atomic 小 PR。
   - 难度：易。风险：低（仅需在请求生命周期内缓存，注意 chunked / 中途增删请求时失效）。

3. **合并 `_populate_cg_buffers` 的 4 次小 H2D + 去 Python 循环**（`model_runner.py:96`）
   - 收益：150–300µs/步（区间）。难度：中。
   - 风险：中——写 `_cg_*` 缓冲，需确认与 CUDA Graph replay 的读时机不产生竞争
     （`investigation.md` §5.1 列为待验证项）。

4.（暂不投入）forward_replay / build_forward_batch / get_next_batch：均在上游 sglang，
   单项收益 <150µs 且改动面大/不可控，性价比低。

---

## Open questions

- **`next_ids_h2d` 的 3.4ms sync 为何在 GPU 空闲时仍阻塞？** cuda/nvtx-only trace
  （`--sample=none --cpuctxsw=none`）看不到 CPU 调度/驱动内部，无法判定这 3.4ms 是
  纯 host 开销（→ 可全回收）、还是某个本 trace 不可见的 host/驱动依赖（→ 部分回收）。
  **需一次带 `--cpuctxsw` / OS-runtime 采样的 trace，或直接 A/B 实测**才能定量。这是
  「可省 μs」给区间而非定值的根本原因。
- **`next_token_ids` 是否必须在 GPU 上？** 取决于上游 `output_processor.process` 与
  lookahead 下 `result` 的消费方式——决定可否直接 CPU 构造（最省）。
- **哪些优化会与 CUDA Graph capture 冲突？** `_populate_cg_buffers` 合并 H2D / 改写
  `_cg_*` 缓冲必须与 graph replay 的读时机严格串行（单 stream 顺序入队下初判安全，
  错位一步后需 `compute-sanitizer` 验证无 in-place race）。
- **是否上游 PR？** `next_ids_h2d` 与 `extract_sampling_params` 都在 omni 仓内
  （`higgs_tts/model_runner.py`、`model.py`），可本地改；`forward_replay` /
  `build_forward_batch` / `get_next_batch` 在上游 sglang，需上游协作。
- **不显眼的兼容性风险**：`_decode_collect_host` 被同步与异步两条路径共用，任何改动
  须同时回归两条路径 + bs=1/bs>1 + EOC/length-finish 的 skip 逻辑。

---

## 诚实性标注的局限

1. **nsys 开销**：逐核 trace 给 CPU 加 ~20%（`stall_analysis.md` 局限 1：非 nsys bs=4
   每步 ~4.3ms，nsys 下 ~5.05ms）。本报告所有**绝对 μs 含 ~20% nsys 膨胀**；**占比
   （% step）比绝对值可靠**。映射到真实 throughput 时绝对可省量应按 ~0.8 折算。
2. **`next_ids_h2d` 可省量是区间不是定值**：见上方 open question——3.4ms 在 GPU 空闲时
   仍阻塞，机制未被本 trace 完全解释，故给 2.8–3.4ms 区间并明确「上限、需 A/B 确认」，
   不夸大为承诺值。
3. **窗口只覆盖中小 KV 段**：decode 第 15–55 步（KV ≈ prompt + 几十 token），未刻画
   长上下文 regime。但对 TTS 实际长度（~100–500 token）有代表性，且本报告的瓶颈
   （`next_ids_h2d`，与 KV 长度无关的固定每步 sync）在任意 KV 段都成立。
4. **read_flags / pyloop 在更大 batch 下会增长**：bs=4 时 collect 的 Python 循环仅
   25µs，bs=32 时会线性放大（但相对 `next_ids_h2d` 的 3.4ms 仍小）；本 profile 未跑
   bs=16/32 对照（second-priority，时间所限），故大 batch 下 collect 循环占比未实测。
5. **`step.get_next_batch` 的真实 decode-步成本是关联估计**：NVTX 计数被 47k 次空转
   调用污染，130µs 取自「紧邻 decode_launch 前」的关联匹配，非纯净独立测量。
