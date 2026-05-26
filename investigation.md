# Async Decode Scheduling — Phase 1 调研报告

> 目标：在 Higgs TTS（及后续其它 omni AR 模型）的 decode 主循环里，用 **方案 A：One-Step Lookahead 异步调度**（单 stream + CUDA Event `query()`）把 forward 后面那段 ~1.1ms 的 CPU 串行气泡藏到 GPU forward 背后。本报告只做调研，不写生产代码。
>
> 所有 `file:line` 引用：`sglang_omni/...` 为本仓库（commit 基线 `feat/higgs-cg-batch-d2h`）；`/sgl-workspace/sglang/python/sglang/...` 为本机安装的上游 sglang（`import sglang` 解析到此）。

---

## 0. 前提校正（与任务描述的出入）

| 任务描述假设 | 实际情况 |
|---|---|
| 有 `./sglang-omni/`、`./sglang-omni-pr572/`、`./sglang/` 三个子目录 | **都不存在**。`/data/sglang-omni` 本身就是 sglang-omni 仓库；当前分支 `feat/higgs-cg-batch-d2h` 上的 commit `9f235ff` 就是「PR #572」的 D2H 工作。无需 clone。 |
| 「PR #572 砍掉 call2/call3 两个 D2H，省 45μs」 | 实际是 commit `9f235ff`「**把 3 个 per-step D2H 合并成 1 个**」（`_collect_step_outputs_cg`）。A/B 结论是 **latency-neutral**——瓶颈不是 sync 数量，而是 per-step CPU 同步本身。方向一致，叙述略有简化。 |
| call1 / call2 / call3 是三个 kernel | 是 `_collect_step_outputs_cg` 里原来的三个 D2H 拷贝（见 §1.3） |

`git diff main` 不能用来隔离 D2H 改动（当前分支相对 main 有 234 文件 / +27917 行的漂移）；要看具体 commit `9f235ff` / `be93a97`。

---

## 1. 当前 decode 单步循环调用图

Higgs TTS 的 AR decode 跑在 `OmniScheduler` 上，**overlap 被显式禁用**（`sglang_omni/models/higgs_tts/stages.py:261` 设 `server_args.disable_overlap_schedule = True`），因此走的是 `_event_loop_normal`（全串行）。

```
OmniScheduler._event_loop_normal()                         [omni_scheduler.py:804]   ← Higgs 跑这条（overlap 关）
 └─ while running:
      recv_requests / process_input_requests               (CPU)
      batch = get_next_batch_to_run()                       (CPU 调度)            ┐
      result = run_batch(batch)  ────────────────────────  [→ _run_batch:584]    │
        └─ self._model_runner.execute(sched_output)         [omni_scheduler.py:607 → base.py:33]
             ├─ get_model_worker_batch / ForwardBatch.init_new   [base.py:47,64]   (CPU)        │ 这一整段
             ├─ prepare_decode → _populate_cg_buffers            [base.py:75 → model_runner.py:53]
             │     · acquire_row + torch.tensor(rows) H2D        [model_runner.py:70-74]         │ ≈ 1.1ms
             │     · _extract_decode_sampling_params             [model_runner.py:76]            │ CPU 气泡
             │         → _flat_sampling_attr ×3  **3× D2H**      [model.py:57]                   │ （与 GPU
             │     · pool[rows] → _cg_active_*  (GPU→GPU)        [model_runner.py:96-99]         │  forward
             ├─ forward_batch_generation  ── GPU forward (CG replay) + on-GPU 采样 ── 3.72ms     │  完全串行）
             │                                                   [base.py:82]                    │
             ├─ post_decode → _collect_step_outputs_cg           [base.py:114 → model_runner.py:128]
             │     · _cg_active_* → pool  (GPU→GPU scatter)      [model_runner.py:146-149]       │
             │     · staging[:n_real].cpu()  ★唯一 D2H 同步★     [model_runner.py:157]  ← 阻塞等 GPU
             │     · per-request Python loop（append/stop/next） [model_runner.py:161-181]       │
             └─ output_processor.process + per-req 记账          [base.py:139-148]   (CPU)        │
      process_batch_result(batch, result)                   (CPU, 上游)                          ┘
```

**关键结构**：`ModelRunner.execute()`（`base.py:33`）是**单体同步**的——prepare + forward + D2H-collect + output 全在一次阻塞调用里完成。`staging[:n_real].cpu()`（`model_runner.py:157`）是 CPU 等 GPU 的那个点；它之前的 prepare/forward-launch 是异步入队的，它之后的 collect 循环 + output 处理是 GPU 已经空闲时纯跑 CPU。**GPU forward（3.72ms）和 CPU 工作（~1.1ms）完全串行**，这就是气泡。

### 1.1 自回归反馈留在 GPU 上（lookahead 可行性的基础）

token 的 AR 反馈（上一步 codes → 下一步 embed）**不经过 host**：采样状态机 `decode_codebooks_batch_cg`（`model.py:321`）在 GPU 上写 `_cg_active_last_codes`，下一步的 `_decode_step_embeds_cg`（`model.py:430-446`）又在 GPU 上读它。因此 GPU 跑下一步 forward **不需要等 host 消费这一步的 codes**。

host 的 D2H（`model_runner.py:157`）只服务三件事，**都不在「启动下一步 GPU forward」的关键路径上**：
1. 把 codes 追加到 host 侧 `data.output_codes`（`model_runner.py:172`）；
2. 读 `was_done` / `generation_done` 做 stop 判定（`model_runner.py:168,173`）；
3. 拼 `result.next_token_ids` 给上游记账（`model_runner.py:177`）。

唯一真正的跨步依赖是 **stop 判定**：调度器要知道哪些请求在第 N 步结束了，才能决定第 N+1 步的 batch 组成。**这正是 one-step lookahead 要延后一步消费的东西**（代价：序列末尾每条请求最多多算 1 步，丢弃溢出）。

### 1.2 call1 / call2 / call3 的对应关系（#572 / #564 上下文）

`#564` 描述的三个 D2H（合并前，`_collect_step_outputs_cg`）：

| 原 call | 张量 | 合并后位置（staging buffer） |
|---|---|---|
| call1 | `_cg_was_done[:n_real]` | `staging[:n_real, num_codebooks]`（`model_runner.py:155`） |
| call2 | `_cg_codes_BN[:n_real]` | `staging[:n_real, :num_codebooks]`（`model_runner.py:154`） |
| call3 | `_cg_active_generation_done[:n_real]` | `staging[:n_real, num_codebooks+1]`（`model_runner.py:156`） |

合并手法：先在 GPU 上把三个张量打包进 `_cg_collect_staging`，再用一次 `.cpu()` 拷回 host，最后在 host 上切片（`model_runner.py:152-160`）。

---

## 2. Host 同步点清单

### 2.1 Decode CG 热路径（本工作的目标循环）

| 阶段 | file:line | 操作 | 触发 GPU 同步? | 说明 |
|---|---|---|---|---|
| prepare | `model.py:57`（经 `model_runner.py:76` 调 3 次：temperatures/top_ps/top_ks） | `.detach().cpu().flatten().tolist()` | **是（D2H ×3）** | **每步重读静态采样参数，冗余**；独立的 caching 机会（见 §6 风险/机会） |
| prepare | `model_runner.py:72-74` | `torch.tensor(rows).to(device)` | H2D（非阻塞） | row indices 上传 |
| forward | `base.py:82` | CG replay + on-GPU 采样 | 否（异步入队） | ~3.72ms |
| collect | `model_runner.py:157` | `staging[:n_real].cpu()` | **是（唯一关键 D2H）** | 阻塞等 GPU forward 完成；#572 后只剩这一个 |
| collect | `model_runner.py:159,160` | `.bool().tolist()` | 否（对象是 `combined_cpu`，已在 host） | host 侧切片 |
| collect | `model_runner.py:175` | `int(codes_N[0].item())` | **否**（`codes_N` 来自 `combined_cpu`，已在 host） | agent 初判误标为同步点，实测是 CPU 侧 |

**净结论**：decode CG 热路径上真正的 GPU 同步 = prepare 阶段最多 3 个（静态参数 D2H）+ collect 阶段 1 个（`model_runner.py:157`）。其余 `.tolist()` / `.item()` 都作用在已落 host 的张量上。~1.1ms 气泡 = 这些 D2H 的 CPU 侧 `cudaStreamSynchronize` 开销 + collect 的 per-request Python 循环 + `output_processor` + 上游 `process_batch_result` + 下一步 `get_next_batch_to_run` 调度。

### 2.2 Prefill / 非-CG 路径（不在 decode 气泡内，列此备查）

| file:line | 操作 | 场景 |
|---|---|---|
| `model_runner.py:207` | `full_mask.sum().item()` | prefill embed 构建 |
| `model_runner.py:244,245,247` | `.cpu().clone()` / `.item()` | 非-CG decode collect（`_collect_step_outputs`） |
| `sampler.py:92,93,98` | `.item()`（`view_row()`） | 主要 prefill |
| `sampler.py:197` | `codes_N[0].item()` | `step()` EOC 检查 |
| `model.py:306` | `was_done.cpu().tolist()` | 非-CG decode 路径 |

---

## 3. Model Runner 前向 + 采样路径

- sglang-omni **不自己实现核心前向**；`execute()`（`base.py:33`）在 `base.py:82` 调 `tp_worker.forward_batch_generation(forward_batch)`，最终落到上游 `sglang/srt/model_executor/model_runner.py` 的 `forward()` / `forward_decode()`，CUDA Graph 模式下走 `cuda_graph_runner.replay()`。
- Higgs 的采样**不在上游 sampler**：`prepare_decode` 返回 `None`（`model_runner.py:47`）→ 走标准 forward；采样状态机在模型内部 GPU 上跑（`decode_codebooks_batch_cg`，`model.py:321-373` → `batched_step_direct`），结果写进 `_cg_codes_BN` / `_cg_active_*`，`next_token_ids` 在 `post_decode` 里由 host 侧 cb0 拼出（`model_runner.py:177`）。
- 采样器无显式 stream 操作，用 current stream。

---

## 4. 现有 overlap 支持的覆盖范围

### 4.1 OmniScheduler 自带的 `_event_loop_overlap`（骨架已有，但禁用）

`omni_scheduler.py:834-877`。`start()`（`:732-737`）按 `enable_overlap` 二选一进 `_event_loop_overlap` 或 `_event_loop_normal`。

它 overlap 的是：把 **第 N-1 步的 `process_batch_result`** 延后到第 N 步 `run_batch` **之后**再处理（`result_queue` + `pop_and_process`，`:837-867`）。`f61653d` 那个改动只是加 `time.sleep(0.001)` 在空闲点让 GIL（避免同进程的 audio encoder 被 AR loop 饿死）。

**为什么这个骨架当前救不了气泡**：omni 的 D2H 同步发生在 `execute()` / `run_batch` **内部**（`model_runner.py:157`），而骨架延后的是 `process_batch_result`。也就是说 `run_batch(N)` 本身就阻塞等 GPU N 完成（`.cpu()`），骨架并没有把「N 步的 D2H/collect」从关键路径挪走。上游能 overlap 是因为它把 launch 和 resolve 拆开了（见 §4.2），omni 的 `execute()` 没拆。

**经验验证（来自同事，2026-05-26）**：EARFQUAKE 已经试过启用 overlap mode（配 `time.sleep(0.001)` 的 GIL-yield 修复，即 commit `f61653d`，现已在 `:846,870`），**profile 下来提升不明显**。这从经验上印证了上面的静态推断——**单纯启用现有骨架救不了气泡**，因为 D2H/resolve 仍在 `execute()` 内部、没被挪出关键路径。该 overlap mode 今天能跑（GIL bug 已修），问题是不提速。

**生产模型全部禁用 overlap**：`higgs_tts/stages.py:261`、`qwen3_omni/talker_scheduler.py:28`、`fishaudio_s2_pro/stages.py:203`、`qwen3_tts/stages.py:170`、`voxtral_tts/pipeline/stages.py:129`，均设 `disable_overlap_schedule=True`；`bootstrap.py:57` 据此导出 `enable_overlap`。

### 4.2 上游 sglang 的 FutureMap overlap（完整、默认开启，但 omni 没接）

- `/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py`：`FutureMap`（:33）+ `alloc_future_indices`（:109）+ `resolve_future`（:118）+ `store_to_map`（:142）。机制：第 i 步 `run_batch` 只**启动** GPU 并把输出 token 当作「future 槽位」（负数索引）；CPU 不等 token 实值就能 prepare 第 i+1 步，forward 时再由 kernel 把负索引解析成实 token。`.cpu()` / resolve 挪到 `process_batch_result`，于是和下一步 launch overlap。
- 上游 `scheduler.py` 的 `event_loop_overlap` 结构与 omni 骨架同形，但**积极使用 FutureMap**；默认 `disable_overlap_schedule=False`。
- **omni 完全没有 FutureMap**（仓内搜 `future_token_ids_map` / `resolve_future` / `TpModelWorkerClient` 均无命中）。

### 4.3 「port 还是从零」的结论

既不是纯「从零」也不是纯「port」：
- 骨架（事件循环级 overlap）omni **已有**，但与 omni 的同步 `execute()` 不对齐，当前无效；
- 上游有成熟的 launch/resolve 拆分（FutureMap），但那是**重机制**，且和你为方案 A 划的 CUDA-Event 边界（单 stream、只用 `event.query()`、不碰 future kernel）不是一回事；
- 本质工作 = **在 omni 的 decode 路径把 `execute()` 的 launch 与 resolve（D2H/collect/stop）拆开，延后一步消费**。这是 omni 特有的改造，FutureMap 不能直接搬。

### 4.4 机制的通用性（general，非 higgs-specific）

这次优化是 scheduler/runner 层的 general 改动（按 EARFQUAKE 的预期，重点验证 higgs + qwen omni）。已核实 **5 个 AR 模型全部共用同一套基座**，因此 lookahead 放在 `ModelRunner.execute()`（`base.py:33`）+ `OmniScheduler` 层就天然覆盖全部：

| 模型 | runner（subclass ModelRunner） | scheduler | overlap |
|---|---|---|---|
| Higgs TTS | `higgs_tts/model_runner.py:28` | OmniScheduler | 禁用 |
| Qwen3-Omni talker | `qwen3_omni/talker_model_runner.py:15` | `QwenTalkerScheduler(OmniScheduler)` | 禁用 |
| Qwen3 TTS | `qwen3_tts/model_runner.py:15` | OmniScheduler | 禁用 |
| Voxtral TTS | `voxtral_tts/model_runner.py:15` | OmniScheduler | 禁用 |
| FishAudio S2-Pro | `fishaudio_s2_pro/model_runner.py:69` | OmniScheduler | 禁用 |

**但气泡形状各模型不同，收益不能照搬**：
- **Higgs**：`post_decode` 有显式的每步 `.cpu()` D2H（`model_runner.py:157`）——profiling 来源、最清晰的靶子。
- **Qwen3-Omni talker**：`post_decode`（`talker_model_runner.py:91-133`）把结果留 GPU 上 `.clone()`、直接把 **GPU tensor** 经 outbox 丢给下游 code2wav，**post_decode 里没有 Higgs 那种阻塞 D2H**。它的每步气泡是 Python emit 循环 + clone，不是一个阻塞同步。
- → 设计放在 general 层，但 **Qwen-Omni 的收益需单独 profile**；先用 Higgs（显式 D2H）证明机制，再量化 Qwen。

---

## 5. Async Lookahead 插入点（两条路中立并列）

**共同点（无论哪条路）**：核心改造都是把 `execute()`（`base.py:33`）的单体流程拆成两半——
- **launch 半**：prepare（用上一轮已 resolve 的 stop 信息）+ forward 入队 + 在采样完成处 `event.record()`；
- **resolve 半**：`event.query()` 非阻塞确认 → `staging.cpu()` D2H → collect 循环 → stop 判定 → output。

时间线对比：

```
现状（串行）：
  step N:   [prepare N][launch fwd N]……GPU fwd N……[.cpu()阻塞][collect N][output N]
                                          (CPU 全程空转等 GPU，GPU 全程等不到下一步)

Lookahead（一步错位）：
  step N:   [resolve N-1: collect/stop/output] [prepare N] [launch fwd N][record evt N]
                          └─ 这段 CPU 工作和 ……GPU fwd N…… 重叠 ─┘
  代价：序列末尾每条请求最多多算 1 个被丢弃的 step（stop 延后一步生效）
```

| 维度 | 路 A：复用现有 `_event_loop_overlap` 骨架 | 路 B：独立新建 lookahead |
|---|---|---|
| 落点 | 启用 `enable_overlap` 分支；把 `execute()` 的 resolve 半拆进 `process_batch_result`（即骨架已延后的那段），prepare/launch 留在 `run_batch` | 不碰骨架（保持禁用）；在 `ModelRunner.execute` 内部维护 `pending_step` 状态，自己做 N→N+1 错位 |
| 改动面 | `omni_scheduler.py`（启用分支 + 拆 result 处理）+ `base.py`/`model_runner.py`（D2H 移出 run_batch）+ 各 stages 的 flag | 主要集中在 `base.py` + `model_runner.py`（execute 状态机）；scheduler 改动小 |
| 与 CUDA-Event 边界契合 | 中：骨架是 Python 队列级 overlap，CUDA Event 作为「N 步采样是否完成」的非阻塞门 | 高：event.record/query 是 execute 状态机的天然组成，单 stream 不变 |
| 与上游语义纠缠 | 较高：复用上游 `process_batch_result` 路径，要确认不依赖 FutureMap 假设 | 低：自包含，debug 半径小（与你「Debug 半径小」的偏好一致） |
| 复用度 | 高（少写循环骨架） | 中（循环骨架要自己搭，但都在 execute 层） |
| 风险 | 骨架原本为「launch/resolve 已拆」的上游设计，omni 没拆，强行复用可能需要同等量的 execute 改造，复用收益打折 | execute 状态机要自己处理 warmup / abort / 新请求中途加入 |

> **方向决策留给 Phase 1→2 边界**（按你的选择中立并列）。我的倾向（仅供参考，非定论）：两条路的「硬骨头」都是同一件事——拆 `execute()`。路 A 多出的「复用骨架」收益，会被「骨架假设与 omni 不匹配」抵消一部分；路 B 的自包含 + 小 debug 半径更贴合方案 A「最小可行、改动聚焦」的初衷。Phase 2 设计时建议先按路 B 画状态机，再回头看能否顺手复用骨架。
>
> **经验数据更新（§4.1）**：EARFQUAKE 实测「启用骨架 as-is → 不提速」，进一步说明路 A 的「复用骨架」本身不带来收益、价值全在拆 `execute()`。这把天平更明确地推向**路 B**（或退一步说：路 A 也必须做同等的 execute() 拆分，复用骨架只是装饰）。

### 5.1 CUDA Graph 共存

方案 A **不改 CUDA Graph capture**。graph 在 GPU 上 replay（`cuda_graph_runner.replay`，上游 `:842`），prepare/collect 在 CPU 上跑，互不干扰。Lookahead 只改变 CPU 端「何时消费 GPU 结果」，replay 的输入缓冲（`_cg_*`）写入时机不变。**需在 Phase 2/3 验证**：错位一步后，`_populate_cg_buffers` 写 `_cg_*` 与上一步 replay 读 `_cg_*` 不能产生 in-place 竞争（当前单 stream 顺序入队天然串行，错位不改变入队顺序，初判安全，待实测确认）。

---

## 6. Plan B（Dual-Stream + GPU-side Future）可行性静态分析（不实现）

### a) CUDA Graph 兼容性
- 上游 `cuda_graph_runner.py` 的 capture（`:472` / `_capture_one_stream:477` / `capture_one_batch_size:552`）和 replay（`:842`）以 **单 stream** 为模型：`torch.cuda.CUDAGraph(stream=stream)` 本身就是单流；capture 期间所有 kernel 必须记录在该 stream 上，跨 stream 的操作不会被记录（数据依赖会丢）。pdmux 多流（`:511`/`:859` 用 `f"{stream_idx}_{bs}"` 维护多个 graph 实例）只是「每个 decode stream 一份单流 graph」，**不是单 graph 内多流**。
- 要支持单 graph 内 fork/join 多流，需在 `capture_one_batch_size` 的 `run_once` 内部显式 `alt_stream.wait_stream(cur)` / `cur.wait_stream(alt_stream)`，并适配 replay。**粗估 20-40 LOC**（仅 runner 层），但这正是方案 B 要用、而方案 A 明确禁用的 `stream.wait_event` / 多 stream 协同。
- **已有可参考的替代**：`/sgl-workspace/sglang/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`（分段 graph，每段可在不同 stream），若真要做 B 应优先评估它而非改标准 runner。

### b) Memory allocator 影响
- KV cache 用自定义 page-aligned pool（`sglang/srt/mem_cache/allocator.py:35`，`BaseTokenToKVPoolAllocator`）；其余张量走 PyTorch 默认 caching allocator。omni 侧无 `PYTORCH_CUDA_ALLOC_CONF` / `set_per_process_memory_fraction`。
- 多流下跨 stream 共享的张量需要 `record_stream(alt_stream)` 防止过早回收。仓内已有先例：`deepseek_v2.py:769`（shared experts 在 alt_stream + `record_event`）、`eagle_worker_v2.py:691-693`。**方案 B 一旦引入 alt stream，所有跨流张量都要补 `record_stream`**——这是分散在多处的改动面，易漏。

### c) Prepare 步骤的 GPU 化潜力
- 能 offload 到 GPU stream 的：`_populate_cg_buffers` 里的 `pool[rows] → _cg_active_*` gather（`model_runner.py:96-99`，已是 GPU op）；`_extract_decode_sampling_params` 的 3 个 D2H（`model.py:57`）**根本不该每步做**——值是静态的，缓存到 host 一次即可消除，不需要 B。
- **纯 Python 控制流、B 也救不了**：`acquire_row` 的 dict 查找、stop 判定的 Python 分支、`output_processor` 记账、`get_next_batch_to_run` 调度。这些是 ~1.1ms 里 B 同样无法 overlap 的部分——**方案 A 把它们藏到 GPU forward 背后，B 也只能藏、不能消除**。
- 估算：B 相比 A 能多 overlap 的绝对时间很有限（A 已经把整段 CPU 藏到 forward 后；B 的额外收益只在「forward 比 CPU 短，A 藏不下」的场景，即大 batch / 小模型采样占比高时），不值当前的复杂度。

### d) 跟 A 的关系
- B **不必叠在 A 之上**，但**最好如此**：A 先把 launch/resolve 拆开（execute 状态机 + event），B 只是把 resolve 半进一步推到第二个 stream 上用 `wait_event` 做 GPU-side 串接。若 A 已 merge，B 的增量 ≈ 引入 alt stream + 多流 capture（20-40 LOC runner）+ 各处 `record_stream`（分散）+ 把 A 的 `event.query()` 升级成 `stream.wait_event()`。
- **建议：本次不做 B**。理由：(1) B 的核心机制（多 stream + `wait_event`）是你为方案 A 明确划掉的边界；(2) 标准 CUDA Graph 不原生支持单 graph 多流，改动有真实风险；(3) 静态估算 B 相对 A 的增量收益有限且场景受限；(4) A 是 B 的干净前置。留作 follow-up PR，A 实现时为它预留 hook（见 §7 给 design.md 的接口约定）。

---

## 7. 新建 async issue 如何 reference #564

`#564`（`[Feat] Higgs TTS: batch the 3 per-step D2H syncs into 1`，作者 Yichi Zhang，我已认领 "Working on it!"）**就是 D2H 合并那件事本身**，由 PR #572 / commit `9f235ff` 落地；它引用 PR #534（profiling）、PR #503（引入 3-sync 的 CG capture）、tracking issue #478。

新建的 async decode 调度 issue 应**作为 #564 的 follow-up**，建议这样 link：

> Follow-up to #564. #564 把 `_collect_step_outputs_cg` 的 3 个 per-step D2H 合并成 1 个；但 A/B 显示该合并 **latency-neutral**——因为瓶颈不是 sync 的**数量**，而是每步那个不可避免的 D2H 同步把 ~1.1ms 的 CPU 工作（collect 循环 + prepare + 调度，占每步 ~24%）和 GPU forward **完全串行化**了。#564 原始假设「sync 数 × 30-50µs」高估了 D2H 计数的影响；真正要啃的是这段串行 CPU 气泡。本 issue 用 one-step lookahead 异步调度（单 stream + CUDA Event）把它藏到 GPU forward 背后。同源 profiling: PR #534；CG capture: PR #503；tracker: #478。

即：**继承 #564 的 profiling 上下文与同一函数 `_collect_step_outputs_cg`，但把目标从「减少 D2H 数量」修正为「overlap 掉 D2H 之后那段串行 CPU」**，并诚实指出 #564 估算与 A/B 实测的差异。

---

## 8. 风险点 / 开放问题

**待你定的关键决策**
- **方向**：路 A（复用骨架）vs 路 B（独立新建）——见 §5，留到 Phase 1→2 边界。

**正确性风险（Phase 2/3 必须验证）**
1. **错位一步的 `_cg_*` 缓冲竞争**：lookahead 后，第 N+1 步的 `_populate_cg_buffers` 写 `_cg_*` 与第 N 步 replay 读 `_cg_*` 是否在单 stream 上仍严格串行（§5.1）。需用 `compute-sanitizer` / 单测验证无 in-place race。
2. **stop 延后一步的语义**：序列末尾多算 1 步并丢弃——要保证 `data.output_codes` 不混入溢出步、`finished_reason` 时序正确、token 计数（`generation_steps`，`base.py:142`）不偏移。
3. **warmup（第一步没有 N-1 结果）**：第一步必须退化成同步（允许用 `event.synchronize()`，在你的边界内）。
4. **abort / preempt / 中途加入新请求**：`OmniScheduler.abort`（`:745`）会从 `running_batch/cur_batch/last_batch` 移除请求；lookahead 多了一层 in-flight 状态，abort 要能清理「已 launch 未 resolve」的那一步。
5. **batch 组成变化**：lookahead 用 N-1 的 stop 信息 prepare N+1，若中间 batch 增删请求，`_cg_row_indices` / padding row 的对应要重新核对（`_populate_cg_buffers:68-74`）。

**独立的低垂果实（与本工作正交，可单独 PR）**
- `_extract_decode_sampling_params` 每步 3 个 D2H 读静态采样参数（`model.py:57`）纯属冗余——host 侧缓存一次即可消除，不依赖任何 async 机制。建议作为前置小 PR 或在本 PR 顺带修（但保持 atomic commit 分开）。

**需要 GPU 才能验证的点（静态分析到此为止）**
- 3.72ms / 1.1ms / 45μs 的拆分复现（torch.profiler + Nsight）；
- 错位后 1.1ms 是否真被藏进 forward；
- §6c 中「B 相对 A 增量收益有限」的量化（需对比不同 batch / forward 时长）。

---

## 附：关键文件速查

| 关注点 | 文件:行 |
|---|---|
| OmniScheduler 串行循环 / overlap 骨架 / 启用分支 | `sglang_omni/scheduling/omni_scheduler.py:804` / `:834-877` / `:732-737` |
| `run_batch` → `execute` 桥接 | `sglang_omni/scheduling/omni_scheduler.py:584,607` |
| 单体 `execute()` 流程（拆点） | `sglang_omni/model_runner/base.py:33`（prepare`:75` / forward`:82` / post`:114` / output`:139`） |
| Higgs prepare（含 3× 冗余 D2H） | `sglang_omni/models/higgs_tts/model_runner.py:53`（`:76`→`model.py:57`） |
| Higgs collect（唯一关键 D2H + 循环） | `sglang_omni/models/higgs_tts/model_runner.py:128`（D2H`:157`、循环`:161-181`） |
| GPU 内采样状态机 / GPU 内 AR 反馈 | `sglang_omni/models/higgs_tts/model.py:321` / `:430-446` |
| overlap flag 设置点 | `*/stages.py`、`qwen3_omni/talker_scheduler.py:28`、`bootstrap.py:57` |
| 上游 FutureMap overlap | `/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py:33,109,118,142` |
| 上游 CUDA Graph runner（单流假设） | `/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py:472,552,842` |
| 多流先例（record_stream / alt_stream） | `.../models/deepseek_v2.py:769`、`.../speculative/eagle_worker_v2.py:691-693`；分段 graph `.../piecewise_cuda_graph_runner.py` |
