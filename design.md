# Async Decode Scheduling — Phase 2 设计文档

> 读者假设：你熟悉 PyTorch / CUDA stream 语义，但**不熟悉 sglang-omni**。读完本文你应能知道：改哪些文件/函数、每个边界情况怎么处理、用什么指标验证。背景见同目录 `investigation.md`。
>
> 代码引用：`sglang_omni/...` 为本仓库；`base.py` = `sglang_omni/model_runner/base.py`；`model_runner.py` = `sglang_omni/models/higgs_tts/model_runner.py`；`omni_scheduler.py` = `sglang_omni/scheduling/omni_scheduler.py`。

---

## 0. 术语与决策（统一口径）

| 轴 | 取值 | 本 PR |
|---|---|---|
| **Lookahead 实现路径** | 路 1 = 复用 `_event_loop_overlap` 骨架；**路 2 = 独立新建 execute 状态机** | **路 2** |
| **CUDA Event 用法层级** | **层级 A = single-stream + `event.query()`**；层级 B = multi-stream + `stream.wait_event()`（follow-up） | **层级 A** |

**一句话方案**：把 `ModelRunner.execute()`（`base.py:33`）拆成 `execute_launch()` 和 `execute_resolve()`；scheduler 用一个新的 `_event_loop_async_decode` 循环，每个 iteration **先 launch 当前步（把 GPU forward + D2H 拷贝预先入队、record event），再 resolve 上一步（纯 CPU 读 pinned host buffer + collect）**，让上一步的 ~1.1ms CPU 工作藏到当前步 GPU forward（3.72ms）背后。单 stream、只用 `event.query()`（warmup/fallback 用 `synchronize()`）。全部在 `--enable-async-decode` flag 后面，默认关。

这是上游 FutureMap overlap（`investigation.md` §4.2）思想的轻量化版本：**同样错位一步消费 sample 结果，但不引入 future-token kernel、不改 CUDA Graph、保持单 stream**。PR description 里据此 reference 上游 FutureMap。

---

## 1. 核心机制与 `execute()` 拆分

### 1.1 不变式（invariant，写进 docstring）

1. **同一时刻最多 1 个 in-flight step**：已 `execute_launch` 但未 `execute_resolve` 的 step 恰好 ≤1（用 `self._pending` 表示）。
2. **错位一步消费**：第 N 步的 sample 结果在第 N+1 步的 launch *之后* 才被 resolve/消费。
3. **末尾最多多算一个被丢弃的 step**：模型在 forward 第 S 步判定某请求 `generation_done` 后，该请求仍会被包含进第 S+1 步的 batch（多一次无害的 forward），其 S+1 步输出在 resolve 时由现有 `_cg_was_done` 机制丢弃（`model_runner.py:168` 的 `if was_done_cpu[b]: continue`），第 S+2 步起被调度器剔除。
4. **GPU 侧 AR 反馈不受影响**：上一步 codes→下一步 embed 全程留在 GPU（`_cg_active_last_codes`，`model.py:430-446`），lookahead 只改变 *host 何时消费*，不改变 GPU 入队顺序。

### 1.2 `execute_launch()` / `execute_resolve()` 定义

把现在 `execute()`（`base.py:33`）单体流程按「GPU 入队」与「host 消费」切开：

**`execute_launch(sched_output) -> None`**（只入队，不等 GPU）
1. `get_model_worker_batch` / `ForwardBatch.init_new`（`base.py:47,64`）
2. `prepare_decode`（`base.py:75` → `_populate_cg_buffers`）：gather `pool→_cg_active_*`（GPU→GPU）+ row indices H2D。**注意**：此处读的 `pool` 已含上一步 launch 里的 scatter（GPU→GPU、stream 有序），不依赖上一步 resolve。
3. `forward_batch_generation`（`base.py:82`）：CG replay + on-GPU 采样，写 `_cg_codes_BN` / `_cg_active_*`。
4. **`post_decode_launch`（新 hook，见 §1.6）**：scatter `_cg_active_*→pool`（GPU→GPU）+ pack 三张量进 `_cg_collect_staging`（GPU→GPU）+ **把 staging 非阻塞拷进 pinned host buffer**（`host_buf.copy_(staging, non_blocking=True)`）+ **`event.record()`**。
5. 把 `_PendingStep`（§1.5）存进 `self._pending`。返回。

**`execute_resolve(pending) -> ModelRunnerOutput`**（纯 CPU，不入队新 GPU 工作）
1. `if pending.event.query(): pass`（命中，overlap 成功）`else: pending.event.synchronize()`（fallback，见 §3）。
2. **`post_decode_resolve`（新 hook）**：从 `pending.host_buf` 切片（host 侧）+ per-request Python collect 循环（append codes / stop 判定 / 建 `next_token_ids`，即现 `model_runner.py:158-181`）。
3. `output_processor.process` + per-req 记账（`base.py:139-148`），set `pending.schedule_batch.output_ids`，返回 `ModelRunnerOutput`（带本步 `req_ids`，用于路由）。

> `execute()`（原同步入口）保留：当 `--enable-async-decode` 关时，`execute = execute_launch(同步变体) + 立即 execute_resolve`，行为与现在逐字节一致（回归保护）。

### 1.3 ⚠️ 设计决策 D1：实际顺序是 launch(N) → resolve(N-1)，而非字面的 resolve→launch

你 prompt 里写的是「resolve(N-1) → launch(N) → return」。**为了真正把那段 CPU 藏住，落地顺序必须反过来：先 launch(N)（把 forward(N) 入队让 GPU 忙起来），再 resolve(N-1)（纯 CPU 工作与 forward(N) 重叠）。** 论证：

- 单 stream 是 FIFO。CPU 异步入队 kernel，直到撞上同步点（`.cpu()`/`event.synchronize()`/`.item()`）才阻塞。
- 若按字面 `resolve(N-1) → launch(N)`：resolve(N-1) 跑在 launch(N) *之前*，此刻 forward(N-1) 已完成、forward(N) 还没入队 → **GPU 空闲**，resolve(N-1) 的 collect 循环白白占着 CPU 而 GPU 闲着，等于没 overlap（只 overlap 了后续的 `process_batch_result` + compose，没 overlap 掉 collect 本身，而 collect 是 1.1ms 的大头）。
- 按 `launch(N) → resolve(N-1)`：forward(N) 先入队（GPU 忙 3.72ms），resolve(N-1) 的 D2H 早在 launch(N-1) 就预入队完成、`query()` 立即命中，collect 纯 CPU 跑在 forward(N) 背后 → **整段 1.1ms 被藏住**。

CUDA 语义依据：PyTorch CUDA semantics（streams 异步、FIFO、event 非阻塞查询）<https://pytorch.org/docs/stable/notes/cuda.html>；`torch.cuda.Event.query/record` <https://pytorch.org/docs/stable/generated/torch.cuda.Event.html>。

> **请在 Phase 2 review 时确认采用 launch-first**。若你坚持字面 resolve-first，收益会缩水（collect 循环不被藏），我可在 design 里降级处理——但不建议。

### 1.4 关键正确性/性能点：D2H 必须在 launch 预入队 + pinned + host buffer 双缓冲

- **D2H 入队点**：`host_buf.copy_(staging, non_blocking=True)` 放在 **launch 的 step 4**（紧跟 forward），这样它在 stream 里排在 forward(N+1) *之前*；resolve 只需 `query()/wait` 这个早已入队的拷贝，不会被 forward(N+1) 挡住。若把 D2H 留在 resolve 再发，单 stream 下它会排到 forward(N+1) 后面 → 适得其反。
- **必须 pinned + non_blocking**：现状 `staging[:n_real].cpu()`（`model_runner.py:157`）分配 pageable host 内存且**阻塞**。要换成预分配的 **pinned**（`torch.empty(..., pin_memory=True)`）host buffer + `non_blocking=True`，否则拷贝同步、无法异步。依据：PyTorch pinned-memory 异步拷贝 <https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers>。
- **host buffer 双缓冲（ping-pong）**：resolve(N) 在 host 侧读 `host_buf` 时，launch(N+1) 的 D2H 正要写 host buffer。两者是「CPU 读 vs GPU 写」，**不被 stream 顺序保护** → 必须用 2 个 pinned buffer 按 step 奇偶轮换。
- **device staging 单缓冲即可**：`_cg_collect_staging` 的 pack(N)→D2H(N)→pack(N+1) 全在 stream 上有序，无需双缓冲。
- **shadow `_cg_active_*` 单缓冲即可**：gather→forward→scatter 在同一 launch 内 stream 有序。

### 1.5 `ModelRunner` 新增状态字段（实现你要求的 `pending_event` / `pending_staging_buf` / `pending_metadata`）

```python
# 仅在 async 模式分配
self._async_enabled: bool
self._host_staging_buffers: list[torch.Tensor]   # 2 个 pinned，ping-pong（= pending_staging_buf 的池）
self._staging_slot: int                           # 0/1，每次 launch 翻转
self._pending: Optional[_PendingStep]             # None = 无 in-flight

@dataclass
class _PendingStep:                               # = pending_event + pending_metadata
    event: torch.cuda.Event                       # 你说的 pending_event
    host_buf: torch.Tensor                        # 指向本步用的 pinned buffer
    requests: list                                # 本步 sched_output.requests（resolve 路由用）
    schedule_batch: Any                           # 用于 set output_ids
    batch_result: Any                             # 持 logits_output 引用（next_token_ids 的 device）
    n_real: int
```

`assert self._pending is None` 在 launch 入口（捕获「N-1 还没 resolve 就又 launch」这种本不该发生的情况，对应你要求的 assertion）。

### 1.6 model-specific hook 拆分（增量、向后兼容）

现有 hook：`prepare_decode` / `post_decode`（`base.py:171-185`，Higgs 在 `model_runner.py:43-51` 覆写）。新增两个，把 `post_decode` 切成 GPU 半 + host 半：

| 新 hook | 职责 | base 默认实现（向后兼容） |
|---|---|---|
| `post_decode_launch(result, forward_batch, requests, host_buf)` | scatter + pack + 非阻塞 D2H 入 `host_buf`（GPU 半） | no-op |
| `post_decode_resolve(host_buf, requests) -> None` | host 切片 + collect 循环（host 半），mutate `data.output_codes` 等 | 调用 `self.post_decode(...)`（即旧逻辑全放 resolve） |

**含义**：没迁移 split 的模型在 async 模式下「能正确跑，但 D2H 落在 resolve、无 overlap 收益」——安全降级。**本 PR 只迁移 Higgs**（把现 `_collect_step_outputs_cg` 的 `model_runner.py:146-156`+D2H 放进 `post_decode_launch`，`158-181` 放进 `post_decode_resolve`）。qwen3-omni 等后续 PR 再迁。

### 1.7 scheduler 新 event loop（`omni_scheduler.py`，gated）

`start()`（`omni_scheduler.py:732-737`）加一条分支：

```python
def start(self):
    self._running = True
    if self.enable_async_decode:        # 新 flag
        self._event_loop_async_decode()
    elif self.enable_overlap:
        self._event_loop_overlap()
    else:
        self._event_loop_normal()
```

```python
def _event_loop_async_decode(self):
    pending_batch = None                 # 与 self._model_runner._pending 配对的 batch（resolve 路由用）
    while self._running:
        recv = self.recv_requests(); recv += self._take_deferred_request_payloads()
        self.process_input_requests(recv)
        if self._engine_paused:
            time.sleep(0.001); continue

        batch = self.get_next_batch_to_run()    # 用当前 finished flags 组 batch（见 §2 不变式 3）
        self.cur_batch = batch

        # —— 1) 先 launch 当前步：GPU 忙起来 ——
        if batch:
            self._run_batch_launch(batch)        # 内部调 model_runner.execute_launch；不等 GPU

        # —— 2) 再 resolve 上一步：纯 CPU，与 forward(batch) 重叠 ——
        if pending_batch is not None:
            result = self._run_batch_resolve()   # 内部调 model_runner.execute_resolve + stream emit
            if result is not _FAILED_BATCH_RESULT:
                self.process_batch_result(pending_batch, result)

        prev_pending_batch = pending_batch
        pending_batch = batch.copy() if batch else None

        if batch is None and prev_pending_batch is None:
            self.self_check_during_idle(); time.sleep(0.001)   # 沿用 #543 的 GIL-yield
        self.last_batch = batch
```

`_run_batch_launch` / `_run_batch_resolve` 是把现 `_run_batch`（`omni_scheduler.py:584-640`）按 launch/resolve 拆开的薄包装；stream 输出 emit（现 `:609-628`）挪到 `_run_batch_resolve` 里，用 resolve 返回的 `mr_output.outputs[rid]` 路由（**因此音频 chunk 的发出比现在晚一个 decode step，对 TTS 流式可接受，~几 ms**）。

> 这是「路 2 独立新建」：不碰 `_event_loop_overlap`（它仍禁用），新循环自包含、可单独 revert、debug 半径小。pending 的 CUDA 状态在 ModelRunner，pending 的 batch 在 scheduler。

---

## 2. 边界情况处理

| 情况 | 策略 |
|---|---|
| **第一步 warmup（无 N-1）** | 第一个 iteration `pending_batch is None` → 只 launch、不 resolve。自然由 `if pending_batch is not None` 处理。**首步 event 允许直接 `synchronize()`**（在层级 A 边界内）。无需特判。 |
| **末尾 stop 延后一步** | 不变式 3。docstring 明写：「请求在 forward 第 S 步被判 done 后，会在第 S+1 步多收到一次无害 forward，其输出经 `_cg_was_done` 丢弃；KV 释放相应晚一步。」需验证 GPU 采样状态机对「已 `generation_done` 的行再跑一步」幂等安全（与 padding row 同类，见 §7 正确性项）。 |
| **abort / preempt** | in-flight 的那一步 **始终 resolve 完**（绝不丢弃已 record 的 event/已发的 D2H，否则 buffer/event 生命周期泄漏），但被 abort 的请求在 resolve 的 collect 里跳过（不 append、不 emit）。实现：把 `self._pending` 对应的 batch 纳入 `_mark_running_request_aborted` / `_release_*` / abort 的批次清理元组（现 `omni_scheduler.py:769-771,777,790` 只含 `running/cur/last_batch`，**加上 pending_batch**）；resolve collect 增加 `if req.to_finish or rid in aborted: skip`。 |
| **新请求中途加入** | lookahead 只作用于「已在某个已 launch 的 decode batch 内」的请求。新请求先走 prefill（独立路径），在下一个 `get_next_batch_to_run` 进入 decode，其首个 decode step = 一次普通 launch，无历史 pending。自然安全，无需特判。 |
| **drain / 关停 / batch 突然为空** | 循环退出或 `batch is None` 但 `pending` 非空时，必须把最后一个 in-flight step resolve 完（`flush`）：`if pending_batch: result = self._run_batch_resolve(); process_batch_result(pending_batch, result)`。保证不漏最后一步输出。 |

---

## 3. CUDA Event 用法

- **record 点**：在 `post_decode_launch` 里、**非阻塞 D2H 拷贝入队之后**立刻 `event.record()`。这样 `event.query()==True` 精确等价于「本步 staging 已拷到 host buffer、resolve 可以读了」。
  - 反例论证：若 record 在 forward 之后、D2H 之前，则 `query()==True` 只意味着 forward 完成，host buffer 还没好，resolve 仍要等拷贝 → event 语义不精确。故 record 紧跟 D2H。
- **query 策略（非 spin）**：进 `execute_resolve` 时 **`query()` 一次**。命中 → 立即读 buffer（steady-state 应几乎总命中，因为 D2H 早在上一个 launch 入队、forward(N) 期间就完成了）。未命中 → **fallback `event.synchronize()`**（一次阻塞等待，不轮询、不 spin）。
- **可观测性**：加计数器 `async_resolve_query_hit / async_resolve_fallback`，作为「overlap 是否真生效」的运行时证据（Phase 4 报告引用）。fallback 率高 ⇒ forward 比 CPU 工作短（小 batch 大模型 or warmup），说明该场景收益有限——诚实暴露。
- **`synchronize()` 仅用于**：首步 warmup、fallback、错误恢复——符合你定的层级 A 边界（禁用 `stream.wait_event` / 第二 stream）。

---

## 4. 与各 omni 模型的兼容性

lookahead 机制放在 `base.py` 的 `execute_launch/resolve` + `OmniScheduler` 层，**对 5 个 AR 模型都是同一套**（`investigation.md` §4.4）。是否有收益，取决于该模型 `post_decode` 里是否有「阻塞 host 同步」可藏。审计结果：

| 模型 | post_decode host 同步 | 适用度 | 说明 |
|---|---|---|---|
| **Higgs TTS** | 1 个批量 `.cpu()`（`model_runner.py:157`）+ per-req 循环 | **高（首选验证）** | profiling 来源；本 PR 唯一迁移 split hook 的模型 |
| **Qwen3 TTS** | per-row `.item()`（`qwen3_tts/model_runner.py:90`） | 中（待 profile） | 同类气泡，后续 PR 迁 |
| **Voxtral TTS** | `.sum().item()`+per-row `.item()`（`voxtral_tts/model_runner.py:88,120`） | 中（待 profile） | 同上 |
| **FishAudio S2-Pro** | `.tolist()`+`.item()`（`fishaudio_s2_pro/model_runner.py:27,190-191`） | 中（待 profile） | 同上 |
| **Qwen3-Omni talker** | **无阻塞 host 读**：结果留 GPU `.clone()`、GPU tensor 经 outbox 发下游 code2wav（`talker_model_runner.py:91-133`） | **低/边际** | 见下 |

**Qwen3-Omni talker：no-op 还是有收益？** —— 对它，lookahead **机制上仍套用**（execute_launch 跑 forward+emit、execute_resolve 近乎空），但因为它的 `post_decode` 本就把 host 移出了关键路径，**预期收益边际**。它的每步开销若存在，更可能在 `prepare_decode`（`prepare_decode_buffers`/`_write_feedback_buffers` 的 H2D）或 emit 循环的 clone，那不是本机制的靶子。**诚实结论：优先目标里 Higgs 收益大、Qwen3-Omni talker 大概率小，需各自 profile 后再决定是否迁 split hook。** 不强行宣称 Qwen 收益。

---

## 5. CUDA Graph 兼容性确认

- **现状**：forward（CG replay，上游 `cuda_graph_runner.replay`，`investigation.md` §6a）只 capture 模型 transformer + 采样 kernel；`replay()` 返回后才回到 omni runner。
- **本设计的 event/D2H 全在 graph 外**：`post_decode_launch` 的 scatter/pack/D2H/`event.record()` 都在 `forward_batch_generation`（`base.py:82`）返回 *之后*执行，**不在 capture 区间内**。staging pack 读的是 graph 写入的固定 buffer（`_cg_codes_BN` 等），这是现同步模式已有的模式，无新增 capture 风险。
- **为什么不能让 record 进 graph**：若 `event.record()` 被 capture 进 graph，每次 `replay()` 会重放同一个 record op、event 语义不可控。**设计强制：record 在 Python 侧、replay 之后**。
- **实验验证方案**：
  1. CG 开启 + async 开启，跑 ≥50 个 decode step，断言无 capture/replay 报错。
  2. 同 seed greedy，async vs sync 的 `output_codes` 必须 **逐 token 相等**（见 §7）。
  3. 用 `nsys` 看 timeline：确认 `cudaEventRecord` / `cudaMemcpyAsync` 落在 graph replay kernel *之外*、且 D2H(N) 排在 forward(N+1) 之前。
  4. （可选）`compute-sanitizer --tool synccheck` 跑短序列，确认无 stream/event 误用。

---

## 6. 改动清单（文件 / 函数级 —— 给 ratish）

| 文件 | 函数 / 位置 | 改动 |
|---|---|---|
| `sglang_omni/model_runner/base.py` | `execute()`（:33） | 拆出 `execute_launch()` / `execute_resolve()`；`execute()` 在 async-off 时 = launch+立即 resolve（行为不变） |
| 同上 | 新增 hook `post_decode_launch` / `post_decode_resolve`（默认 no-op / 调 `post_decode`） | §1.6 |
| 同上 | 新增字段 `_async_enabled` / `_host_staging_buffers` / `_staging_slot` / `_pending` + `_PendingStep` dataclass | §1.5 |
| `sglang_omni/models/higgs_tts/model_runner.py` | `_collect_step_outputs_cg`（:128） | 拆成 `post_decode_launch`（:146-156 + 非阻塞 D2H 入 pinned buf + record）与 `post_decode_resolve`（:158-181，从 host buf 切片） |
| 同上 | （顺带，独立 commit）`_extract_decode_sampling_params`（:101→`model.py:57`） | host 侧缓存静态采样参数，消除每步 3 个冗余 D2H（`investigation.md` §8） |
| `sglang_omni/models/higgs_tts/model.py` | `_cg_collect_staging` 旁（:173 附近） | 加 2 个 pinned host buffer（ping-pong） |
| `sglang_omni/scheduling/omni_scheduler.py` | `start()`（:732）、新增 `_event_loop_async_decode`、`_run_batch_launch`/`_run_batch_resolve`（拆 `_run_batch`:584） | §1.7 |
| 同上 | `abort` / `_mark_running_request_aborted` / `_release_*`（:745,774,788） | 把 `_pending` 对应 batch 纳入清理元组 | 
| `sglang_omni/scheduling/bootstrap.py` | `enable_overlap` 提取处（:57） | 加 `enable_async_decode = getattr(server_args, "enable_async_decode", False)`，传给 OmniScheduler |
| `sglang_omni/models/higgs_tts/stages.py` | scheduler 构造（:261,287） | 设 `server_args.enable_async_decode = True`（Higgs opt-in；其余模型暂不开） |
| server args / CLI | 定义 `--enable-async-decode`（默认 False） | 与 `disable_overlap_schedule` 并列 |

---

## 7. 测试与验证计划

**单元测试（状态机，不需 GPU 或用 fake event）**
- `pending` 轮换：launch→resolve 配对、`assert _pending is None` 触发条件。
- warmup：首 iteration 只 launch 不 resolve。
- drain：循环结束 flush 最后一步。
- abort：in-flight 步被 abort 的请求在 resolve 跳过、其余请求正常。
- ping-pong：连续两步用不同 host buffer（防 CPU 读/GPU 写竞争）。
- mock `event.query()` 返回 False → 验证走 `synchronize()` fallback 分支。

**正确性 A/B（需 GPU，1 张卡）**
- 同 seed、greedy，Higgs decode：`--enable-async-decode` on vs off，**逐请求 `output_codes` 逐 token 相等**（这是硬门槛，不等就是 bug）。
- 含末尾 stop、abort、mid-stream join 的混合负载下 finish_reason / token 计数（`generation_steps`，`base.py:142`）与 baseline 一致。
- CG 兼容 §5 的 4 项。

**性能 A/B（需 GPU，用 `gpu-perf-ab` skill 的方法论）**
- 指标：decode tokens/sec、p50/p99 inter-token latency、`async_resolve_query_hit` 率。
- 复现 `investigation.md` 的 3.72ms/1.1ms profile（torch.profiler + nsys），验证 1.1ms 是否真被藏进 forward（看 GPU 利用率连续性、CPU collect 是否落在 forward kernel 窗口内）。
- 矩阵留给 Phase 4（小/中/大 batch；Higgs 必测，Qwen3-Omni talker 测了才知收益）。诚实写适用边界。

---

## 8. Follow-up：Plan B（层级 B，multi-stream）设计草图

> 仅 1-2 页草图，**本 PR 不实现**。基于 `investigation.md` §6。

层级 A 已把整段 ~1.1ms CPU 藏到 forward 后。层级 B 要榨的是「forward 比 CPU 短、A 藏不下」的残余场景（小模型/大 batch 采样占比高时）。B 的核心 = 第二个 CUDA stream + `stream.wait_event()` 做 GPU-side 串接。两个分叉：

- **分叉 (i)：自建 dual-stream（轻）**——把 resolve 的 D2H + 后续小 GPU op 放到 alt stream，用 `wait_event` 让 alt stream 等 forward 完成、主 stream 不被 D2H 占用。需处理：标准 `torch.cuda.CUDAGraph` 单流限制（`investigation.md` §6a，或改用 `piecewise_cuda_graph_runner`）、跨流 tensor 的 `record_stream`（参考 `deepseek_v2.py:769`）。增量 ~20-40 LOC（runner）+ 分散的 record_stream。
- **分叉 (ii)：接入上游 FutureMap（重）**——直接 port `sglang/srt/managers/overlap_utils.py` 的 `FutureMap`/`resolve_future`，让 omni 复用上游成熟的 launch/resolve 拆分 + future-token kernel。收益最完整但耦合上游、改动最大，且与层级 A 的「不引入 future kernel」初衷相悖——只在 (i) 不够用时才考虑。

**层级 A 要给 B 预留的 hook（A 实现时就做，省得 B 返工）**：
1. `_PendingStep` 里 event/buffer 的所有权清晰，B 只需把 resolve 的 D2H 从「主 stream + query」换成「alt stream + wait_event」，不动状态机结构。
2. `post_decode_launch/resolve` 的边界即「GPU 半 / host 半」的天然切点，B 可在中间插入 alt-stream 段而不改 hook 签名。
3. D2H buffer 已 pinned + ping-pong，B 直接复用。
4. `enable_async_decode` flag 预留 `async_decode_level: "A" | "B"` 子选项位（先只接受 "A"）。

---

## 9. 风险登记与开放问题

| # | 项 | 处理 |
|---|---|---|
| R1 | 末尾多算一步要求 GPU 采样状态机对「已 done 行再跑一步」幂等 | §7 正确性 A/B + 读 `batched_step_direct`/padding row 逻辑确认；若不幂等需在 prepare 把已 done 行当 padding 处理 |
| R2 | host buffer 双缓冲漏做 → CPU 读/GPU 写竞争、间歇性错码 | §1.4 强制 ping-pong + §7 ping-pong 单测 |
| R3 | D2H 误留在 resolve（没挪进 launch）→ 单 stream 下排在下一个 forward 后、零收益甚至变慢 | §1.4；code review 检查点；性能 A/B 的 query_hit 率会暴露 |
| R4 | 音频 chunk emit 晚一步 | §1.7，TTS 流式可接受；如有端到端首包延迟门槛需复核 |
| R5 | **D1 顺序待你确认**（launch-first vs 字面 resolve-first） | §1.3，Phase 2 review 拍板 |
| O1 | 其余 3 个 TTS 模型的迁移与各自收益 | 本 PR 不迁，安全降级；后续 PR + profile |
| O2 | `enable_async_decode` 与 `disable_overlap_schedule` 的优先级/互斥 | start() 分支顺序：async > overlap > normal；文档写明 |

**Phase 3 实现 commit 切分（原子、可单独 revert）**：
1. `_PendingStep` + ModelRunner 状态字段 + pinned ping-pong buffer（纯数据结构，无行为变化）
2. `execute_launch/resolve` 拆分 + `post_decode_launch/resolve` hook（async-off 行为不变，加回归测试）
3. Higgs `_collect_step_outputs_cg` 迁移到 split hook
4. `_event_loop_async_decode` + scheduler launch/resolve 拆分 + abort 纳管
5. `enable_async_decode` flag + bootstrap + Higgs stages opt-in + CLI
6. 单元测试
7.（独立）`_extract_decode_sampling_params` 冗余 D2H 缓存

每个子任务完成后停下给你 review diff 再继续下一个。
