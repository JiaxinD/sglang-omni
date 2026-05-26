# Plan B 预研：multi-stream + `stream.wait_event()` 的可行性与改动面

> 只读架构调研。基于分支 `feat/async-decode-lookahead` 的真实代码。Plan A（single-stream + `event.query()` 一步 lookahead）已落地；本文判断 A 是否为 B 铺好了路、B 需要什么。**不实现 B、不决定是否做 B。**
>
> 引用约定：`base.py` = `sglang_omni/model_runner/base.py`；`omni_scheduler.py` = `sglang_omni/scheduling/omni_scheduler.py`；`higgs model_runner.py` = `sglang_omni/models/higgs_tts/model_runner.py`；`upstream cuda_graph_runner.py` = `/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`；`upstream utils.py` = `/sgl-workspace/sglang/python/sglang/srt/models/utils.py`。

---

## 结论速览（feasibility verdict）

**Plan B 可行（feasible），且 Plan A 是干净的前置——不需要推倒重来。** 关键证据：
1. 仓内已有「**CUDA Graph capture 期间** fork/join 第二条 stream + `wait_stream`」的成熟先例（`upstream utils.py:248-256`，由 `get_is_capture_mode()` 守门，Qwen3-Omni talker / Qwen3-TTS 已在用，见 `talker.py:376` / `thinker_model.py:304`）。这直接推翻了 investigation.md §6a「标准 CUDA Graph 不原生支持单 graph 内多流」的悲观假设——**graph 内多流在本仓库是已验证可行的**。
2. Plan A 的 `_PendingStep` / `post_decode_launch`/`resolve` 切点 / pinned ping-pong buffer / `execute_launch`/`resolve` 拆分，B 全部能复用，B 的增量主要是把 resolve 的「`event.query()`」换成「alt-stream + `wait_event`」并把 D2H 挪到 alt stream。
3. **但 B 的收益靶点与 A 完全不同**：Plan A 的 D2H/collect 已经全在 graph 外（`post_decode_launch` 在 `forward_batch_generation` 返回之后跑，base.py:144-149），所以 B 在「graph 外」加 alt stream 几乎拿不到额外收益（A 已经把它藏进 forward 了）。B 真正能榨的是 investigation.md §6c 划定的残余场景，需要 GPU profile 才能确认值不值得——这一点纯静态分析无法定论，下文 Q3/Q6 明确标出。

---

## Q1：上游 / 仓内 multi-stream 实践

### Q1.1 所有 `torch.cuda.Stream()` / `wait_stream` / `wait_event` 用法（file:line + 作用）

**omni 仓内**（`grep "torch.cuda.Stream()\|alt_stream\|wait_stream\|wait_event\|record_stream"`）：

| file:line | 用法 | 作用 |
|---|---|---|
| `talker.py:376` | `alt_stream = torch.cuda.Stream()` | Qwen3-Omni talker 文本 backbone 构造时建一条 alt stream，传给每个 decoder layer |
| `talker.py:507` | `alt_stream = torch.cuda.Stream()` | 同上，另一个 backbone 变体 |
| `talker.py:709`、`talker.py:1377` | `alt_stream=attn.alt_stream` 传给 `apply_qk_norm` | 把 q_norm / k_norm 分到两条 stream 并行 |
| `thinker_model.py:620` | `alt_stream = torch.cuda.Stream()` | Qwen3-Omni thinker 同样模式 |
| `thinker_model.py:250,304` | `self.alt_stream = alt_stream` → 传 `apply_qk_norm` | 同上 |
| `qwen3_tts/sglang_model.py:811` | `alt_stream=attn.alt_stream` | Qwen3-TTS 复用同一 `apply_qk_norm` 模式 |
| `vendor/sglang/models.py:35-47` | `apply_qk_norm(..., alt_stream=...)` thin wrapper | 仓内 monkeypatch 包装上游 `apply_qk_norm`，把 alt_stream 透传下去 |

**真正做 fork/join 的地方在上游 `apply_qk_norm`**（`upstream utils.py:248-256`）：
```python
if alt_stream is not None and get_is_capture_mode():
    current_stream = torch.cuda.current_stream()
    alt_stream.wait_stream(current_stream)        # alt 等主流到此点
    q_by_head = q_norm(q.reshape(-1, head_dim))   # q_norm 留主流
    with torch.cuda.stream(alt_stream):
        k_by_head = k_norm(k.reshape(-1, head_dim))# k_norm 进 alt 流并行
    current_stream.wait_stream(alt_stream)        # 主流回收 alt 的结果
```
这是经典 **fork → 并行 → join** 模式，用 `wait_stream`（≈ 在对方 stream 上 record 一个 event 再 wait_event 的语法糖）。

**上游 record_stream / wait_event 先例**（omni 没有，B 要照抄的对象）：
- `deepseek_v2.py:1010-1015`：`alt_stream.wait_stream(current)` → `with stream(alt): shared_output = ...; shared_output.record_stream(alt_stream); shared_event = alt_stream.record_event()`。这是 shared-experts 与 dispatch overlap 的标准写法，**正是 Plan B 要复制的骨架**（任务描述里的 "deepseek_v2.py:769" 在本机已挪到 **`deepseek_v2.py:1010-1015`**）。
- `eagle_worker_v2.py:664-666`、`multi_layer_eagle_worker_v2.py:625`：`batch.seq_lens.record_stream(current_stream())`——「张量在别的 stream 上分配，用 record_stream 防止 PyTorch 在 forward stream 还在跑时就回收复用」，注释直接点明 record_stream 的目的（见 Q2）。
- `cache_controller.py:478-548`：H2D/D2H 拷贝线程把 `host_indices` / `device_indices` 在 write/load stream 上 `record_stream`——**这是与 Plan B 最像的场景：跨 stream 的拷贝 + record_stream**。

### Q1.2 CUDA Graph capture 与 multi-stream 如何共存

这是本调研最重要的发现，**直接修正 investigation.md §6a**：

1. **graph 内多流是合法且仓内在用的**。`upstream utils.py:248` 的 `get_is_capture_mode()` 守门：只有在 capture 期间才走 alt-stream 分支。也就是说 fork/join（`wait_stream` + `with stream(alt)`）**被录进了 graph**，replay 时按录下的依赖关系重放——CUDA Graph 本身支持 capture 跨 stream 依赖（前提是用 event/wait_stream 显式建立依赖，且 fork 出去的 stream 最终 join 回 capture stream）。
2. **capture stream 的来源**：`graph_capture()` 上下文（`parallel_state.py:439-461`）建一条专用 capture stream，`stream.wait_stream(curr_stream)` 后 `with device_module.stream(stream)` 进入；`capture_one_batch_size` 用 `self.stream`（cuda_graph_runner.py:552）→ `_capture_graph` 里 `with graph_fn(cuda_graph=graph, pool=pool, stream=stream)`（:540）。alt stream 在 `apply_qk_norm` 里 fork 自这条 capture stream、再 join 回去。
3. **capture mode 是全局开关**：`cuda_graph_runner.py:93` `is_capture_mode=False`，`set_capture_mode()`（:102-107）在 capture 区间内置 True、退出置 False。`get_is_capture_mode()`（:96-97）被各模型查询以决定走单流还是多流分支。
4. **pdmux 多流 ≠ 单 graph 多流**：`cuda_graph_runner.py:520-525,506-507,854` 的 `f"{stream_idx}_{bs}"` 是「每条 decode pipeline stream 各存一份独立的单流 graph」，与 graph 内 fork alt stream 是两回事。Plan B 要的是后者（graph 内 fork），那有 `apply_qk_norm` 这个现成模板。

**适用场景**：Plan B 若要把 D2H/collect-pack 放进 graph 内的 alt stream，可照抄 `apply_qk_norm` 的 `get_is_capture_mode()` 守门 + fork/join 模板。但见 Q3——Plan A 的 D2H 本就在 graph **外**，把它移进 graph 内是另一种重写，未必划算；更可能的 B 形态是「alt stream 留在 graph 外」（Q3 选项 b）。

---

## Q2：`record_stream` 正确用法

### 读每个 record_stream 调用点

| 调用点 | 张量 | 在哪个 stream record | 为何需要 |
|---|---|---|---|
| `deepseek_v2.py:1014` | `shared_output`（在 alt_stream 上算出的结果） | `alt_stream` | 该张量在 alt 流分配，下游在主流消费；防止 caching allocator 在 alt 流的 kernel 还没跑完时就回收这块显存 |
| `eagle_worker_v2.py:664` | `batch.seq_lens`（在另一 stream 分配） | `current_stream()` | 注释：「allocated in another stream → record_stream() to prevent pytorch gc and reuse the gpu memory while forward_stream is still running」 |
| `cache_controller.py:478,480,546,548` | `host_indices` / `device_indices`（D2H/H2D 拷贝的 index 张量） | write/load stream | 拷贝在后台 stream 异步进行，主流可能已经返回并想释放这些 index；record_stream 把它们的生命周期挂到拷贝 stream 上 |
| `nsa_indexer.py:987,1008`、`deepseek_v2_attention_mla_npu.py:326` | `q` / `weights` | alt/indexer stream | 同理，跨流使用的中间张量 |

### 何时**必须**、何时**可省**、漏了会怎样

- **必须**：当一个张量 **A. 由 PyTorch caching allocator 分配（不是你自己 pin/预分配的固定 buffer）**，且 **B. 在 stream X 上发起异步操作（kernel 或拷贝）后，Python 侧最后一个引用在 stream Y（通常主流）上消失/被覆盖**。caching allocator 只追踪「张量在其**分配 stream**上的最后一次使用」，**不知道**它还被另一条 stream 用着。一旦 Python 引用归零，allocator 认为这块显存可在分配 stream 上立即复用——但 stream X 的异步 op 可能还没跑完 → **新分配复用同一块显存 → 数据被覆写 → 间歇性错码/NaN（与时序相关、极难复现）**。`record_stream(X)` 告诉 allocator「这块显存还在 stream X 上被用，等 X 上当前进度过了才能回收」。
- **可省**：
  1. 张量是你**自己预分配、生命周期由你持有的固定 buffer**（不交给 allocator 自动回收）——例如 Plan A 的 `_host_staging_buffers`（`base.py:80-89`，2 个 pin_memory buffer，挂在 `self._host_staging_buffers` 上长期持有）和模型的 `_cg_*` graph 输入 buffer。它们**不会被 allocator 在中途回收**，所以**不需要 record_stream**。
  2. 跨流依赖已用 `wait_stream`/`wait_event` join 回主流、且张量随后只在主流被引用——这种情况依赖已由 event 保证，record_stream 主要管的是「Python 引用消失后的回收时机」，若张量被长期持有则无忧。

### Plan B 若在 alt stream 上把 D2H 拷进 pinned host buffer，record_stream 需要吗？对哪个张量？

- **host pinned buffer（`_host_staging_buffers[slot]`）**：**不需要 record_stream**。它是长期持有的固定 buffer，不走 allocator 自动回收（Q2「可省」第 1 条）。它的 CPU-读 vs GPU-写竞争由 **ping-pong 双缓冲 + event** 处理（Plan A 已有，base.py:70-92），不是 record_stream 的职责。
- **device staging（`model._cg_collect_staging`）**：**不需要**。同样是模型持有的固定 buffer（pack 写它、D2H 读它，都在 `_decode_pack_gpu`，higgs model_runner.py:206-225）。
- **真正需要 record_stream 的情形**：**如果 B 把 `_decode_pack_gpu` 里的 pack（scatter `_cg_active_* → pool` + 打包进 staging）也挪到 alt stream**，而这些 op 读的源张量（`_cg_codes_BN` / `_cg_active_*`）是 graph 写入的固定 buffer——它们也是持有的，**仍不需要 record_stream**。**只有当 B 引入新的、由 allocator 临时分配的中间张量并在 alt 流上用它**（例如临时 gather 出一个新 tensor 再拷），那个临时张量才需要 `record_stream(alt_stream)`。

### record_stream 决策清单

```
对每个要在 alt stream 上参与 op 的张量 T：
  1) T 是你预分配 / 长期持有的固定 buffer（pinned host buf / _cg_* / staging）？
        → 不需要 record_stream（生命周期你管，allocator 不回收）
  2) T 是 caching allocator 临时分配（torch.empty / gather / clone 的结果）？
     且 在 alt stream 发起 op 后，Python 侧最后引用在主流消失？
        → 必须 T.record_stream(alt_stream)，紧跟那条 op 之后
  3) 跨流结果要回主流消费？
        → 用 wait_stream/wait_event join，且若结果张量随后会被释放，
          在 alt 流上对它 record_stream(主流) 或保持引用直到 join 后
  4) 拿不准？→ 加 record_stream 几乎无副作用（只是推迟回收），漏加才致命
  5) 验证：compute-sanitizer --tool synccheck / racecheck 跑短序列
```

---

## Q3：cuda_graph_runner 改 multi-stream 的工作量

### 现状是否假设单 stream？哪些行依赖该假设？

`upstream cuda_graph_runner.py` 的 capture/replay 以**一条 capture stream**为中心：
- `capture()`（:467）→ `_capture_one_stream`（:472）→ `capture_one_batch_size`（:547），其中 `stream = self.stream`（:552），`self.stream` 由 `graph_capture()` 的 context 赋值（:516 `self.stream = graph_capture_context.stream`）。
- `_capture_graph`（:530-542）：`with graph_fn(cuda_graph=graph, pool=pool, stream=stream): out = run_once_fn()`——整个 `run_once`（:691-716，即模型 forward）被录进这**一条** stream 的 graph。
- `replay`（:837-857）：`self.graphs[graph_key].replay()`，单条 graph replay，无 stream 参数。

**但「单 stream 假设」是软的**：`run_once` 里调用的模型 forward **内部**已经可以 fork alt stream（`apply_qk_norm` 的 `wait_stream`/`with stream(alt)`），只要在 `run_once` 结束前 join 回 capture stream。换言之，**runner 层不需要知道有 alt stream**——alt stream 的 fork/join 完全发生在被 capture 的 `forward` 内部。这就是为什么 Qwen3-Omni 现在能在 CG 模式下用 alt stream 而 cuda_graph_runner.py **一行没改**。

### 选项 (a)：capture 期间 fork alt stream（graph 内多流）

**改哪里**：不改 cuda_graph_runner.py 本身；改的是**被 capture 的那段 forward / post 逻辑**——但 Plan A/B 的 D2H/collect **当前不在 capture 区间内**（见下），所以要走 (a) 必须把 pack/D2H **移进** `forward_batch_generation` 能 capture 的范围，这是结构性重写。
- 若沿用 `apply_qk_norm` 模板：`get_is_capture_mode()` 守门 + `alt_stream.wait_stream(cur)` + `with stream(alt): <D2H>` + `cur.wait_stream(alt)`。runner LOC ≈ 0（runner 不变），但「把 D2H 塞进 graph」需要 D2H 的目标/源 buffer 全是 graph-capturable 的固定地址，且 **graph 内 D2H 到 pinned host 的语义**需谨慎（graph 重放时拷贝目标地址固定 → 仍要 ping-pong，否则每次 replay 写同一 host buffer 与 host 读竞争）。粗估 **20-40 LOC + 验证成本高**（investigation.md §6a 的估计在此成立，但前提是真要塞进 graph）。

### 选项 (b)：alt-stream 工作留在 captured graph **外**

**改哪里**：cuda_graph_runner.py **完全不改**。Plan A 的 D2H/pack 本来就在 graph 外（`post_decode_launch` 在 `_prepare_and_forward` → `forward_batch_generation` 返回**之后**调用，base.py:144-149；graph replay 在 `forward_batch_generation` 内部、早已结束）。B 只需在 `post_decode_launch` 里把那段 pack/D2H 从「当前 stream」改到「alt stream」，并用 event 串接：
```
# launch（graph 外）：
#   forward(graph replay 在主流) → record event_fwd
#   alt_stream.wait_event(event_fwd)             # alt 等 forward 完成
#   with stream(alt): pack + D2H 进 pinned buf → record event_d2h
# resolve：
#   event_d2h.query()/synchronize()（与 A 同），读 host buf
```
**改动面**：纯在 `higgs model_runner.py:post_decode_launch`（:53-81）+ `base.py:execute_launch`（:127-169）内，**runner 0 改、scheduler 0 改、状态机 0 改**。粗估 **15-30 LOC，集中在一两个函数**。

### (a) vs (b) 权衡表 + 推荐

| 维度 | (a) graph 内 fork alt stream | (b) alt stream 留 graph 外 |
|---|---|---|
| cuda_graph_runner.py 改动 | 不改 runner，但要把 D2H 重写进 capturable forward | **0 改** |
| 需改的代码 | 把 pack/D2H 结构性移进 capture 区间 + `get_is_capture_mode` 守门 | 仅 `post_decode_launch` + `execute_launch` 里加 alt stream + 2 个 event |
| LOC | 20-40 + 高验证成本（graph 内 D2H 到 pinned 的重放语义） | 15-30 |
| 复用 A 的成果 | 部分（D2H 要重写） | **几乎全部**（A 的切点天然就是 graph 外） |
| 与 A 收益靶点的关系 | 把 D2H 塞进 forward 内部并行——但 A 已经把 D2H 藏在 forward **后**，graph 内并行的边际收益更难量化 | 在 graph 外用 alt stream 解放主流，让主流更早入队下一步 forward |
| 风险 | graph capture/replay 的多流重放语义、pinned 目标地址重放竞争 | 标准跨流 event 串接，有 deepseek_v2 / cache_controller 模板 |
| 先例 | `apply_qk_norm`（仓内已验证，但那是 norm 不是 D2H） | `cache_controller.py` 的跨流拷贝 + record_stream（场景几乎同构） |

**推荐 (b)**。理由：(1) Plan A 已经把 D2H/collect 干净地放到 graph **外**（base.py:144-149），(b) 顺着这个切点走，runner 一行不改、状态机一行不改；(2) (b) 有 `cache_controller.py` 这个「跨流 D2H/H2D 拷贝 + record_stream」的同构先例可抄；(3) (a) 把 D2H 塞回 graph 内，与 A「不改 CUDA Graph」的设计初衷（design.md §5）相悖，且 graph 内 pinned 拷贝的重放语义要额外验证，性价比低。**只有当 profile 证明「主流上 D2H 本身（不是它后面的 CPU collect）显著阻塞下一步 forward 入队」时，(b) 才有收益**——而 A 已把 collect 藏住，D2H 本身只是一次 `cudaMemcpyAsync` 入队（不阻塞 CPU），所以 (b) 的收益大概率也很小（见 Q6 R-INF-2）。

---

## Q4：Plan B 最小可行 stream 拓扑

### 几条 stream，为什么

**2 条**：
1. **主 stream（compute/capture stream）**：跑 graph replay（forward + on-GPU 采样）。这是 sglang 的 current/capture stream，不能动。
2. **alt stream（copy stream）**：跑 `_decode_pack_gpu` 的 scatter+pack（可选）+ D2H 拷贝进 pinned host buffer。

为什么不是 3 条：H2D（下一步的 row indices / sampling buffer 上传，higgs model_runner.py:115-156）是下一步 launch 的 prepare 的一部分、且是非阻塞 H2D，A 已经异步入队；单独给它一条 stream 收益微乎其微且增加 event 管理复杂度。**最小可行 = 2 条**（主 compute + alt copy），与 deepseek_v2 / cache_controller 的做法一致。

### alt stream 上放什么

- **必放**：D2H 拷贝 `host_buf[:n_real].copy_(staging[:n_real], non_blocking=True)`（现 higgs model_runner.py:69，A 在主流上发）。
- **可选放**：`_decode_pack_gpu` 的 GPU→GPU pack（scatter pool + 打包 staging，higgs model_runner.py:206-225）。把它也挪到 alt 流，主流 forward 一结束就能立刻入队下一步——但这些 op 极小，收益边际。
- **不放**：H2D（留主流 prepare）、graph replay（必须主流）、CPU collect 循环（那是 host 侧，不在任何 stream 上，A 已用 ping-pong + event 藏在 forward 后）。

### wait_event 插入点 + ASCII 依赖图

```
                  主 stream (compute/capture)              alt stream (copy)
launch(N):
  prepare_decode (H2D row idx) ──┐
  graph.replay  forward(N)       │  (3.72ms, 主流)
  on-GPU sample → _cg_codes_BN   │
  event_fwd_N.record() ──────────┼───────────────►  alt.wait_event(event_fwd_N)   ← (1) alt 等 forward 完成
                                 │                  with stream(alt):
                                 │                     [可选] _decode_pack_gpu
  (主流可立即继续入队 launch(N+1))│                     host_buf.copy_(staging, non_blocking)
                                 │                  event_d2h_N.record() ──────┐  ← (2) D2H 完成的 event
                                 ▼                                            │
launch(N+1): forward(N+1) ...    （与 alt 的 D2H 并行）                        │
                                                                              ▼
resolve(N-1)  [host, 纯 CPU]:  event_d2h_{N-1}.query()/synchronize() ◄────────┘  ← (3) host 等上一步 D2H
              读 host_buf[N-1] → collect 循环（藏在 forward(N) 背后）
```

三个 wait/sync 点：
- **(1) `alt.wait_event(event_fwd_N)`**：alt stream 在 launch(N) 里、发 D2H 之前，等主流的 forward(N)+sample 完成（否则 D2H 拷到的是脏数据）。**这是 A→B 的核心升级**：A 用「主流 FIFO 顺序」隐式保证 D2H 排在 forward 后；B 用显式 `wait_event` 让 D2H 进 alt 流但仍在 forward 后。
- **(2) `event_d2h_N.record()`**：在 alt stream 上、D2H 拷贝之后 record。语义同 A 的 `event.record()`（base.py:159），只是现在 record 在 alt stream 而非主流。
- **(3) resolve 里 `event_d2h.query()/synchronize()`**：与 A 逐字相同（base.py:180-184），不用改。

### 现有 pinned 双缓冲在 multi-stream 下还够吗？

**够，且仍然必要——但需要补一个新的同步点。**
- **为什么仍必要**：ping-pong 解决的是「resolve(N-1) 在 host 读 buffer A，launch(N+1) 的 D2H 在 GPU 写 buffer——若同一 buffer 则 CPU 读/GPU 写竞争」（base.py:70-92 注释）。multi-stream 不改变这个竞争的存在，反而**更尖锐**：A 里 D2H 在主流、与 resolve 的 host 读靠「主流忙着 forward(N) 时 CPU 跑 collect」错开；B 里 D2H 在 alt 流、可能与主流 forward 完全并行，host 读和 GPU 写的时间窗更容易重叠。所以 **ping-pong 在 B 里只会更必要**。
- **为什么够（2 个就行）**：同一时刻最多 1 个 in-flight step（`_PendingStep` 不变式，base.py:29-33），resolve(N-1) 读 slot⊕1、launch(N) 写 slot，2 个 buffer 严格够用。**前提**：B 必须保证 launch(N) 的 D2H（写 buffer slot）**不早于** resolve(N-1) 读完 buffer slot⊕1——这在 A 里由「先 launch 后 resolve、且两者都在主流/CPU 顺序」隐含；在 B 里因为 D2H 进了 alt 流，要确认 launch(N) 写的 slot 与 resolve(N-1) 读的 slot⊕1 确实是不同 buffer（`_staging_slot ^= 1` 每次 launch 翻转，base.py:91，**已保证**）。**新增检查点**：alt 流的 D2H(N) 写 slot 与「下一次 resolve 即 resolve(N) 读 slot」之间隔了一次翻转，安全；但 alt 流的 D2H 可能比主流跑得慢，若某步 alt 流的 D2H(N-1) 还没写完、下一轮 launch(N+1) 就要写同一个 slot（N-1 和 N+1 同奇偶）→ **需要在写 slot 前 wait 上上步 D2H 完成**。A 里这个由主流 FIFO 隐含保证（D2H 全在主流，严格有序）；**B 里 D2H 进 alt 流后，slot 复用的有序性要显式用 event 保证**（见 Q6 R-SCOPE-2）。

### wait_event 插入点清单

```
[ ] launch: forward+sample 后 record event_fwd（主流）
[ ] launch: 发 D2H 前 alt.wait_event(event_fwd)            ← (1) 防脏读
[ ] launch: D2H 后在 alt 流 record event_d2h               ← (2) 替代 A 的主流 record
[ ] launch: 写 host_buf[slot] 前，确认上一个用该 slot 的 D2H 已完成
            （slot 复用有序性，A 由主流 FIFO 保证，B 需 event）  ← (3) Q6 R-SCOPE-2
[ ] resolve: event_d2h.query()/synchronize()（复用 A，base.py:180-184 不改）
[ ] 若 pack 也进 alt 流：scatter 写的 pool buffer 下一步 prepare 又要读，
    需 join 回主流（cur.wait_event(alt 的 pack event)）       ← (4) 防 prepare 读到半写的 pool
```

---

## Q5：Plan A → Plan B 接口契约表

| Plan A artifact | Plan B 原样复用？ | 必须改（怎么改）？ |
|---|---|---|
| **`_PendingStep` dataclass**（base.py:20-43） | **是，结构不变** | 仅**新增字段**：把单个 `event`（:36）拆成/补成 `event_fwd`（forward 完成）+ `event_d2h`（alt 流 D2H 完成）。resolve 仍只 query `event_d2h`。host_buf/n_real/各 batch 引用全不变。**纯加字段，不动语义。** |
| **pinned ping-pong host buffer**（`_host_staging_buffers` base.py:64,80-92；`_next_host_staging`） | **是，原样复用** | 不改。multi-stream 下双缓冲只会更必要（Q4）。**不需要 record_stream**（长期持有的固定 buffer，Q2）。唯一新增：slot 复用前确认上上步 alt-流 D2H 已完成（用 event，不改 buffer 本身）。 |
| **`execute_launch` / `execute_resolve` 拆分**（base.py:127-198） | **是，骨架原样** | `execute_launch`：把 `event.record()`（:159，主流）改成「record event_fwd → alt.wait_event(event_fwd) → with stream(alt): D2H → record event_d2h」。`post_decode_launch` 的 D2H 行（higgs:69）包进 `with torch.cuda.stream(alt)`。`execute_resolve`（:171-198）**一行不改**（query 的还是同一个 event 语义）。 |
| **`post_decode_launch` / `post_decode_resolve` hooks**（base.py:343-377；higgs:53-94） | **是，签名不变** | `post_decode_launch` 内部把 D2H（higgs:69）改到 alt 流；可选把 `_decode_pack_gpu`（higgs:206-225）也挪 alt 流并 join 回主流。**hook 签名、调用点、`post_decode_resolve` 全不改。** design.md §8 预言的「B 在 GPU 半/host 半之间插 alt-stream 段而不改 hook 签名」**成立**。 |
| **`event.record()` 放置**（base.py:156-159） | **否，需移位** | A：在主流、D2H 之后 record 一个 event（base.py:159）。B：需要**两个** event——`event_fwd`（主流、forward+sample 后）给 alt 流 wait；`event_d2h`（alt 流、D2H 后）给 resolve query。这是 A→B 改动最实质的一处，但仍是「同一函数内加几行」。 |
| **`_event_loop_async_decode` 主循环**（omni_scheduler.py:1014-1054） | **是，完全原样** | **0 改**。launch-first 顺序、`_async_pending` 配对、`_resolve_and_process` 的 overrun 丢弃（:965-1002）、drain/flush（`_resolve_pending_async` :1004-1012）、prefill 同步降级、abort 纳管（:826,836）全部与 stream 拓扑无关。B 的多流改动**全部封在 ModelRunner 内**，scheduler 看不见。 |

**结论：Plan A 确实为 B 铺好了路，不是返工。** 6 行里 4 行「原样复用」、2 行「同函数加几行」（`_PendingStep` 加字段 + `event.record` 拆成两个 event）。**没有任何一项需要推翻 A 的结构。** scheduler 层、状态机、ping-pong、hook 签名全部稳定；B 的全部增量集中在 `base.py:execute_launch` + `higgs model_runner.py:post_decode_launch` 两个函数（与 Q3 (b) 一致）。design.md §8「层级 A 要给 B 预留的 hook」4 条预言**逐条兑现**：(1) event/buffer 所有权清晰 ✔（在 `_PendingStep`，caller 持有）；(2) GPU 半/host 半切点天然 ✔（`post_decode_launch`/`resolve`）；(3) D2H buffer 已 pinned+ping-pong ✔；(4) flag 预留——**唯一未兑现**：`enable_async_decode` 当前是 bool（omni_scheduler.py:103,133；bootstrap.py 尚无 async_decode_level 子选项），design.md §8 设想的 `async_decode_level: "A"|"B"` 还没占位（见 Q6 R-SCOPE-3，小事）。

---

## Q6：风险登记

### 1. 可能让 Plan B **不可行**的风险（须先解决）

| # | 风险 | 一句话缓解 |
|---|---|---|
| R-INF-1 | graph 内多流 replay 语义：若选 Q3(a) 把 D2H 塞进 capture 区间，pinned 拷贝目标地址在 replay 时固定，与 host 读形成跨 replay 竞争 | 选 **Q3(b)**（alt stream 留 graph 外）直接绕开；A 的切点本就在 graph 外，天然支持 (b) |
| R-INF-2 | **B 相对 A 可能没有可测收益**：A 已把整段 CPU collect 藏进 forward(N) 背后（base.py 主循环 launch-first），D2H 本身只是一次 `cudaMemcpyAsync` 入队（不阻塞 CPU）；B 把这次入队挪到 alt 流，省的只是「主流上一条 memcpy 的入队/执行不挡下一步 forward 入队」——在 forward≫D2H 的 Higgs 场景近乎为零 | 上 GPU 用 `gpu-perf-ab` 方法论 A/B（A vs A+B），先量化「主流 D2H 是否真挡 forward 入队」；**若收益 <噪声则 B 不值得做**（这是 should-we-do-B 的前置，本调研不替它拍板） |

### 2. 会**扩大 Plan B 范围**的风险（可缓解但须规划）

| # | 风险 | 一句话缓解 |
|---|---|---|
| R-SCOPE-1 | 跨流临时张量漏 record_stream → 间歇错码 | 用 Q2 清单逐张量过；B 的 D2H 只碰固定 buffer（pinned host + `_cg_*`），**当前不引入 allocator 临时张量**，所以大概率无需 record_stream——但任何新增 gather/clone 要立刻补上 |
| R-SCOPE-2 | **slot 复用有序性**：D2H 进 alt 流后，D2H(N-1) 写 slot 与两步后 D2H(N+1) 写同一 slot 之间，A 靠主流 FIFO 隐含有序，B 失去该保证 | 在写 host_buf[slot] 前 `alt.wait_event(上一个用该 slot 的 D2H 的 event)`；或保留「launch 前确保前前步已 resolve」的不变式（resolve 读完即可复用），nuance 见 Q4 清单 (3) |
| R-SCOPE-3 | flag 仍是 bool，design.md §8 设想的 `async_decode_level: "A"\|"B"` 子选项没占位 | 加一个 `--async-decode-level`（默认 "A"），bootstrap.py:57 附近透传；小改动，B 落地时一并做 |
| R-SCOPE-4 | 若把 `_decode_pack_gpu` 的 scatter 也挪 alt 流，scatter 写 `pool`，下一步 prepare 又读 `pool`（higgs:151-156），跨流读写需 join | scatter 后 `cur.wait_event(alt pack event)` join 回主流（Q4 清单 (4)）；或干脆 pack 留主流、只挪 D2H（最小 B） |

### 3. 调试时会踩的坑（提前知道）

| # | 坑 | 一句话缓解 |
|---|---|---|
| R-DBG-1 | 漏 `alt.wait_event(event_fwd)` → D2H 拷到 forward 还没写完的脏 staging，**greedy 同 seed 下表现为偶发错码**（与 forward/D2H 时序赛跑，不稳定复现） | 硬门槛回归：async-B vs sync 逐 token 相等（design.md §7）；compute-sanitizer synccheck/racecheck 跑短序列 |
| R-DBG-2 | event 在错误的 stream 上 record（在主流 record 了 D2H 的 event 而 D2H 在 alt 流）→ query 永远命中但数据没到 | 严格遵守「event 必须在它要标记的那条 stream 上 record」：`event_d2h` 在 `with stream(alt)` 内 record |
| R-DBG-3 | `event.record()` 不带 stream 参数默认 record 在 current stream——在 `with torch.cuda.stream(alt)` 块内/外位置错会 record 到错流 | 显式 `event_d2h.record(alt_stream)` 或确保在 `with stream(alt)` 块内调用，code review 检查点 |
| R-DBG-4 | nsys timeline 误读：A 已经把 collect 藏在 forward 后，B 的 timeline 看起来「几乎一样」，容易误判 B「没生效」 | 看的不是「collect 是否藏住」（A 已藏），而是「主流上是否还有一段 memcpy 占用、B 后是否消失」——量的是 R-INF-2 那个微小差值 |
| R-DBG-5 | warmup 首步：A 里首步 `event.query()` 可能 miss 走 synchronize（base.py:180-184），B 引入 alt 流后首步 alt 流可能还没「就绪」，event 语义需确认 | 首步沿用 A 的 synchronize fallback（在层级 B 边界内允许），单测 mock event.query→False 已覆盖该分支（test_async_decode.py:116-122） |

---

## 附：A→B 改动落点速查（若 B 实施，按 Q3(b) 最小形态）

| 文件:函数 | A 现状 | B 增量 |
|---|---|---|
| `base.py:execute_launch`（:127-169） | `event=torch.cuda.Event(); event.record()`（:156-159，主流） | record `event_fwd`（主流）→ `alt.wait_event(event_fwd)` → `with stream(alt): post_decode_launch 的 D2H` → record `event_d2h`（alt 流）；`_PendingStep` 存 `event_d2h` |
| `base.py:_PendingStep`（:20-43） | 单 `event` 字段 | 加 `event_fwd` / `event_d2h`（或复用 event 表示 d2h + 新增 fwd） |
| `base.py:execute_resolve`（:171-198） | `pending.event.query()`（:180） | **不改**（query `event_d2h`） |
| `higgs model_runner.py:post_decode_launch`（:53-81） | `host_buf.copy_(staging, non_blocking=True)`（:69，主流） | 把 :69 包进 `with torch.cuda.stream(alt)`；alt 来自 ModelRunner 新建的 `self._copy_stream = torch.cuda.Stream()`（仅 async-B 模式建） |
| `omni_scheduler.py:_event_loop_async_decode`（:1014-1054） | launch-first 主循环 | **不改** |

参考模板（抄这两处）：`deepseek_v2.py:1010-1015`（fork/join + record_stream + record_event）、`cache_controller.py:478-548`（跨流 D2H/H2D 拷贝场景，与 B 最同构）。
