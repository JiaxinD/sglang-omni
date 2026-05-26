# Phase 3 澄清：Q1（滞后步数）/ Q2（_FAILED_BATCH_RESULT）

## Q1：launch-first 循环的实际滞后 = 1 个浪费 step（不是 2）

你担心 §1.7 里 `get_next_batch_to_run()` 在 resolve 之前调用，flags 是 N-2 的 → 滞后 2 步。**flag 陈旧度确实是 2，但被丢弃的浪费 forward 只有 1 个**——因为「判定 done 的那一步 S 本身产出有效输出，不算浪费」。

### 精确推演（launch-first，请求在 forward 第 S 步被判 done）

循环：`(1) compose batch_i` → `(2) launch batch_i` → `(3) resolve+process batch_{i-1}`。
- step (3) 处理 `batch_{i-1}`；compose（step 1）用的 flags 来自上一轮 step (3) = process(`batch_{i-2}`) → **batch_i 知道截至 forward_{i-2} 的 stop**。
- 请求在 batch_i 中当且仅当 `i-2 < S`，即 `i ≤ S+1`：
  - `batch_S`：收到 **forward_S**（判 done 的那步，输出有效，**不浪费**）
  - `batch_{S+1}`：收到 **forward_{S+1}**（**浪费 1 步**，输出经 `_cg_was_done` 丢弃，R1 已验证幂等安全）
  - `batch_{S+2}`：剔除
- 对比同步基线：flags 截至 forward_{i-1}，请求在 batch 中当 `i ≤ S`，0 浪费。**async 比 sync 多 1 个浪费 step。**

→ **不变式 3 不改**：仍是「末尾最多多算一个被丢弃的 step」。陈旧度 2 ⇒ 浪费 = 陈旧度 − 1 = 1（finishing step 不算浪费）。

### 你提的「resolve → get_next_batch → launch」重排的利弊

该顺序下 batch_i 用截至 forward_{i-1} 的 flags（0 浪费 step），**但 resolve(N-1) 跑在 launch(N) 之前 → 此刻 GPU 空闲（forward_{i-1} 已完成、forward_i 未入队），collect 循环白占 CPU、不与任何 GPU 工作重叠 → 退化成同步、零收益**。
→ **结论：不重排。** 单个 in-flight step 下，「完整 overlap」与「<1 浪费 step」不可兼得；1 个浪费 step 是 lookahead overlap 的固有、不可约代价（上游 FutureMap 同理）。当前 launch-first 已是「1 浪费 step + 完整 overlap」的最优点，正是你想要的 1 步滞后。

### KV pool 容量
浪费的 forward_{S+1} 给 done 请求多 append 1 个 token 的 KV，retire 推迟到 iter S+2 → **KV 释放晚 1 个 decode step**（不是 2）。最坏情况（同一步全 batch 同时 done）瞬时多占 `batch_size` 个 token-row，对 TTS 长序列可忽略。实施时确认 pool 非「恰好卡满」无 1-step headroom（几乎必然有）；列入 verify 观察项。

## Q2：`_FAILED_BATCH_RESULT` 是 SGLang-omni 现有定义，非新引入

- 定义：`omni_scheduler.py:36` — `_FAILED_BATCH_RESULT = object()`（哨兵单例）。
- 产生：`_run_batch`（:584）捕获异常 → `_handle_batch_failure`（abort 相关请求、向 outbox 发 error）→ `return _FAILED_BATCH_RESULT`（:582）。
- 消费：`_event_loop_normal`（:824）和 `_event_loop_overlap`（:858）检查 `if result is not _FAILED_BATCH_RESULT:` 后才 `process_batch_result`。
- **async 循环沿用同语义**：`_run_batch_launch`/`_run_batch_resolve` 出错时返回该哨兵，`_event_loop_async_decode` 同样跳过 `process_batch_result`。无新失败语义。
