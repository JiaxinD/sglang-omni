# Master Summary — Async Decode PR 收尾（T1/T2/T3/T4）

分支 `feat/async-decode-lookahead`。本轮在已有 Plan A（single-stream + one-step
lookahead）基础上：修好 bs>1、修好采样器确定性、用 nsys 量化决定 Plan B 值不值得、
并预研 Plan B 架构。**结论先行：bs>1 不再 crash 且 async ON 在 bs=4 实测快
~13–20%（吞吐 ~1.2×）；Plan B 不值得做；建议把本 PR 作为「bs>1 吞吐优化 + 正确性/
确定性修复」合入。**

新增 3 个 commit（叠在原 10 个之上，无 coauthor trailer）：
`fbeb5b0`（T1）、`12dbdfb`（T4）、`9b70ca9`（验证/profile 工具）。

---

## T1 + T4 修复（bs>1 + sampler greedy）

### T1 — bs>1 replay size mismatch（commit `fbeb5b0`）

**选了哪个方案：三选一都不是，根因与原假设不同。** 用户给的 (a)/(b)/(c) 都假设是
「mid-batch finish 后 batch tensor 没 trim 干净」。GPU 复现 + 逐张量 shape trace 后，
真正根因是：`_finalize()` 在 **resolve** 阶段重写 `schedule_batch.output_ids`，而
launch-first 下 resolve 落后一步、且 `pending.schedule_batch` 就是**活的
running batch**——当前 launch 已按正确长度发布过 output_ids，resolve 又用上一步
(N-1) 的 next_token_ids 盖回去，留下一个**长度过期**的 output_ids。等某个请求
mid-batch finish、`filter_batch` 那一步又恰好没 drop 任何人（提前 return、不碰
output_ids），过期的 output_ids 就被下一次 `prepare_for_decode` 当成 input_ids，与
seq_lens 长度不符 → `replay_prepare` 的 `input_buffers.py` copy_ 报 "tensor a (2) vs
b (3)"。

`filter_batch` 本身 trim 是对的，所以 (a) 没必要、(b) 会把 bug 藏起来、(c) 是更大
改造。**最小正确修法**：`execute_resolve` 调 `_finalize(..., set_output_ids=False)`，
让 launch 成为 output_ids 的唯一发布者（同步路径仍默认 True，零行为变化）。3 行。

bit-identical 不受影响：bs=1 下 output_ids 长度恒为 1（bug 从不触发）；且 Higgs
decode 的 input_ids 对带 code 的行被 `_decode_step_embeds_cg` 屏蔽（用 GPU 常驻的
`_cg_active_last_codes`），不影响生成。

**验证**：
- `verify_correctness.py` **bs=4，10 prompt × 10 run = 100/100 bit-identical** OFF vs ON。
- `verify_correctness.py` **bs=1，10×10 = 100/100**（无回归）。
- bs=4/bs=8 错落并发 + EOC/length 混合压力测试，0 crash。
- 回归单测：`test_async_decode.py` 断言 resolve 必须 `set_output_ids=False`。

### T4 — batched sampler greedy short-circuit（commit `12dbdfb`）

`_sample_independent_batched` 一直走 multinomial，temp=0 时 near-one-hot 仍被
multinomial 随机破平、run-to-run 不确定（per-row 的 `_sample_independent` 有
argmax 短路，它没有）。修法：加 per-row greedy 掩码（`temperature <=
阈值` 或 `top_k == 1`）→ argmax over raw logits，**无分支**（两路都算 + torch.where，
因为它在 CUDA Graph capture 内、不允许数据依赖的 host 控制流）。

**验证**：3 个新单测（temp=0 跑 100 次确定性 = argmax；top_k=1 = argmax；混合行
逐行）。production greedy 现已确定性，**verify_correctness 不再依赖 argmax 注入**
（`verify_inject` 只保留逐步 code dump），上面 bs=1/bs=4 的 100/100 就是在**无注入
**下用真实生产采样器跑出来的。

### 现有单测
`38 passed`（原 35 + T4 的 3 个 greedy 测 + T1 回归断言）。
`tests/unit_test/pipeline/test_ipc.py` 的 7 个 PRE-EXISTING 失败与本work无关（在
base commit 上同样失败）。

更新的文档：`phase3_summary.md`（#1、#3 标为 FIXED + 新 commit 表），
`benchmark_results.md`（补 bs=4 ON 数字 + bs>1 增益分析）。

---

## T2 — stall analysis 结论（`stall_analysis.md`）

**TL;DR = (b)：没有 Plan B（multi-stream）能救的 stall。** 6 个配置全跑
（bs1/4/16/32 × olen256，bs4 × olen64/1024，async ON，nsys decode-isolated）。

关键数字（每配置高度一致）：
- decode 期间 **GPU busy ≈ 1.3%**，每步 **~3.8ms 串行 GPU 空闲**（每步一个 gap，
  ≥0.5ms）。
- 行为对 batch size **几乎不变**（bs1 busy 1.45% ↔ bs32 1.28%）——CUDA Graph padding
  使 forward 总跑同一 padded graph，GPU 真实计算仅 ~50µs/步。
- 那 3.8ms 是**下一步**的 `get_next_batch_to_run` + `prepare_for_decode` CPU 工作，
  发生在下一个 forward 入队**之前**，任何 stream 重叠都碰不到；而 D2H/collect 已被
  Plan A 藏住（`query_hit=100%`）。Plan B 的靶点（主流 D2H 入队）在 Higgs ≈0 收益。

**对 PR frame 的 implication = reframe（往上调）**：之前 frame 是「bs=1
latency-neutral，可能只是结构 cleanup」。新数据把它升级为**「bs>1 真实吞吐优化」**：
nsys 解释了为什么——launch-first 藏住的是「上一步 collect」，collect 随 bs 变大，
所以 bs=4 实测 **+13–20% 延迟 / 1.13–1.25× 吞吐**（两次跑确认，`query_hit=100%`），
bs=1 neutral。两个测量互相印证（可重叠的 collect 随 bs 增长 = 增益；不可重叠的
next-step prepare = 残留 3.8ms 空闲）。

副产物：decode 是彻底 CPU/dispatch-bound（GPU 98% 空闲），TTS decode 最大的单一杠杆
是**砍每步 CPU 路径**（向量化/下沉 `_populate_cg_buffers`、扩大 CUDA Graph capture
覆盖、或接上游 overlap 调度），不是 stream 拓扑。

---

## T3 — Plan B 架构准备结论（`plan_b_prep.md`）

- **可行性**：Plan B 技术上**可行，且 Plan A 是干净前置、不需推倒**。仓内已有
  「CUDA Graph capture 内 fork/join 第二条 stream」的成熟先例（Qwen3-Omni talker /
  Qwen3-TTS 在用），推翻了 investigation.md §6a 的悲观假设。
- **改动量**：按最小形态（Q3 方案 b，alt-stream 工作留 graph 外）约 **15–30 LOC**，
  集中在 `base.py:execute_launch` + `higgs post_decode_launch` 两处；scheduler 主循环、
  状态机、ping-pong buffer、hook 签名 **0 改**（接口契约表 6 项里 4 项原样复用、2 项
  同函数加几行）。
- **最大风险 = R-INF-2**：B 相对 A 可能**没有可测收益**——这正是 T2 用 nsys 证实
  的：A 已藏住 collect，B 省的那点主流 memcpy 入队在 98% 空闲、forward 仅 50µs 的
  时间线上≈0。

**T3 自己的推荐**：技术铺路 OK，但「该不该做 B」留给 stall_analysis 拍板——现在
T2 已拍板：**不做**。

---

## 给你的建议

**推荐：选项 A 的修正版 —— 直接合入当前 PR，但 frame 升级为「bs>1 吞吐优化 +
正确性/确定性修复」，不是单纯 cleanup；不启动 Plan B；把那 3.8ms CPU 每步气泡作为
独立的后续优化（≈选项 C）。**

理由：
1. **不是 cleanup 而已**：bs=4 实测 +13–20% 延迟、~1.2× 吞吐（两次跑确认，机制由
   nsys 解释清楚），这是吞吐相关 regime 的真实收益。bs=1 neutral 是预期内（collect
   太小）。
2. **不做 Plan B**：T2(b) + T3 R-INF-2 一致——B 的靶点在 Higgs ≈0。投 15–30 LOC +
   多流调试风险（record_stream/wait_event/slot 有序性）换≈0 收益，不划算。
3. **真正的下一步（选项 C）**：GPU 98% 空闲的根因是每步 ~3.8ms CPU 路径。下一个值得
   投入的方向是砍这段 CPU（向量化 `_populate_cg_buffers`/采样参数提取/collect 循环、
   扩大 CUDA Graph capture、或评估接上游 FutureMap overlap 调度），潜在收益远大于
   Plan B。这是另一个独立 work item，不该塞进本 PR。

不推荐选项 B（启动 Plan B 实现）。

### 建议的 PR description 草稿（你改后再用）

> **Async decode (one-step lookahead) for the omni AR loop — bs>1 throughput win**
>
> Splits the omni `ModelRunner.execute()` into launch/resolve so step N-1's host
> collect overlaps step N's GPU forward (single stream + CUDA `event.query()`,
> launch-first). Off by default (`--enable-async-decode` /
> `SGLANG_OMNI_ENABLE_ASYNC_DECODE=1`).
>
> - **Correctness**: output_codes bit-identical OFF vs ON — bs=1 and bs=4, 100/100
>   each (greedy). bs=1 already shipped; this PR fixes the bs>1 path (a launch-first
>   `output_ids` republish bug) so concurrent requests work.
> - **Perf**: bs=1 neutral; **bs=4 ~13–20% lower latency, ~1.2× throughput**
>   (query_hit=100%). The overlapped per-step collect scales with batch size.
> - Also makes the batched Higgs sampler short-circuit greedy → `temperature=0` is
>   now reproducible (was multinomial-nondeterministic).
> - A decode-isolated nsys profile (stall_analysis.md) shows the residual ~3.8 ms
>   per-step GPU idle is CPU scheduler/prepare work, not D2H — so a multi-stream
>   follow-up ("Plan B") would not add to this on Higgs; the next lever is the CPU
>   per-step path.

---

## 结束前自检 checklist

- [x] T1 修复后，verify_correctness.py 在 **bs=4** 下 100/100 bit-identical
- [x] T1 修复后，bs=1 仍然 100/100 bit-identical（无回归）
- [x] T4 后删除了 verify_inject 的 argmax patch，verify 仍 PASS（无注入跑出的 100/100）
- [x] T4 新增 deterministic-greedy 单测（3 个）
- [x] 38 个 unit test 全过（`38 passed`；7 个 test_ipc 失败为 PRE-EXISTING，与本work无关）
- [x] T2 跑完全部 6 个配置（数据见 stall_analysis.md / `/tmp/nsys/stall_stats.json`）
- [x] stall_analysis.md TL;DR 明确二选一 = **(b)**
- [x] T3 的 6 个问题都回答（plan_b_prep.md，无「待补充」）
- [x] phase3_summary.md 的 #1、#3 已更新为 FIXED
- [x] benchmark_results.md 补了 bs=4 ON 数字 + bs>1 增益
- [x] 无 scope 外「顺手优化」改生产代码（生产改动仅 base.py:T1 + sampler.py:T4；
      profile 的 mem-cap 补丁只在 `scripts/profile_inject` 内、PYTHONPATH-only）
- [x] 末尾「给你的建议」明确推荐了选项（A 修正版）

⚠️ 需诚实标注的两点（非 unchecked，是测量精度声明）：
1. **nsys 给 CPU 加了 ~20% 开销**（非 nsys bs=4 ~4.3ms/步 vs nsys ~5.15ms/步）；3.8ms
   gap 含此膨胀，但打折后 GPU 仍 >95% 空闲、瓶颈仍在 CPU 侧，定性结论稳健。
2. **benchmark 每配置仅 2 次跑**，production 采样使输出长度浮动（bs=4 OFF 跨度
   888–976ms）；bs=4 增益方向/量级两次一致且有机制解释，但属 ballpark 而非紧致测量。
3. **nsys 窗口只覆盖典型 TTS 的中小 KV**（decode 第 10–40 步；prompt ~100 步内 EOC，
   到不了超长上下文）；超长序列 regime 未刻画。
