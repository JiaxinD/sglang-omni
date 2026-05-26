# R1 幂等性验证（Phase 3 前置门槛）

**结论：(a) 幂等安全 → 按 design.md §1.1 不变式 3 实施，无需把 done rows 当 padding 特判，无 scope 变更。**

## 问题

one-step lookahead 下，请求在 forward 第 S 步被判 `generation_done` 后，会被多包含进第 S+1 步的 batch（一次「浪费」的 forward）。需确认：对已 done 的行再跑一次 step，是否会改动它的状态、或污染同 batch 的 active 行。

## 静态分析（`batched_step_direct`，sampler.py:295-366）

状态机用 `active = ~generation_done`（:330）显式屏蔽 done 行，逐项核对一个 `generation_done=True` 的行：

| 输出 | 表达式 | done 行结果 |
|---|---|---|
| `new_delay_count` | `where(in_delay_active, delay+1, delay)`（:337），`in_delay_active = active & ...` | `active=False` → **= delay（不变）** |
| `new_eoc_countdown` | `where(cb0_eoc_now_active, ..., where(in_winddown_active, eoc-1, eoc))`（:340-344） | 两条件皆 `active&...=False` → **= eoc（不变）** |
| `new_generation_done` | `generation_done \| done_this_step`（:353），`done_this_step` 含 `active` | `True \| False` → **= True（保持 done）** |
| `new_last_codes` | `where(update_codes, codes, last)`（:356），`update_codes = active & ~done_this_step` | `active=False` → **= last_codes（不变）** ← 关键，`_cg_active_last_codes` 不被改 |
| `out_codes` | `where(generation_done, STOP_CODE, codes)`（:359） | **= STOP_CODE (-1)** |

模型包装 `decode_codebooks_batch_cg`（model.py:320-373）只是 persist 上述返回值（:358-366），不额外改状态；并在 :340 设 `_cg_was_done = 输入的 generation_done`，done 行 = True → collect 循环 `if was_done_cpu[b]: continue`（model_runner.py:168）跳过其输出。

**跨行污染**：decode forward 每序列只 attend 自己的 KV（无跨行算子）；CG 固定 padded batch size，done 行 vs padding 行都是 inactive，active 行 logits 不受影响。采样在 padded batch 上整体跑（shape 恒定），active 行的采样 slot/RNG 不被 done 行扰动。

## 实证测试（`jiaxin-tools/r1_idempotency_test.py`，CPU，纯函数）

```
T1 PASS: done row fully frozen, out_codes == STOP_CODE
T2 PASS: repeated step on done row is idempotent
T3 PASS: greedy active-row output invariant to appended done row
T4 PASS: greedy deterministic across 20 seeds
ALL R1 CHECKS PASS -> conclusion (a) idempotent-safe
```

- **T1**：done 行 `delay_count/eoc_countdown/generation_done/last_codes` 全部不变，`out_codes=STOP_CODE`。
- **T2**：把 T1 输出状态再喂一次 step，状态完全一致（重复应用幂等）。
- **T3**（verify_correctness 硬门槛的地基）：greedy 下，active 行的采样 codes 在「单独」vs「附带一个 done 行」两种 batch 下**逐 token 相等**——即 sync（done 行已 drop）与 async（done 行多留一步）对 active 行输出一致。
- **T4**：greedy 跨 20 个随机种子确定性一致（argmax-like），佐证 bit-identical 不依赖 RNG。

## 对实施的影响

- 不变式 3 成立，**无需**改 `_populate_cg_buffers` 把 done rows 当 padding。
- verify_correctness 用 **greedy** 是对的：done 行多留一步不会扰动 active 行输出（T3/T4）。stochastic 采样在 stop 附近可能因 batch 组成差异而非 bit-identical，属预期（采样本就非确定），不作为门槛。
