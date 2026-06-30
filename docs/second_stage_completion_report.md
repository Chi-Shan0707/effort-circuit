# Second-Stage Completion Report

日期：2026-06-30 UTC  
目标来源：`second_prompt.md`  
核心宗旨：一切从第一性原理出发。  
完成范围：Step 1–7，每个阶段均已独立 commit + push。

## 0. 结论先行

本轮路线已经完成。最重要的结果不是证明了一个漂亮的 positive claim，而是把原始强线索收紧成一个更可靠的机制故事：

> Qwen3-0.6B 的 late decoder-block states 确实能因果改变 generation posture；但该 posture 目前不是一个已验证的 useful reasoning controller。严格首答、task-out controls、data-driven metrics、cross-question patching 和 closed-loop CET 都显示：现有信号混合了 reasoning、format、readout、answer-token suppression 与 prompt artifact。

当前最可信的最终结论：

1. **成立**：late-layer hook states 能控制 continue/final posture。
2. **成立**：mean process direction 和 cross-question patch 能转移 posture。
3. **成立**：旧 layer27 hidden-state patch 存在 site-mismatch 风险。
4. **不成立**：现有 intervention 稳定提高 first committed answer accuracy。
5. **不成立**：手工或 naive data-driven token metric 足以代表 useful reasoning。
6. **不成立**：当前 CET controller 已可用。

## 1. Requirement Audit

| Requirement from `second_prompt.md` | Evidence | Status |
| --- | --- | --- |
| Activation-site audit | `src/site_audit.py`, `docs/step_01_site_audit.md`, `outputs/site_audit_n2_layers18_27.*` | Completed |
| Stop-after-first-final generation | `src/stop_after_final_experiment.py`, `docs/step_02_stop_after_final.md`, `outputs/stop_after_final_heldout_n5_hook_states.*` | Completed |
| Expanded heldout statistics | `src/paired_stats.py`, `docs/step_03_expanded_statistics.md`, `outputs/stop_after_final_stats_n5.*` | Completed |
| Task-out and verbosity controls | `src/task_out_controls.py`, `docs/step_04_task_out_controls.md`, `outputs/task_out_controls_n2_layers18_27.*` | Completed |
| Data-driven token metric | `src/data_driven_token_metric.py`, `docs/step_05_data_driven_metric.md`, `outputs/data_driven_metric_train8_heldout5.*` | Completed |
| Process/task disentanglement | `src/process_task_disentanglement.py`, `docs/step_06_process_task_disentanglement.md`, `outputs/process_task_disentanglement_n5_layers18_27.*` | Completed |
| Closed-loop CET controller | `src/closed_loop_cet.py`, `docs/step_07_closed_loop_cet.md`, `outputs/closed_loop_cet_n3_layer18.*` | Completed |
| Every step commit + push | Git history from `1833891` through `3f64028` | Completed |

## 2. Step-by-Step Findings

### Step 1: Activation-site audit

Key result:

- layer 18: hidden-state-to-hook patch is site-equivalent.
- layer 27: hidden-state-to-hook patch is not site-equivalent.

Interpretation:

> Old layer27 claims must be interpreted as site-mismatch interventions unless rerun with hook-captured states.

### Step 2: Stop-after-first-final

Heldout n=5 strict first-final result:

| condition | first-final acc | last-answer acc | strict stop rate |
| --- | ---: | ---: | ---: |
| baseline | 0.000 | 0.400 | 0.200 |
| layer18_t1 | 0.000 | 0.000 | 0.600 |
| layer27_t0.75 | 0.000 | 0.600 | 0.200 |
| layer27_t1 | 0.000 | 0.200 | 0.200 |

Interpretation:

> The previous heldout-small accuracy gain does not survive first-commit evaluation.

### Step 3: Paired statistics

First-final metric:

- all conditions `0/5`;
- Wilson CI for zero successes remains wide, `[0.000, 0.434]`;
- paired diff vs baseline is exactly zero.

Interpretation:

> n=5 cannot estimate final effect size, but it proves there is no observed first-final gain in this run.

### Step 4: Task-out and verbosity controls

Key result:

- reasoning/high_reasoning layer27 delta gap: `+0.230902`;
- reasoning/high_verbosity layer27 delta gap: `+0.095382`;
- non_reasoning/high_format layer27 delta gap: `+0.057764`.

Interpretation:

> The signal is not reducible to verbosity alone, but it is also not cleanly reasoning-specific. Format and task family matter.

### Step 5: Data-driven token metric

Learned clusters:

- high/continue cluster includes odd tokens like `Diagram`, `Trace`, `Circ`, `Xiao`;
- low/final cluster includes numeric answer tokens like `3`, `2`, `1`.

Interpretation:

> Naive high-vs-low token contrast mostly exposes prompt artifacts and next-answer-token bias. It does not produce a clean useful-reasoning metric.

### Step 6: Process/task disentanglement

Key result:

| condition | layer | gap | target rank | source beats target |
| --- | ---: | ---: | ---: | ---: |
| low | 27 | 0.000099 | 7.4 | 0.200 |
| cross_question_high_replace | 27 | 0.284528 | 681.4 | 0.200 |
| mean_process_direction_add | 27 | 0.258192 | 440.8 | 0.200 |

Interpretation:

> Cross-question and mean-direction interventions transfer posture without obvious source-answer leakage, but they badly suppress target answer availability.

### Step 7: Closed-loop CET

Key result:

| controller | accuracy | strict stop rate | mean alpha |
| --- | ---: | ---: | ---: |
| baseline | 0.000 | 0.000 | 0.000 |
| static_mean_direction | 0.000 | 0.000 | 1.000 |
| closed_loop_cet | 0.000 | 0.000 | 0.275 |

Interpretation:

> Closed-loop infrastructure exists and reduces alpha, but the current control vector causes degenerate generation and does not solve the task.

## 3. Final Scientific State

### Settled Evidence

- Late hook-state interventions can causally alter next-token posture.
- Layer 27 old hidden-state patching is not site-clean.
- First-final evaluation is stricter and invalidates previous optimistic last-answer claims.
- Prompt/task controls are necessary; without them, posture metrics are misleading.
- Mean process directions are not automatically safe controllers.

### Weak Evidence

- Layer18/27 still look like real posture sites.
- Mean direction may reduce source-answer leakage relative to single-source replacement.
- Closed-loop control is structurally feasible on CPU with KV cache.

### Contradictions / Negative Results

- No first-final accuracy gain was observed.
- Data-driven token clusters are not semantically clean.
- Cross-question posture transfer damages target answer rank.
- CET controller degrades text under current signal.

## 4. Engineering State

Added scripts:

- `src/site_audit.py`
- `src/stop_after_final_experiment.py`
- `src/paired_stats.py`
- `src/task_out_controls.py`
- `src/data_driven_token_metric.py`
- `src/process_task_disentanglement.py`
- `src/closed_loop_cet.py`

Added tests:

- `tests/test_stop_after_final_experiment.py`
- `tests/test_paired_stats.py`
- `tests/test_task_out_controls.py`
- `tests/test_data_driven_token_metric.py`
- `tests/test_process_task_disentanglement.py`
- `tests/test_closed_loop_cet.py`

Final local verification:

```bash
python -m pytest -q
```

Result: `24 passed`.

## 5. Recommended Next Research Direction

Do not continue increasing alpha on the current vector. The right next project is to build a safer controller signal:

1. Train a hidden-state probe for process posture using behavioral labels, not raw token clusters.
2. Add an answer-readiness verifier to prevent target answer rank collapse.
3. Separate format-following from reasoning by factorial prompts.
4. Use controller actions that are smaller and decay over time.
5. Evaluate first-final correctness only, with last-answer accuracy marked as auxiliary.

## 6. Final Claim

The completed second-stage route transforms the project from a fragile positive result into a stronger mechanistic audit:

> Effort-circuit has a real late-layer posture control signal, but the signal is not yet a reliable useful-reasoning intervention. The project is now correctly positioned for a third stage focused on validated probes, answer-readiness preservation, and safer closed-loop control.
