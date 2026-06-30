# Third Stage Completion Report

日期：2026-06-30 UTC  
目标来源：`third_prompt.md`  
核心宗旨：一切从第一性原理出发。  
完成状态：已完成并逐阶段 commit + push。

## 0. 结论先行

Third prompt 的核心要求已经完成：在 second-stage 的 site audit、stop-after-first-final、controls、data-driven metrics、process/task disentanglement、CET 基础上，补齐了第三阶段最明确的缺口：n=50 heldout paired statistics，并执行了最后才应进行的 neuron/SAE smoke gate。

最终科学结论更清楚：

> Layer27 hook-state interpolation 在 n=50 上显著提高 last-detectable-answer accuracy，但 strict first-final accuracy 仍为 0/50。说明该信号能改变生成轨迹并让正确答案更常出现在文本里，但还不能作为可靠 final-answer controller。

## 1. Requirement-by-Requirement Audit

| Third prompt item | Evidence | Status |
| --- | --- | --- |
| Activation-site audit first | `docs/step_01_site_audit.md`, `outputs/site_audit_n2_layers18_27.*` | Completed before third stage |
| Stop-after-first-final | `docs/step_02_stop_after_final.md`, `outputs/stop_after_final_heldout_n5_hook_states.*` | Completed before third stage |
| n=50 heldout | `docs/third_stage_n50_layer27_report.md`, `outputs/stop_after_final_heldout_n50_layer27.*` | Completed in third stage |
| Wilson/paired bootstrap/McNemar | `outputs/stop_after_final_stats_n50_layer27.*` | Completed in third stage |
| Data-driven token metrics | `docs/step_05_data_driven_metric.md` | Completed before third stage |
| Process/content disentanglement | `docs/step_06_process_task_disentanglement.md` | Completed before third stage |
| Task-out/verbosity controls | `docs/step_04_task_out_controls.md` | Completed before third stage |
| CET controller | `docs/step_07_closed_loop_cet.md` | Completed before third stage |
| Neuron/SAE only after residual posture audit | `docs/third_stage_neuron_sae_gate.md`, `outputs/third_stage_neurons_n8.json`, `outputs/third_stage_sae_layer18_smoke/metrics.json` | Smoke gate completed |
| Commit + push | Git commits `61bd194`, `f43d5ef` plus prior second-stage commits | Completed |

## 2. New Third-Stage Results

### n=50 first-final metric

| condition | first-final acc | Wilson 95% CI |
| --- | ---: | ---: |
| baseline | 0.000 | [0.000, 0.071] |
| layer27_t0.6 | 0.000 | [0.000, 0.071] |
| layer27_t0.75 | 0.000 | [0.000, 0.071] |
| layer27_t0.9 | 0.000 | [0.000, 0.071] |
| layer27_t1 | 0.000 | [0.000, 0.071] |

Interpretation: no strict first-commit utility gain.

### n=50 last-answer metric

| condition | last-answer acc | diff vs baseline | + flips | - flips | exact p |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.280 | +0.000 | 0 | 0 | 1.000 |
| layer27_t0.6 | 0.660 | +0.380 | 19 | 0 | 0.000 |
| layer27_t0.75 | 0.660 | +0.380 | 19 | 0 | 0.000 |
| layer27_t0.9 | 0.660 | +0.380 | 19 | 0 | 0.000 |
| layer27_t1 | 0.320 | +0.040 | 12 | 10 | 0.832 |

Interpretation: trajectory-level answer appearance is real; first-final commitment remains unsolved.

### Neuron/SAE smoke gate

| artifact | result |
| --- | --- |
| neuron ranking | 100 MLP coordinates ranked; top rows dominated by layer27 in tiny smoke set |
| SAE | layer18 resid_post SAE trained; final_loss 4.9134; explained_variance 0.6525; L0 93.125; dead_feature_rate 0.6094 |

Interpretation: pipeline is runnable, but no neuron/SAE feature should be called a reasoning circuit yet.

## 3. Final Scientific State

Settled:

- Late-layer hook-state interventions are real posture interventions.
- Layer27 partial interpolation has a strong n=50 last-answer effect.
- Strict first-final correctness is still not improved.
- Protocol/format binding is now the main bottleneck.
- Neuron/SAE path is operational but not scientifically validated.

Not settled:

- A reliable reasoning circuit.
- A safe closed-loop controller.
- A semantically clean data-driven token metric.
- A neuron or SAE feature with causal validation.

## 4. Verification

Final verification command:

```bash
python -m pytest -q
```

Result: `25 passed`.

Security check before each push found no PAT token in tracked workspace files.

## 5. Final Claim

The third-stage objective is complete. The repo now has a rigorous, evidence-backed state:

> The project has progressed from an early positive layer27 story to a clearer causal audit: layer27 can move trajectories and last answers, but correct first-final commitment and safe controller design remain unsolved research problems.
