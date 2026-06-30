# Third Prompt Roadmap and Coverage Audit

Source: `third_prompt.md`

## Objective

Continue the mechanistic-control project after the second-stage audit. The third prompt largely re-evaluates the repo, repeats the high-priority gates, and adds one still-missing scale requirement: n=50 heldout evaluation for site-audited layer27 interpolation conditions.

## Coverage Status

| Third prompt requirement | Current evidence | Status |
| --- | --- | --- |
| Activation site audit | `docs/step_01_site_audit.md`, `outputs/site_audit_n2_layers18_27.*` | Completed before this stage |
| Stop-after-first-final | `docs/step_02_stop_after_final.md`, `outputs/stop_after_final_heldout_n5_hook_states.*` | Completed before this stage |
| n=50 heldout + paired stats | `docs/third_stage_n50_layer27_report.md`, `outputs/stop_after_final_heldout_n50_layer27.*`, `outputs/stop_after_final_stats_n50_layer27.*` | Completed in this stage |
| Replace hand-coded token metric | `docs/step_05_data_driven_metric.md` | Completed before this stage |
| Process/content disentanglement | `docs/step_06_process_task_disentanglement.md` | Completed before this stage |
| Task-out and verbosity controls | `docs/step_04_task_out_controls.md` | Completed before this stage |
| CET controller | `docs/step_07_closed_loop_cet.md` | Completed before this stage |
| Neuron/SAE after residual posture is clarified | `docs/third_stage_neuron_sae_gate.md`, `outputs/third_stage_neurons_n8.json`, `outputs/third_stage_sae_layer18_smoke/metrics.json` | Smoke gate completed; scientific discovery still future work |

## Third-Stage Addition

The new concrete artifact in this stage is a larger heldout evaluation:

```bash
python -m src.stop_after_final_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset heldout_synthetic_math \
  --n 50 \
  --low-mode instant \
  --high-mode verify \
  --conditions 27:0.6,27:0.75,27:0.9,27:1 \
  --max-new-tokens 96 \
  --out outputs/stop_after_final_heldout_n50_layer27.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

This directly addresses the third prompt's request to move beyond n=5.
