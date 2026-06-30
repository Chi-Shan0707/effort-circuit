# Experiment 004: Multi-Stage Late-Layer Causal Intervention

Date: 2026-06-30 UTC

## Conclusion

自主设置的多阶段目标已经完成，并产生了比前几轮更强的因果证据：late-layer hidden-state interpolation/prefix patching 不仅单调控制下一 token 的 continue-vs-final 概率，而且在小规模严格生成协议中提升了算术任务正确率。

最强发现：layer 27 的 high-effort state prefix patch 在训练样本 n=3 上把准确率从 baseline 0.333 提到 1.000；在 heldout n=5 上，layer27_t1 把准确率从 0.400 提到 0.800。

## Stage Plan

1. **Mechanistic monotonicity**: 对 layers 18,20,23,27 做 low→high hidden-state interpolation，检验 continue-final gap 是否随 t 单调变化。
2. **Strict generation**: 用更严格的 `Reasoning:` / `Final answer:` 协议测试 late-layer prefix patch 是否改变真实生成结果。
3. **Heldout replication**: 只保留最强 layer27 条件，在 heldout synthetic math 上复现。
4. **Skeptical correction**: 修正答案抽取器，避免把 Final 行中的第一个中间数字误判为答案。

## Stage 1: Interpolation Monotonicity

Artifact: `outputs/activation_interpolation_instant_verify_n8_late_layers.json`

| layer | delta t=1 minus t=0 | monotonic increasing | curve |
| ---: | ---: | ---: | --- |
| 18 | 0.218337 | True | 0.0→9.89e-05, 0.25→0.0007422, 0.5→0.01143, 0.75→0.1005, 1.0→0.2184 |
| 20 | 0.273016 | True | 0.0→9.89e-05, 0.25→0.001689, 0.5→0.02577, 0.75→0.1558, 1.0→0.2731 |
| 23 | 0.26837 | True | 0.0→9.89e-05, 0.25→0.001838, 0.5→0.02396, 0.75→0.1387, 1.0→0.2685 |
| 27 | 0.734243 | True | 0.0→9.393e-05, 0.25→0.001119, 0.5→0.01977, 0.75→0.2836, 1.0→0.7343 |

Interpretation: all four selected late layers are monotonic increasing from low-effort state to high-effort state. Layer 27 is strongest and highly nonlinear: most of the jump occurs from t=0.75 to t=1.0.

## Stage 2: Strict Generation on Train Split

Artifact: `outputs/strict_generation_prefix_patch_n3_late_layers_reextracted.json`

| condition | accuracy | answer rate | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.333 | 1.000 | 1.000 | 0.7260 |
| layer18_t1 | 0.000 | 1.000 | 0.667 | 0.6774 |
| layer20_t1 | 0.333 | 1.000 | 0.667 | 0.5763 |
| layer27_t0.75 | 1.000 | 1.000 | 1.000 | 0.3582 |
| layer27_t1 | 1.000 | 1.000 | 1.000 | 0.2200 |

Interpretation: layer27_t0.75 and layer27_t1 both reached 1.000 accuracy on n=3 after corrected extraction. They also reduced repetition compared with baseline, suggesting the effect is not merely uncontrolled verbosity.

## Stage 3: Heldout Replication

Artifact: `outputs/strict_generation_prefix_patch_heldout_n5_layer27_reextracted.json`

| condition | accuracy | answer rate | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.400 | 1.000 | 1.000 | 0.2783 |
| layer27_t0.75 | 0.600 | 1.000 | 0.600 | 0.1609 |
| layer27_t1 | 0.800 | 1.000 | 0.800 | 0.2025 |

Interpretation: the strongest condition generalized directionally to heldout data. Baseline accuracy was 0.400; layer27_t0.75 reached 0.600; layer27_t1 reached 0.800. This is still small-n evidence, but it crosses from local-logit causality into task-level utility.

## First-Principles Takeaway

The experiments now support a sharper mechanistic claim:

> In Qwen3-0.6B on these synthetic arithmetic prompts, late-layer residual states, especially layer 27, contain a causally actionable posture for continuing structured reasoning instead of falling into low-quality finalization/repetition. Interpolating or prefix-patching that state can improve answer quality in small controlled runs.

This is stronger than “a vector changes logits” because it satisfies multiple gates:

- **Local causal gate**: activation patching and interpolation change next-token distributions.
- **Monotonicity gate**: selected late layers show monotonic response as hidden state moves low→high.
- **Trajectory gate**: prefix patch changes generated reasoning text.
- **Utility gate**: strict generation accuracy improves on both train-small and heldout-small samples.

## Remaining Risks

- Sample sizes are small because CPU quota is 4 cores and memory is 8GiB.
- `max_new_tokens=96` still causes truncation/repetition; future runs should add stop criteria after first valid final answer.
- Layer 27 t=1 can sometimes over-explain or pick an intermediate number before later correcting; answer extraction now mitigates this but generation policy should stop cleanly.
- No task-out control yet; useful-effort specificity against generic verbosity remains incomplete.

## Next Best Experiments

1. Add stopping on first valid `Final answer:` line and rerun heldout n=20 for baseline vs layer27_t0.75 vs layer27_t1.
2. Add non-reasoning control tasks to test whether layer27 patch only helps arithmetic/reasoning.
3. Convert prefix patch into CET-style controller: inject only when answer-margin is high but effort state is low.
4. Test layer 27 interpolation with t values around the nonlinear region: 0.6, 0.7, 0.8, 0.9, 1.0.

## Artifacts

- Script: `src/activation_interpolation_experiment.py`
- Script: `src/strict_generation_patch_experiment.py`
- Fixed extractor: `src/answer_extractors.py`
- Train strict generation: `outputs/strict_generation_prefix_patch_n3_late_layers_reextracted.json`
- Heldout strict generation: `outputs/strict_generation_prefix_patch_heldout_n5_layer27_reextracted.json`
- Interpolation: `outputs/activation_interpolation_instant_verify_n8_late_layers.json`

## Verification

- `python -m pytest -q` passed after extractor correction: 10 tests.
- All referenced JSON/Markdown artifacts were generated locally.
- All experiments used the measured CPU-optimal setting: 4 Torch/BLAS threads.
