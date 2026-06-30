# Experiment 002: Causal Intervention Controls and Patching

Date: 2026-06-30 UTC

## Conclusion

We ran three causal follow-ups from `background.md`: random-direction controls, short autoregressive generation, and activation patching. The most valuable finding is not a clean “reasoning vector solves the task” story; it is a more precise mechanistic picture:

1. The layer-4 alpha=-3 direction remains the strongest tested direction on the local continue-vs-final token metric, but random directions can get close. This means the effect is real under the metric but the metric is not yet specific enough to prove a unique effort circuit.
2. Direct generation interventions did not yet produce a useful accuracy result. The model often emits a number and then starts explaining, so short generation is contaminated by formatting behavior and cutoff effects.
3. Activation patching gives a stronger causal clue: patching `verify` hidden state into an `instant` run partially restores the high-effort next-token gap, strongest around layer 10. Reverse patching did **not** suppress high-effort behavior, so the mechanism is not a simple one-layer replaceable variable.

## First-Principles Interpretation

From first principles, a useful causal intervention must pass increasingly strict gates:

- **Local causal gate**: changing internal state changes next-token probabilities in the intended axis.
- **Specificity gate**: the effect beats random directions and verbosity controls.
- **Trajectory gate**: the intervention changes generated trajectories, not only one-step logits.
- **Utility gate**: changed trajectories improve correctness or cost-adjusted correctness without uncontrolled verbosity.

Experiment 001 passed the local causal gate. Experiment 002 shows the specificity and trajectory gates are only partially satisfied.

## Random Direction Control

Artifact: `outputs/random_control_verify_n8_layer4_alpha-3.json`

- Candidate delta gap: 0.115317
- Random mean delta: 0.0383364
- Random min delta: -0.0457047
- Random max delta: 0.114225
- Candidate percentile among 32 random directions: 1.000

Interpretation: the candidate beats all 32 sampled random directions, but the best random direction is very close. Therefore this is not yet a highly specific effort vector; it is a promising candidate requiring stronger controls.

## Short Generation Intervention

Artifacts:

- `outputs/generation_intervention_verify_n3_layer4_alpha-3.json`
- `outputs/generation_instant_intervention_verify_n5_layer4_alpha-3.json`

Instant-prompt summary:

| condition | accuracy | mean reasoning tokens | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 16.00 | 0.000 | 0.0000 |
| effort_candidate | 0.000 | 16.00 | 0.000 | 0.0000 |
| effort_opposite | 0.000 | 16.00 | 0.000 | 0.0000 |
| random_control | 0.000 | 16.00 | 0.000 | 0.0000 |

Interpretation: this generation test is mostly a negative/diagnostic result. The model already tends to emit a number and then explanation even under `instant`; all conditions hit the short token cap and produced no clean final-answer format. This prevents a fair correctness or verbosity conclusion.

Concrete lesson: future trajectory tests should use a stricter output protocol or stepwise controller metrics rather than relying on short free generation.

## Activation Patching

Artifact: `outputs/activation_patching_instant_verify_n8_layers0-10.json`

- Low mode: `instant`
- High mode: `verify`
- Mean low gap: 9.88969e-05
- Mean high gap: 0.307173
- Best low→high restore layer: 10, delta vs low 0.0578238
- Smallest high→low delta layer: 1, delta vs high 0.0786593

Low→high patch ranking:

| layer | patched delta vs low |
| ---: | ---: |
| 10 | 0.0578238 |
| 9 | 0.0468804 |
| 7 | 0.0194822 |
| 8 | 0.0194175 |
| 6 | 0.0174922 |
| 5 | 0.00619129 |
| 2 | 0.000952404 |
| 4 | 0.000600995 |
| 3 | 0.000457203 |
| 1 | 0.000227861 |
| 0 | 5.8573e-05 |

Interpretation: patching the high-effort state into the low-effort run partially restores the high-effort next-token gap, especially at layers 9–10. This is a stronger causal result than pure direction addition because it transfers an actual internal state from one run to another.

However, the reverse patch did not suppress the high-effort run; all high→low deltas are positive rather than negative. This contradicts the simplest hypothesis that one layer contains a clean scalar “continue reasoning” state. A better hypothesis is that high-effort behavior is distributed and the one-layer low-state replacement creates an off-manifold state that can increase continuation mass.

## Updated Mechanistic Hypotheses

1. **Early steering direction hypothesis**: layers 0–7 contain directions that can strongly shift local continuation token mass, but these directions are not highly unique under the current metric.
2. **Mid-layer patch hypothesis**: layers 9–10 carry a more state-like representation that can transfer high-effort prompt effects into instant prompts.
3. **Distributed trajectory hypothesis**: useful reasoning effort is not a single layer variable; reverse patching failure suggests multi-layer or trajectory-level interactions.

## Next High-Value Interventions

1. Run activation patching over all 28 layers to see whether layers beyond 10 dominate restore/suppress effects.
2. Add a matched verbosity direction so the local metric is not merely measuring generic “starts explanatory text”.
3. Replace free generation with stepwise CET-style control that records per-token effort score, answer-token margin, and stop decisions.
4. Use task prompts that enforce `Reasoning:` and `Final:` more tightly, then evaluate correctness only after `Final:` appears.

## Verification

- `python -m pytest -q` passed after adding the new scripts.
- Random control artifact exists and is readable.
- Generation intervention artifacts exist and are readable.
- Activation patching artifact exists and is readable.
