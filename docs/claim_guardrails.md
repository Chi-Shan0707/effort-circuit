# Claim Guardrails

Use this document when writing abstracts, README text, paper drafts, issue summaries, or future experiment reports.

## Green Claims: Safe To Say

- Qwen3-0.6B has late-layer decoder-block hook states that causally affect continuation/finalization posture in the tested synthetic arithmetic setting.
- Activation patching and interpolation can alter next-token continuation/finalization behavior.
- Hook-captured decoder block states are the safe default for current interventions.
- The site audit found that layer 18 was site-equivalent in the current run, while layer 27 `outputs.hidden_states[layer+1]` was not equivalent to the decoder-block hook output.
- Older layer 27 hidden-state patching should be treated as site-mismatch/off-manifold/readout-boundary intervention unless rerun with hook-captured states.
- Stop-after-first-final evaluation found no reliable first-committed answer correctness improvement in the current heldout runs.
- Last-detectable-answer accuracy can improve under some layer27 hook-state interventions, but this is an auxiliary weak metric and not first-commit correctness.
- The current result is a strong process-posture control lead, not a demonstrated reasoning-accuracy intervention.

## Yellow Claims: Hypotheses Only

- Late-layer posture states may be useful for future closed-loop controllers if paired with answer-readiness and format-validity checks.
- Layer 27 may sit near a readout boundary that strongly controls output policy, but its old hidden-state patch results are not site-clean.
- Partial layer27 interpolation may move generations into a trajectory basin that makes correct answers appear more often somewhere in the text.
- Mean process directions may reduce source-answer leakage compared with raw cross-question state replacement, but they still damage target answer availability.
- Neuron/SAE analyses may eventually identify contributors to the posture signal, but current neuron/SAE artifacts are smoke gates, not validated features.

## Red Claims: Do Not Say

- Do not say the repo found a reasoning neuron.
- Do not say the repo found a stable reasoning circuit.
- Do not say layer 27 improves reasoning accuracy.
- Do not say the current interventions reliably improve first-final answer correctness.
- Do not say last-extracted-answer accuracy is equivalent to first-committed correctness.
- Do not present old `docs/final_analysis_report.md` as the current conclusion.
- Do not describe old layer27 hidden-state patching as site-clean.
- Do not claim neuron/SAE features are interpretable or causal unless future ablation/steering validation proves it.
