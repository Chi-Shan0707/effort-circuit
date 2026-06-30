# Closure Report

## Abstract

This repository provides early mechanistic evidence that Qwen3-0.6B contains causally controllable late-layer generation-posture states. Activation patching and interpolation alter continuation/finalization behavior. However, a site audit shows that prior layer 27 hidden-state patching mixed non-equivalent activation sites, and stop-after-first-final evaluation shows no reliable first-committed answer correctness improvement in the current heldout n=5 run. The current result is therefore a strong process-posture control lead, not a demonstrated reasoning-accuracy intervention.

## What Was Found

- Late decoder-block hook states causally affect continuation/finalization posture in the tested synthetic arithmetic setting.
- Activation patching and interpolation alter local next-token behavior and generation trajectories.
- Hook-captured states are safer than `outputs.hidden_states` for current interventions.
- Layer 18 passed the current site-equivalence audit, while layer 27 failed it.
- n=50 heldout evaluation found a strong last-detectable-answer signal for partial layer27 hook-state interpolation, but that signal remains an auxiliary weak metric.

## What Was Falsified

- The strong claim that current interventions reliably improve first-committed answer correctness is not supported.
- The older heldout-small optimistic accuracy story does not survive strict stop-after-first-final evaluation.
- Old layer27 hidden-state patching cannot be described as a clean same-site activation patch.
- Naive hand-coded or raw data-driven continue/final token metrics are not sufficient evidence of useful reasoning.
- The current closed-loop CET scaffold is not a reliable controller; it can reduce steering strength, but it does not produce correct strict finals.

## What Remains Open

- Whether a safer hidden-state probe can separate process posture from answer content and formatting.
- Whether answer-readiness checks can preserve target answer availability while using posture control.
- Whether task-out and verbosity controls remain clean at larger scale and across more tasks.
- Whether neuron/SAE analyses can identify validated causal contributors to the posture signal.
- Whether future closed-loop controllers can turn posture control into reliable first-final correctness.

## Allowed vs Disallowed Claims

Allowed:

- The repository has evidence for a causally controllable late-layer generation-posture signal.
- Hook-captured late decoder-block states can alter continuation/finalization behavior.
- The current interventions are not yet reliable first-final answer controllers.
- Old layer27 hidden-state patching is site-mismatch/off-manifold/readout-boundary evidence unless rerun with hook-captured states.

Disallowed:

- Reasoning neuron.
- Stable accuracy improvement.
- Layer 27 improves reasoning accuracy.
- Reliable first-final answer correctness improvement.
- Site-clean layer27 hidden-state patching.

## Statistical Interpretation

The heldout n=5 stop-after-first-final run is not enough to settle a precise effect size. It is enough to falsify the old strong first-final utility claim in the current run because all tested conditions had zero strict first-final accuracy.

The later n=50 run sharpens this conclusion: first-final accuracy remains zero for baseline and layer27 interpolation conditions, while last-detectable-answer accuracy can improve. This separates trajectory-level answer appearance from first-committed answer correctness.

## Authoritative Report Order

1. `docs/current_analysis_report.md` is the authoritative current synthesis.
2. `docs/closure_report.md` summarizes the closed iteration.
3. `docs/claim_guardrails.md` governs future wording.
4. `docs/final_analysis_report.md` is historical only.

## Verification

Final verification should run:

```bash
python -m pytest -q
bash scripts/run_closure_checks.sh
```

Final closure results on 2026-06-30 UTC:

- `python -m pytest -q`: `26 passed`.
- `bash scripts/run_closure_checks.sh`: completed successfully.
- The closure script ran CLI help checks and, because the local Qwen3-0.6B model path existed, generated small smoke artifacts under `outputs/closure_*`.
- No failures were observed.
