# Second Prompt Roadmap

Source: `second_prompt.md`

## Objective

Turn the current strong lead into a cleaner causal intervention result by auditing activation sites, improving generation metrics, expanding heldout evaluation, and adding task-out / verbosity controls.

## Route

1. **Activation-site audit**
   - Implement same-run self patch.
   - Implement hidden-states-to-hook patch.
   - Implement high prompt self-equivalence.
   - Gate: layer 27 patching must have an explicit site-equivalence interpretation before further claims.

2. **Stop-after-first-final generation**
   - Generate token-by-token.
   - Stop once a valid `Final answer: <number>` line appears.
   - Record first-final accuracy, last-final accuracy, tokens to first final, answer changes, post-final continuation, repetition.

3. **Expanded heldout statistics**
   - Evaluate baseline and layer27 interpolation conditions.
   - Report accuracy, Wilson confidence intervals, paired bootstrap CI, and McNemar/exact-style paired counts.
   - Use CPU-aware staged runs; start small but keep commands scalable to n=50+.

4. **Task-out and verbosity controls**
   - Add non-reasoning control tasks.
   - Add verbosity/generic continuation controls.
   - Gate: useful-effort claim requires reasoning-task gains without uncontrolled task-out verbosity.

5. **Data-driven token metric**
   - Replace hand-coded token sets with train-selected high/low discriminative token clusters.
   - Evaluate on heldout with KL, answer-token margin, and EOS/Final margin.

6. **Process/task disentanglement**
   - Run cross-question patching and mean process-direction addition.
   - Check source-answer leakage and target-answer preservation.

7. **Closed-loop CET controller**
   - Implement token-level alpha control from effort/final/repetition signals.
   - Stop on valid final and reduce steering under repetition or late-stage answer readiness.

8. **Synthesis**
   - Produce a comprehensive second-stage report that separates settled evidence, weak evidence, contradictions, and next experiments.

## Commit Policy

Every completed route step must be committed and pushed before moving to the next step.
