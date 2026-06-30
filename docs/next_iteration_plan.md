# Next Iteration Plan

Closure goal for the current iteration is clarity, not breadth. The following are future directions, not current positive claims.

## 1. Task-Out and Verbosity Controls

Scale task-out controls beyond the current smoke runs. Include arithmetic/date/symbolic reasoning and non-reasoning tasks such as capital lookup, translation, sentiment rewrite, and formatting-only prompts.

## 2. Data-Driven Continue/Final Token Metric

Replace fragile hand-coded token sets with metrics that separate prompt artifacts from process posture. Train on one split and evaluate on heldout. Include KL, answer-token margin, EOS/Final margin, and format/verbosity controls.

## 3. Process/Task Disentanglement

Use cross-question patching and mean process-direction interventions to test whether process posture transfers without source-answer leakage or target-answer suppression.

## 4. Token-Level CET Controller

Build a safer closed-loop controller that monitors final margin, answer readiness, repetition, and valid final markers. Reduce or stop steering when target answer availability collapses.
