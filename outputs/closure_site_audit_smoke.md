# Activation Site Audit

## Purpose

Audit whether `outputs.hidden_states[layer+1]` can be safely patched into decoder block hook output at the same layer.

## Summary

| layer | passive hook logit Δ max | self-hook patch logit Δ max | hidden-states→hook logit Δ max | hidden-states vs hook-state Δ max |
| ---: | ---: | ---: | ---: | ---: |
| 18 | 0 | 0 | 0 | 0 |
| 27 | 0 | 0 | 16.8005 | 253.924 |

## Checks Reported

- Passive hook identity check: `base_vs_hook_run_logit_max_abs_delta`.
- Self-hook patch identity check: `self_hook_patch_logit_max_abs_delta`.
- Hidden-states-to-hook equivalence check: `hidden_state_patch_logit_max_abs_delta` and `hidden_state_vs_hook_state_max_abs_delta`.
- Per-layer max logit deltas and max state deltas are summarized above and retained per row in JSON.

## Current Run Note

In the current archived audit (`outputs/site_audit_n2_layers18_27.*`), layer 18 passed site-equivalence while layer 27 failed site-equivalence. Therefore old layer27 hidden-state patch conclusions are downgraded to site-mismatch/off-manifold/readout-boundary interventions unless rerun with hook-captured decoder block states.

## Interpretation Guide

- Passive hook logit delta should be near zero; otherwise merely observing the hook changes the run.
- Self hook patch logit delta should be near zero; otherwise the hook/patch machinery is not identity-preserving.
- Hidden-state patch delta near zero means HuggingFace hidden state and hook output are equivalent for that layer/site.
- Large hidden-state-vs-hook delta means prior patching mixed incompatible activation sites and must be interpreted as a site-mismatch intervention.
