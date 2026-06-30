# Activation Site Audit

## Purpose

Audit whether `outputs.hidden_states[layer+1]` can be safely patched into decoder block hook output at the same layer.

## Summary

| layer | self hook patch logit Δ max | hidden-state patch logit Δ max | hidden-state vs hook-state Δ max |
| ---: | ---: | ---: | ---: |
| 18 | 0 | 0 | 0 |
| 27 | 0 | 16.8005 | 256.941 |

## Interpretation Guide

- Self hook patch logit delta should be near zero; otherwise the hook/patch machinery is not identity-preserving.
- Hidden-state patch delta near zero means HuggingFace hidden state and hook output are equivalent for that layer/site.
- Large hidden-state-vs-hook delta means prior patching mixed incompatible activation sites and must be interpreted as a site-mismatch intervention.
