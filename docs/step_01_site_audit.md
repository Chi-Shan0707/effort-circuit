# Step 1: Activation Site Audit

## Requirement

`second_prompt.md` flagged a critical risk: prior experiments patched `outputs.hidden_states[layer+1]` into decoder block hook outputs. These may not be the same activation site. Before further claims, we must audit site equivalence.

## Implementation

Added `src/site_audit.py` with three checks:

1. **Base vs hook run**: adding a passive hook should not change logits.
2. **Self hook patch**: capture hook output and patch it back into the same hook site; logits should remain unchanged.
3. **Hidden-state-to-hook patch**: patch `outputs.hidden_states[layer+1]` into the hook site; nonzero delta indicates site mismatch.

## Command

```bash
python -m src.site_audit \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 2 \
  --layers 18,27 \
  --out outputs/site_audit_n2_layers18_27.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Result

See `outputs/site_audit_n2_layers18_27.md` and JSON artifact.

## Interpretation

If self-hook patch deltas are near zero but hidden-state patch deltas are large, the patching machinery itself is valid, but the hidden-state index/site is not equivalent to block hook output. In that case previous layer-27 results remain useful as interventions, but they must be described as **site-mismatch interventions** unless rerun with hook-captured states.

## Follow-up

Step 2 should implement stop-after-first-final generation. A later step should rerun strict generation using hook-captured source states rather than `outputs.hidden_states` if this audit shows mismatch.

## Closure Note

The closure smoke audit confirms the key site result:

- Layer 18: passive hook and self-hook patch are identity-preserving, and `outputs.hidden_states[layer+1]` is equivalent to the decoder-block hook output in the current run.
- Layer 27: passive hook and self-hook patch are identity-preserving, but `outputs.hidden_states[layer+1]` is not equivalent to the decoder-block hook output; the closure smoke run observed a hidden-states-to-hook logit max delta of about `16.8005` and state max delta of about `253.924`.
- Therefore old layer27 hidden-state patch conclusions are downgraded to site-mismatch/off-manifold/readout-boundary interventions unless rerun with hook-captured decoder block states.
