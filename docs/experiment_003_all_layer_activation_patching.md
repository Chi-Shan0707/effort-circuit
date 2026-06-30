# Experiment 003: All-Layer Activation Patching Finds Late-Layer Causal Sites

Date: 2026-06-30 UTC

## Conclusion

The all-layer activation patching experiment found a much stronger and cleaner causal signal than the earlier direction-addition experiments.

- Low mode `instant` mean continue-final gap: 9.88969e-05
- High mode `verify` mean continue-final gap: 0.307173
- Best low→high patch: layer 27, delta vs low 0.734238
- Best high→low patch: layer 18, delta vs high -0.317724

This passes a stronger causal criterion than Experiment 001/002: replacing an actual high-effort hidden state into a low-effort run restores continuation-token preference, and replacing a low-effort state into a high-effort run suppresses it.

## Why This Is Valuable

From first principles, an internal state is causally relevant if intervening on it changes downstream logits while holding the prompt/run otherwise fixed. Direction addition can be off-manifold and random directions can accidentally affect logits. Activation patching is more diagnostic because it transfers a real hidden state from a clean run to a corrupt run.

The all-layer result suggests the “continue reasoning vs answer now” variable is most accessible in late layers, especially layers 18–27, not in the early layers that looked strongest under raw direction addition.

## Command

```bash
python -m src.activation_patching_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 8 \
  --low-mode instant \
  --high-mode verify \
  --layers all \
  --out outputs/activation_patching_instant_verify_n8_all_layers.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Low→High Restore Ranking

Patching `verify` hidden state into the `instant` run:

| rank | layer | patched delta vs low | high→low delta vs high |
| ---: | ---: | ---: | ---: |
| 1 | 27 | 0.734238 | -0.307079 |
| 2 | 20 | 0.273016 | -0.308342 |
| 3 | 23 | 0.26837 | -0.307154 |
| 4 | 21 | 0.26558 | -0.306911 |
| 5 | 24 | 0.261163 | -0.307258 |
| 6 | 19 | 0.259649 | -0.311671 |
| 7 | 25 | 0.250527 | -0.306917 |
| 8 | 26 | 0.249288 | -0.306968 |
| 9 | 22 | 0.245819 | -0.307303 |
| 10 | 18 | 0.218337 | -0.317724 |
| 11 | 17 | 0.193115 | -0.306183 |
| 12 | 16 | 0.156109 | -0.261429 |
| 13 | 15 | 0.131339 | -0.247052 |
| 14 | 11 | 0.0788699 | 0.0911547 |
| 15 | 14 | 0.0742495 | -0.0853407 |
| 16 | 12 | 0.0690262 | 0.012455 |
| 17 | 13 | 0.0626366 | -0.000199211 |
| 18 | 10 | 0.0578238 | 0.115668 |
| 19 | 9 | 0.0468804 | 0.150691 |
| 20 | 7 | 0.0194822 | 0.146019 |
| 21 | 8 | 0.0194175 | 0.130894 |
| 22 | 6 | 0.0174922 | 0.120168 |
| 23 | 5 | 0.00619129 | 0.122743 |
| 24 | 2 | 0.000952404 | 0.10262 |
| 25 | 4 | 0.000600995 | 0.176525 |
| 26 | 3 | 0.000457203 | 0.150314 |
| 27 | 1 | 0.000227861 | 0.0786593 |
| 28 | 0 | 5.8573e-05 | 0.110359 |

## High→Low Suppression Ranking

Patching `instant` hidden state into the `verify` run:

| rank | layer | high→low delta vs high | low→high delta vs low |
| ---: | ---: | ---: | ---: |
| 1 | 18 | -0.317724 | 0.218337 |
| 2 | 19 | -0.311671 | 0.259649 |
| 3 | 20 | -0.308342 | 0.273016 |
| 4 | 22 | -0.307303 | 0.245819 |
| 5 | 24 | -0.307258 | 0.261163 |
| 6 | 23 | -0.307154 | 0.26837 |
| 7 | 27 | -0.307079 | 0.734238 |
| 8 | 26 | -0.306968 | 0.249288 |
| 9 | 25 | -0.306917 | 0.250527 |
| 10 | 21 | -0.306911 | 0.26558 |
| 11 | 17 | -0.306183 | 0.193115 |
| 12 | 16 | -0.261429 | 0.156109 |
| 13 | 15 | -0.247052 | 0.131339 |
| 14 | 14 | -0.0853407 | 0.0742495 |
| 15 | 13 | -0.000199211 | 0.0626366 |
| 16 | 12 | 0.012455 | 0.0690262 |
| 17 | 1 | 0.0786593 | 0.000227861 |
| 18 | 11 | 0.0911547 | 0.0788699 |
| 19 | 2 | 0.10262 | 0.000952404 |
| 20 | 0 | 0.110359 | 5.8573e-05 |
| 21 | 10 | 0.115668 | 0.0578238 |
| 22 | 6 | 0.120168 | 0.0174922 |
| 23 | 5 | 0.122743 | 0.00619129 |
| 24 | 8 | 0.130894 | 0.0194175 |
| 25 | 7 | 0.146019 | 0.0194822 |
| 26 | 3 | 0.150314 | 0.000457203 |
| 27 | 9 | 0.150691 | 0.0468804 |
| 28 | 4 | 0.176525 | 0.000600995 |

## Mechanistic Interpretation

1. **Late-layer state hypothesis**: layers 18–27 encode an actionable “continue reasoning / finalization posture” that directly controls next-token probabilities.
2. **Layer 27 amplification**: layer 27 low→high patch overshoots the original high-mode gap. This may be a real amplification site or an off-distribution artifact from patching only the last residual position.
3. **Bidirectionality**: unlike the early-layer-only patch, all-layer patching shows true bidirectionality in late layers: high states restore continuation in low runs, and low states suppress continuation in high runs.
4. **Direction vs state distinction**: early direction addition can move logits, but late activation patching provides cleaner evidence for a causal state variable.

## Updated Experimental Priority

The next most valuable work is no longer broad random steering. It is focused late-layer intervention:

- Test layers 18, 20, 23, 27 with generation under stricter `Reasoning:`/`Final:` protocol.
- Use patching-derived vectors or interpolation between low/high states instead of arbitrary alpha addition.
- Add off-manifold checks: interpolate `h = (1-t) h_low + t h_high` for t in `[0, 0.25, 0.5, 0.75, 1]` and look for monotonic logit response.
- Add task controls where extra reasoning should not help.

## Artifacts

- JSON: `outputs/activation_patching_instant_verify_n8_all_layers.json`
- Markdown: `outputs/activation_patching_instant_verify_n8_all_layers.md`
- Script: `src/activation_patching_experiment.py`

## Verification

- `python -m pytest -q` passed: 8 tests.
- Experiment used measured best CPU setting: 4 Torch/BLAS threads.
