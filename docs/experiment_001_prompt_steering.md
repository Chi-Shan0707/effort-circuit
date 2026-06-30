# Experiment 001: Prompt-Level Effort Steering

Date: 2026-06-30 UTC

## Conclusion

We found a reproducible prompt-level causal signal in Qwen3-0.6B: the hidden-state difference between `verify` and `instant` prompts can shift the next-token distribution toward continuation/reasoning tokens rather than final-answer tokens.

The strongest replicated intervention is early-layer and sign-reversed relative to the raw `verify - instant` direction:

- n=2 screen: best layer 4, alpha -3.0, delta gap 0.165293
- n=8 all-layer run: best layer 4, alpha -3.0, delta gap 0.115317

This is **not yet** a full accuracy or reasoning-length claim. It is a first-principles local causal result: an internal direction changes the immediate token distribution in the intended conceptual axis.

## First-Principles Framing

Autoregressive generation starts with one irreducible mechanism: hidden state -> logits -> next token distribution. Before asking whether a direction improves long-horizon reasoning accuracy, we first ask whether it causally changes the local distribution between:

- continuation/reasoning tokens: `First`, `Let's`, `We`, `Since`, `Calculate`, `Step`, `Wait`, `The`
- finalization tokens: `Final`, `Answer`, `Therefore`, `Thus`, `So`

If this local causal test fails, expensive generation sweeps are unlikely to be informative. It passed on the current small experiment.

## Environment Configuration

Resource profiling showed the host exposes 240 logical CPUs, but the container cgroup quota is approximately 4 cores and 8 GiB RAM. The measured best configuration is:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

and `torch.set_num_threads(4)` / `--num-threads 4`.

See `docs/resource_profile.md` and `outputs/cpu_thread_benchmark.json`.

## Commands Run

```bash
python -m src.prompt_steering_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 8 \
  --high-mode verify \
  --layers all \
  --alphas=-3,-1,0,1,3 \
  --out outputs/prompt_steering_verify_n8_all_layers.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## n=8 Baseline

- Mean continue mass: 0.379785
- Mean final mass: 0.00180739
- Mean continue-final gap: 0.377977

## n=8 Top Positive Interventions

| rank | layer | alpha | continue mass | final mass | gap | delta gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | -3.0 | 0.494609 | 0.00131444 | 0.493294 | 0.115317 |
| 2 | 0 | -3.0 | 0.491051 | 0.00163811 | 0.489413 | 0.111436 |
| 3 | 3 | -3.0 | 0.477953 | 0.00118226 | 0.476771 | 0.0987934 |
| 4 | 6 | -3.0 | 0.475395 | 0.00177789 | 0.473617 | 0.0956394 |
| 5 | 7 | -3.0 | 0.462938 | 0.00173844 | 0.4612 | 0.0832224 |
| 6 | 5 | -3.0 | 0.458414 | 0.00165448 | 0.45676 | 0.0787823 |
| 7 | 2 | -3.0 | 0.448104 | 0.00133158 | 0.446773 | 0.0687955 |
| 8 | 1 | 3.0 | 0.440786 | 0.00173272 | 0.439053 | 0.0610757 |
| 9 | 8 | -3.0 | 0.421164 | 0.00202056 | 0.419143 | 0.0411658 |
| 10 | 0 | 3.0 | 0.416343 | 0.00175668 | 0.414586 | 0.0366092 |
| 11 | 10 | -3.0 | 0.416126 | 0.00202771 | 0.414099 | 0.0361212 |
| 12 | 9 | -3.0 | 0.413893 | 0.00219166 | 0.411701 | 0.0337241 |

## n=8 Top Negative Interventions

| rank | layer | alpha | continue mass | final mass | gap | delta gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 3.0 | 0.301831 | 0.00148453 | 0.300347 | -0.0776305 |
| 2 | 6 | 3.0 | 0.315117 | 0.00158385 | 0.313533 | -0.0644441 |
| 3 | 5 | 3.0 | 0.331457 | 0.00144536 | 0.330011 | -0.047966 |
| 4 | 14 | 3.0 | 0.348265 | 0.00177704 | 0.346488 | -0.0314889 |
| 5 | 8 | 3.0 | 0.348381 | 0.00166406 | 0.346717 | -0.03126 |
| 6 | 10 | 3.0 | 0.349958 | 0.00172612 | 0.348232 | -0.0297453 |
| 7 | 6 | 1.0 | 0.352596 | 0.00176393 | 0.350832 | -0.0271448 |
| 8 | 7 | 1.0 | 0.353884 | 0.00176427 | 0.352119 | -0.025858 |
| 9 | 15 | 3.0 | 0.354135 | 0.00181627 | 0.352319 | -0.0256587 |
| 10 | 13 | 3.0 | 0.35406 | 0.00170521 | 0.352354 | -0.0256228 |
| 11 | 12 | 3.0 | 0.359873 | 0.00169515 | 0.358178 | -0.0197992 |
| 12 | 5 | 1.0 | 0.362075 | 0.00175227 | 0.360323 | -0.0176546 |

## Interpretation

1. Early layers dominate the local steering effect. Layers 0–7 contain most of the large positive and negative deltas.
2. The best direction is `alpha=-3` at layer 4, not `alpha=+3`. This means the useful local axis may be opposite to the naive `verify - instant` vector under the current token-mass metric, or the prompt contrast entangles formatting and effort.
3. The effect is directional: layer 4 at `alpha=-3` increases the gap by +0.1153, while nearby layers 5–7 at `alpha=+3` reduce the gap by -0.0480 to -0.0776.
4. This supports continuing with controlled generation tests, but only after adding random-direction and verbosity controls.

## Verification

- `python -m pytest -q` passed: 8 tests.
- Result artifact exists: `outputs/prompt_steering_verify_n8_all_layers.json`.
- Human-readable result exists: `outputs/prompt_steering_verify_n8_all_layers.md`.
- Resource report exists: `docs/resource_profile.md`.

## Next Experiments

1. Reproduce n=8 with `high-mode=cot` and `high-mode=budget_forcing` to test whether layer-4 negative steering is specific to `verify` prompts.
2. Add random direction controls at layer 4 with matched vector norm.
3. Run short autoregressive generation only for the small candidate set: layer 4 alpha -3, layer 0 alpha -3, layer 7 alpha +3, and alpha 0.
4. Measure answer correctness and verbosity/repetition before claiming useful reasoning effort.
