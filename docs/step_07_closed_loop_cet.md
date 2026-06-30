# Step 7: Closed-Loop CET Controller

## Requirement

`second_prompt.md` asked to evolve one-shot prefix patching into a closed-loop controller:

```text
for each generated token:
  read effort_score / final_margin / repetition_score
  if model is trying to Final too early: push toward reasoning posture
  if model has valid final answer: stop
  if repetition rising: reduce steering or push away from repetition mode
```

The controller should not blindly apply a constant intervention. Earlier steps showed why:

- Step 2/3: first-final utility is not established.
- Step 4: posture signal includes format/verbosity components.
- Step 5: token metrics are prompt-artifact sensitive.
- Step 6: process-direction interventions suppress target answer availability.

## Implementation

Added `src/closed_loop_cet.py` and `tests/test_closed_loop_cet.py`.

Key features:

- Computes a mean process direction from train rows:

```text
v_process = mean(h_verify - h_instant)
```

- Runs greedy token-level generation with KV cache.
- Reads unsteered next-token signals at each step:
  - continue mass;
  - final mass;
  - repetition rate.
- Chooses alpha dynamically:
  - full alpha if final mass dominates before a minimum reasoning budget;
  - half alpha during early reasoning if continuation is already plausible;
  - zero alpha after repetition threshold;
  - reduced alpha after minimum reasoning budget.
- Stops once strict `Final answer: <number>` or `Final: <number>` appears.

Controllers compared:

| controller | behavior |
| --- | --- |
| `baseline` | no intervention |
| `static_mean_direction` | constant mean-direction addition |
| `closed_loop_cet` | dynamic alpha policy |

## Command

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m src.closed_loop_cet \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --train-dataset synthetic_math \
  --train-n 8 \
  --eval-dataset heldout_synthetic_math \
  --eval-n 3 \
  --low-mode instant \
  --high-mode verify \
  --layer 18 \
  --controllers baseline,static_mean_direction,closed_loop_cet \
  --alpha-max 1 \
  --min-reasoning-tokens 16 \
  --repetition-threshold 0.18 \
  --max-new-tokens 64 \
  --out outputs/closed_loop_cet_n3_layer18.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Results

Artifacts:

- `outputs/closed_loop_cet_n3_layer18.json`
- `outputs/closed_loop_cet_n3_layer18.md`

Summary:

| controller | n | accuracy | strict stop rate | mean tokens | repetition | mean alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | `0.000` | `0.000` | `64.00` | `0.0000` | `0.000` |
| static_mean_direction | 3 | `0.000` | `0.000` | `64.00` | `0.0000` | `1.000` |
| closed_loop_cet | 3 | `0.000` | `0.000` | `64.00` | `0.0000` | `0.275` |

Qualitative output inspection shows severe degeneration under static mean-direction steering and reduced but still degraded text under closed-loop CET. The controller successfully lowers mean alpha relative to static steering, but it does not produce valid strict finals or correct answers.

## Interpretation

Supported:

1. A real token-level loop now exists.
2. The controller reads live generation signals and changes alpha dynamically.
3. KV-cache generation makes the loop practical enough for CPU smoke experiments.
4. Closed-loop CET reduces steering strength compared with static steering.

Not supported:

1. CET does not improve accuracy in the n=3 run.
2. CET does not produce valid `Final answer:` stops.
3. Mean-direction steering causes degenerate token patterns such as repeated angle/placeholder tokens.
4. The current effort signal is not safe enough to be applied as a raw additive control vector during generation.

The clean conclusion is:

> Closed-loop control infrastructure is implemented, but the current control signal is not yet a useful controller. It needs stronger process/task disentanglement, better anti-format-artifact metrics, and safer alpha calibration before claiming CET utility.

## Why This Negative Result Matters

This failure is consistent with Steps 4–6 rather than surprising:

- Step 4 showed the signal is not purely reasoning-specific.
- Step 5 showed token metrics are artifact-prone.
- Step 6 showed mean process direction worsens target answer rank.

Therefore static steering and simple closed-loop alpha scheduling are expected to be brittle. The right next controller should use a validated hidden-state probe or behavioral verifier, not just raw `verify - instant` mean direction.

## Verification

```bash
python -m src.closed_loop_cet --help
python -m pytest -q
```

Result: `24 passed`.
