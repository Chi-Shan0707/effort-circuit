# Step 2: Stop-After-First-Final Generation

## Requirement

`second_prompt.md` identified a major evaluation flaw in the earlier strict-generation experiment: completions were allowed to continue after answer markers, and answer extraction preferred the last detectable answer. That can inflate accuracy when the model first gives an invalid or wrong final answer and only later drifts into a correct number.

From first principles, an answer should be evaluated at the point where the policy commits. For an autoregressive model, the causal question is therefore not only "does an intervention change text somewhere in a long completion?" but also:

1. Does it make the model emit a valid final answer in the required protocol?
2. Is the first committed final answer correct?
3. Does the effect survive when we prevent post-final self-correction or drift?

## Implementation

Added `src/stop_after_final_experiment.py`.

Key design choices:

- Uses hook-captured decoder block states for source activations, not `outputs.hidden_states`, because Step 1 showed that layer 27 hidden-state patching is not site-equivalent to the decoder block hook output.
- Generates with a `StoppingCriteria` that halts once a strict final marker appears.
- Treats only `Final answer: <number>` or `Final: <number>` as a first committed final answer.
- Intentionally ignores generic `Answer: <number>` for stop-after-first-final, because the strict protocol in `second_prompt.md` requires `Final answer:` and the model often emits `Answer: 1` as a malformed numbered placeholder.
- Records first-final accuracy, last-answer accuracy, stop rate, generated tokens, answer changes, and repetition rate.

Additional regression tests were added in `tests/test_stop_after_final_experiment.py` to ensure generic `Answer: 1` is not treated as a strict final answer.

## Command

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m src.stop_after_final_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset heldout_synthetic_math \
  --n 5 \
  --low-mode instant \
  --high-mode verify \
  --conditions 18:1,27:0.75,27:1 \
  --max-new-tokens 128 \
  --out outputs/stop_after_final_heldout_n5_hook_states.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Results

Artifact files:

- `outputs/stop_after_final_heldout_n5_hook_states.json`
- `outputs/stop_after_final_heldout_n5_hook_states.md`

Summary on heldout n=5:

| condition | first-final acc | last-answer acc | strict stop rate | mean tokens | answer changes | repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 0.400 | 0.200 | 105.40 | 0.00 | 0.0800 |
| layer18_t1 | 0.000 | 0.000 | 0.600 | 74.00 | 0.00 | 0.0466 |
| layer27_t0.75 | 0.000 | 0.600 | 0.200 | 105.80 | 0.00 | 0.0251 |
| layer27_t1 | 0.000 | 0.200 | 0.200 | 105.80 | 0.00 | 0.0861 |

The most important result is negative: **all tested conditions have 0/5 strict first-final accuracy**.

The earlier heldout-small result had suggested that layer 27 patching improved extracted final accuracy. Under stop-after-first-final evaluation, that claim does not currently survive. Many completions contain correct arithmetic later in the text, but fail to emit the required `Final answer: <number>` line before the token budget or emit malformed final placeholders such as `Final answer: 1`.

## Interpretation

This does not falsify the local activation-patching result. Step 1 and earlier patching/interpolation experiments still show that late residual states causally alter next-token continue-vs-final posture. However, Step 2 sharply narrows the claim:

- Supported: late hook-captured states can change generation style, continuation posture, and sometimes reduce repetition.
- Not supported yet: the intervention reliably improves first committed answer correctness under a strict answer protocol.
- Contradicted at n=5: the previous optimistic `last extracted answer` accuracy should not be treated as a clean task-utility gain.

A useful mechanistic interpretation is that the intervention appears to affect **process posture** more than **task-solvedness**. The model may continue, verbalize, or drift into correct arithmetic, but it does not robustly bind that computation to the required final-answer action.

## Consequences For The Roadmap

Step 3 should not simply scale the old strict-generation metric. It should scale the stricter metrics introduced here:

1. first strict final accuracy;
2. last detectable answer accuracy, explicitly marked as a weaker auxiliary metric;
3. strict stop rate;
4. malformed-answer rate, especially generic `Answer:` and placeholder `Final answer: 1` patterns;
5. paired comparisons with Wilson intervals and exact paired counts.

Step 4 controls become more important after this result. The next experiments must distinguish:

- useful reasoning improvement;
- generic verbosity;
- format-following failure;
- answer-marker drift;
- task-content leakage from patch source states.

## Verification

```bash
python -m src.stop_after_final_experiment --help
python -m pytest -q
```

Result: `12 passed`.

## Current Conclusion

Step 2 is valuable because it prevents a false positive. The clean conclusion is no longer "layer 27 improves heldout accuracy". The clean conclusion is:

> Hook-captured late-layer interventions still affect generation dynamics, but under first-commit evaluation they do not yet produce reliable correct final answers. The project should now pivot from celebrating accuracy gains to isolating why process-posture control fails to become correct answer commitment.

## Closure Note

The implementation now reports first strict final answer, last detectable answer as an auxiliary weak metric, strict stop rate, malformed rate, generated tokens, repetition, and answer changes. The default protocol remains strict: only `Final answer:` or `Final:` count as first-final markers. Generic `Answer:` is ignored by default and is accepted only when the optional `--relaxed-final-markers` flag is passed.
