# Third Stage: n=50 Layer27 Heldout Evaluation

## Requirement

`third_prompt.md` explicitly requested moving the heldout evaluation beyond n=5:

```text
heldout_synthetic_math n=50
conditions:
  baseline
  layer27_t0.6
  layer27_t0.7 / 0.75
  layer27_t0.9
  layer27_t1.0
report Wilson CI, paired bootstrap, McNemar/exact test
```

Because prior audits showed layer27 `outputs.hidden_states` mismatch, this run uses `src.stop_after_final_experiment`, which captures source states at the decoder-block hook site before patching.

## Command

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m src.stop_after_final_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset heldout_synthetic_math \
  --n 50 \
  --low-mode instant \
  --high-mode verify \
  --conditions 27:0.6,27:0.75,27:0.9,27:1 \
  --max-new-tokens 96 \
  --out outputs/stop_after_final_heldout_n50_layer27.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4

python -m src.paired_stats \
  --input outputs/stop_after_final_heldout_n50_layer27.json \
  --out outputs/stop_after_final_stats_n50_layer27.json \
  --bootstrap 10000 \
  --seed 123
```

## Artifacts

- `outputs/stop_after_final_heldout_n50_layer27.json`
- `outputs/stop_after_final_heldout_n50_layer27.md`
- `outputs/stop_after_final_stats_n50_layer27.json`
- `outputs/stop_after_final_stats_n50_layer27.md`

## Main Result

Strict first-final accuracy is still zero for every condition:

| condition | n | first-final acc | Wilson 95% CI | diff vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 50 | 0.000 | [0.000, 0.071] | +0.000 |
| layer27_t0.6 | 50 | 0.000 | [0.000, 0.071] | +0.000 |
| layer27_t0.75 | 50 | 0.000 | [0.000, 0.071] | +0.000 |
| layer27_t0.9 | 50 | 0.000 | [0.000, 0.071] | +0.000 |
| layer27_t1 | 50 | 0.000 | [0.000, 0.071] | +0.000 |

This means the strongest possible task-utility claim remains unsupported under first-commit evaluation.

## Auxiliary Last-Answer Result

The weaker last-detectable-answer metric shows a strong positive signal for partial layer27 interpolation:

| condition | n | last-answer acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | + flips | - flips | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 50 | 0.280 | [0.175, 0.417] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.6 | 50 | 0.660 | [0.522, 0.776] | +0.380 | [+0.240, +0.520] | 19 | 0 | 0.000 |
| layer27_t0.75 | 50 | 0.660 | [0.522, 0.776] | +0.380 | [+0.240, +0.520] | 19 | 0 | 0.000 |
| layer27_t0.9 | 50 | 0.660 | [0.522, 0.776] | +0.380 | [+0.240, +0.520] | 19 | 0 | 0.000 |
| layer27_t1 | 50 | 0.320 | [0.208, 0.458] | +0.040 | [-0.140, +0.220] | 12 | 10 | 0.832 |

Important: this is not first-commit correctness. It remains vulnerable to post-marker drift and malformed `Answer:` outputs.

## Protocol Quality

| condition | strict stop rate | strict malformed rate | mean tokens | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.260 | 0.740 | 78.08 | 0.1062 |
| layer27_t0.6 | 0.260 | 0.740 | 75.60 | 0.0515 |
| layer27_t0.75 | 0.260 | 0.740 | 75.60 | 0.0515 |
| layer27_t0.9 | 0.260 | 0.740 | 75.60 | 0.0515 |
| layer27_t1 | 0.240 | 0.760 | 77.16 | 0.1414 |

The malformed rate remains around 74–76%, so the protocol failure is not solved by scaling n.

## Additional Observation

For this deterministic CPU run, `layer27_t0.6`, `layer27_t0.75`, and `layer27_t0.9` produced identical completions on all 50 problems. This suggests a saturated basin under the current prompt and greedy decoding: once the layer27 interpolation crosses a threshold, the same trajectory is selected.

## Interpretation

This larger run sharpens the project conclusion:

1. The last-answer positive signal is real enough to survive n=50 paired statistics.
2. The first-final utility claim remains false in this protocol.
3. The intervention often makes the model write a correct `Answer:` number somewhere, but not a valid first `Final answer:` commitment.
4. The next bottleneck is not sample size; it is protocol/format binding and answer-commitment control.

The correct claim is therefore:

> Layer27 hook-state interpolation can improve last-detectable answer accuracy in n=50 heldout arithmetic, but it does not improve strict first-final accuracy. This is a posture/trajectory signal, not yet a reliable final-answer controller.

## Verification

```bash
python -m pytest -q
```

Final verification in this stage should pass before commit.
